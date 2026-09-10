"""Three-stage trainer for the BlackFeather Local-SNR architecture."""

from __future__ import annotations

import json
import logging
import math
import shutil
import time
from pathlib import Path
from typing import Any, Dict, Sequence

import numpy as np
import torch
from tqdm import tqdm

from config import TrainConfig
from utils import BlackFeatherMultiTaskLoss, EarlyStopping
from utils.evaluate import aggregate_windows, compute_metrics, compute_snr_band_metrics
from utils.history_logger import plot_confusion_matrix, save_confusion_matrix
from utils.losses import si_sdr

logger = logging.getLogger(__name__)


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    return value


class LocalSNRTrainer:
    def __init__(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        device: torch.device,
        config: TrainConfig,
        checkpoint_directory: str,
        label_names: Sequence[str],
        train_config_path: str,
    ) -> None:
        self.model = model
        self.optimizer = optimizer
        self.device = device
        self.config = config
        self.objective = BlackFeatherMultiTaskLoss(config.multitask_loss).to(device)
        self.checkpoint_directory = Path(checkpoint_directory)
        self.checkpoint_directory.mkdir(parents=True, exist_ok=True)
        self.label_names = list(label_names)
        self.base_learning_rates = [group["lr"] for group in optimizer.param_groups]
        self.snr_bands = [
            (band.name, band.min_db, band.max_db) for band in config.snr_bands
        ]
        config_path = Path(train_config_path)
        stored_config = self.checkpoint_directory / "train_config.json"
        # Re-scoring a finished run means passing the very file this would write.
        if config_path.is_file() and config_path.resolve() != stored_config.resolve():
            shutil.copy2(config_path, stored_config)
        (self.checkpoint_directory / "labels.json").write_text(
            json.dumps(self.label_names, indent=2, ensure_ascii=False), encoding="utf-8"
        )

    def _stage_plan(self) -> list[tuple[str, int]]:
        stages = self.config.stages
        return [
            ("extractor", stages.extractor_epochs),
            ("heads", stages.heads_epochs),
            ("joint", stages.joint_epochs),
        ]

    def _set_stage(self, stage: str) -> None:
        self.model.set_stage(stage)
        scale = self.config.stages.joint_lr_scale if stage == "joint" else 1.0
        for group, base_lr in zip(self.optimizer.param_groups, self.base_learning_rates):
            group["lr"] = base_lr * scale
        trainable = sum(parameter.numel() for parameter in self.model.parameters() if parameter.requires_grad)
        logger.info("Stage %s: %d trainable parameters, lr scale %.3f", stage, trainable, scale)

    def _targets_to_device(self, batch: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        return {
            key: batch[key].to(self.device, non_blocking=True)
            for key in (
                "target",
                "noise_waveform",
                "local_snr_db",
                "local_snr_mask",
            )
        }

    def _forward_training(
        self,
        waveform: torch.Tensor,
        target: Dict[str, torch.Tensor],
        stage: str,
    ) -> Dict[str, torch.Tensor]:
        if stage == "extractor":
            return {"estimated_noise": self.model.noise_extractor(waveform)}
        oracle = None
        if stage == "heads" and torch.rand((), device=waveform.device) < self.config.stages.oracle_noise_probability:
            oracle = target["noise_waveform"]
        return self.model(waveform, noise_reference=oracle)

    def _train_epoch(self, loader: Any, stage: str, epoch: int) -> Dict[str, float]:
        self.model.train()
        totals: Dict[str, float] = {}
        sample_count = 0
        progress = tqdm(loader, desc=f"{stage} epoch {epoch}", unit="batch", dynamic_ncols=True)
        for batch in progress:
            waveform = batch["waveform"].to(self.device, non_blocking=True)
            target = self._targets_to_device(batch)
            output = self._forward_training(waveform, target, stage)
            losses = self.objective(output, target, stage=stage)
            self.optimizer.zero_grad(set_to_none=True)
            losses["loss"].backward()
            torch.nn.utils.clip_grad_norm_(
                [parameter for parameter in self.model.parameters() if parameter.requires_grad],
                max_norm=5.0,
            )
            self.optimizer.step()
            batch_size = waveform.shape[0]
            sample_count += batch_size
            for name, value in losses.items():
                totals[name] = totals.get(name, 0.0) + float(value.detach()) * batch_size
            progress.set_postfix(loss=f"{float(losses['loss'].detach()):.4f}")
        return {name: value / max(sample_count, 1) for name, value in totals.items()}

    @torch.no_grad()
    def evaluate(self, loader: Any, stage: str = "joint") -> Dict[str, Any]:
        self.model.eval()
        sample_ids: list[str] = []
        probabilities: list[np.ndarray] = []
        targets: list[np.ndarray] = []
        global_snrs: list[np.ndarray] = []
        local_absolute_errors: list[np.ndarray] = []
        local_squared_errors: list[np.ndarray] = []
        separation_scores: list[np.ndarray] = []
        loss_totals: Dict[str, float] = {}
        sample_count = 0
        for batch in tqdm(loader, desc="Evaluating", unit="batch", dynamic_ncols=True):
            waveform = batch["waveform"].to(self.device, non_blocking=True)
            target = self._targets_to_device(batch)
            output = (
                {"estimated_noise": self.model.noise_extractor(waveform)}
                if stage == "extractor"
                else self.model(waveform)
            )
            losses = self.objective(output, target, stage=stage)
            batch_size = waveform.shape[0]
            for name, value in losses.items():
                loss_totals[name] = loss_totals.get(name, 0.0) + float(value) * batch_size
            sample_count += batch_size
            separation_scores.append(
                si_sdr(output["estimated_noise"], target["noise_waveform"]).cpu().numpy()
            )
            if stage != "extractor":
                sample_ids.extend(list(batch["audio_name"]))
                probabilities.append(torch.softmax(output["clipwise_output"], dim=-1).cpu().numpy())
                targets.append(target["target"].cpu().numpy())
                global_snrs.append(batch["target_snr_db"].cpu().numpy())
                mask = target["local_snr_mask"].bool()
                error = output["local_snr_output"] - target["local_snr_db"]
                local_absolute_errors.append(error[mask].abs().cpu().numpy())
                local_squared_errors.append(error[mask].square().cpu().numpy())

        averaged_losses = {
            name: value / max(sample_count, 1) for name, value in loss_totals.items()
        }
        mean_separation_score = float(np.concatenate(separation_scores).mean())
        if stage == "extractor":
            return {
                **averaged_losses,
                "top1_accuracy": float("nan"),
                "f1_macro": float("nan"),
                "local_snr_mae_db": float("nan"),
                "local_snr_rmse_db": float("nan"),
                "noise_si_sdr_db": mean_separation_score,
            }

        window_probability = np.concatenate(probabilities)
        window_target = np.concatenate(targets)
        window_snr = np.concatenate(global_snrs)
        clip_ids, clip_probability, clip_target, clip_snr = aggregate_windows(
            sample_ids,
            window_probability,
            window_target,
            window_snr,
        )
        metrics = compute_metrics(clip_target, clip_probability, self.label_names)
        absolute_error = np.concatenate(local_absolute_errors)
        squared_error = np.concatenate(local_squared_errors)
        metrics.update(
            {
                **averaged_losses,
                "sample_ids": clip_ids,
                "target_snr_db": clip_snr,
                "num_clips": len(clip_ids),
                "num_windows": len(sample_ids),
                "local_snr_mae_db": float(absolute_error.mean()) if absolute_error.size else float("nan"),
                "local_snr_rmse_db": float(math.sqrt(squared_error.mean())) if squared_error.size else float("nan"),
                "noise_si_sdr_db": mean_separation_score,
                "snr_metrics": compute_snr_band_metrics(
                    clip_target,
                    clip_probability,
                    clip_snr,
                    self.label_names,
                    self.snr_bands,
                ),
            }
        )
        return metrics

    def _history_row(
        self,
        stage: str,
        epoch: int,
        train_losses: Dict[str, float],
        validation: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Keep every validation number, not just the five the console prints.

        mAP, macro-AUC, top-3 and the per-band breakdown are computed every
        epoch and were being discarded, which left no curve to plot afterwards.
        The extractor stage has no classifier, so those keys are absent there
        rather than wrong, and read back as null.
        """
        missing = float("nan")
        row: Dict[str, Any] = {
            "stage": stage,
            "epoch": epoch,
            **{f"train_{key}": value for key, value in train_losses.items()},
            "val_loss": validation["loss"],
            "val_top1": validation["top1_accuracy"],
            "val_top3": validation.get("top3_accuracy", missing),
            "val_balanced_accuracy": validation.get("balanced_accuracy", missing),
            "val_macro_f1": validation["f1_macro"],
            "val_mAP": validation.get("mAP", missing),
            "val_macro_auc": validation.get("macro_auc", missing),
            "val_local_snr_mae_db": validation["local_snr_mae_db"],
            "val_local_snr_rmse_db": validation.get("local_snr_rmse_db", missing),
            "val_noise_si_sdr_db": validation["noise_si_sdr_db"],
        }
        row["val_snr_bands"] = {
            name: {"top1": values["top1_accuracy"], "macro_f1": values["macro_f1"]}
            for name, values in (validation.get("snr_metrics") or {}).items()
        }
        return row

    def _append_history(self, row: Dict[str, Any]) -> None:
        """Flush one epoch as soon as it finishes.

        The rows used to accumulate in memory until all three stages and the
        test pass had completed, so a job killed in its final epoch lost every
        epoch it had already paid for. One open per epoch costs nothing beside
        an epoch of training.
        """
        path = self.checkpoint_directory / "history.jsonl"
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(_json_safe(row), ensure_ascii=False) + "\n")
            handle.flush()

    def _checkpoint_path(self, stage: str) -> Path:
        return self.checkpoint_directory / {
            "extractor": "stage1_extractor.pt",
            "heads": "stage2_heads.pt",
            "joint": "stage3_joint.pt",
        }[stage]

    def _save_checkpoint(self, stage: str, epoch: int, score: float) -> Path:
        path = self._checkpoint_path(stage)
        torch.save(
            {
                "stage": stage,
                "epoch": epoch,
                "score": score,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "label_names": self.label_names,
                "local_snr_frames": self.model.local_snr_frames,
            },
            path,
        )
        logger.info("Saved %s checkpoint to %s", stage, path)
        return path

    def train(self, train_loader: Any, val_loader: Any, test_loader: Any) -> Dict[str, Any]:
        started = time.perf_counter()
        (self.checkpoint_directory / "history.jsonl").write_text("", encoding="utf-8")
        final_checkpoint: Path | None = None
        for stage, epochs in self._stage_plan():
            if epochs <= 0:
                continue
            self._set_stage(stage)
            best_score = float("-inf")
            # A fresh stopper per stage: the three stages optimise different
            # objectives, so patience accrued under one says nothing about the
            # next. Both scores below are higher-is-better, which is what
            # EarlyStopping expects.
            stopper = (
                EarlyStopping(
                    patience=self.config.patience,
                    delta=self.config.delta,
                    verbose=True,
                )
                if self.config.early_stopping
                else None
            )
            for epoch in range(1, epochs + 1):
                train_losses = self._train_epoch(train_loader, stage, epoch)
                validation = self.evaluate(val_loader, stage=stage)
                score = (
                    -float(validation["separation_loss"])
                    if stage == "extractor"
                    else float(validation["f1_macro"])
                )
                if score > best_score:
                    best_score = score
                    final_checkpoint = self._save_checkpoint(stage, epoch, score)
                self._append_history(
                    self._history_row(stage, epoch, train_losses, validation)
                )
                logger.info(
                    "%s %d/%d | val top1 %.4f macro-F1 %.4f | SNR MAE %.3f dB | noise SI-SDR %.3f dB",
                    stage,
                    epoch,
                    epochs,
                    validation["top1_accuracy"],
                    validation["f1_macro"],
                    validation["local_snr_mae_db"],
                    validation["noise_si_sdr_db"],
                )
                if stopper is not None and stopper.step(score):
                    logger.info(
                        "Early stopping %s after epoch %d of %d", stage, epoch, epochs
                    )
                    break
            if final_checkpoint is not None and final_checkpoint == self._checkpoint_path(stage):
                checkpoint = torch.load(
                    final_checkpoint, map_location=self.device, weights_only=False
                )
                self.model.load_state_dict(checkpoint["model_state_dict"], strict=True)

        if final_checkpoint is None:
            raise RuntimeError("No training stage ran and no checkpoint was produced")
        checkpoint = torch.load(final_checkpoint, map_location=self.device, weights_only=False)
        self.model.load_state_dict(checkpoint["model_state_dict"], strict=True)
        test_metrics = self.evaluate(test_loader)
        # summary.json holds the matrix as numbers; these two make it readable.
        save_confusion_matrix(
            self.checkpoint_directory, self.label_names, test_metrics,
            "confusion_matrix_test.csv",
        )
        plot_confusion_matrix(
            self.checkpoint_directory, self.label_names, test_metrics,
            "confusion_matrix_test.png", "Test confusion matrix",
        )
        summary = {
            "training_time_seconds": time.perf_counter() - started,
            "final_checkpoint": str(final_checkpoint),
            "test": test_metrics,
        }
        (self.checkpoint_directory / "summary.json").write_text(
            json.dumps(_json_safe(summary), indent=2, ensure_ascii=False), encoding="utf-8"
        )
        return summary

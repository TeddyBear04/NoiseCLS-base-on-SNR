from __future__ import annotations

import contextlib
import copy
import logging
import os
from collections import defaultdict
from collections.abc import Callable
from pathlib import Path
from typing import Any, ClassVar

import numpy as np
import torch
from torch import Tensor
from torch.optim import AdamW
from torch.utils.data import DataLoader

from src.core import BaseModel
from src.loss import KDMoELoss, KDMoELossConfig
from src.metrics import ClassificationMetrics
from src.model.frcrn import FRCRNSeparator
from src.model.kdmoe import KDMoEModel, frcrn_snr_db
from src.utils.ema import EMA

from .base import StepOutput, Trainer
from .beat_trainer import cosine_with_warmup
from .config import KDMoETrainConfig

STAGES = (1, 2)
# stage 1 chỉ học cổng + head phụ (+ patch/khối lai với ViT); thân PANNs pretrained (encoder.,
# post_fc.) giữ nguyên ở stage 1 rồi mới fine-tune ở stage 2, theo tinh thần LP-FT (Kumar et al., ICLR 2022)
STAGE1_FROZEN_PREFIXES = ("vit.", "final_ln.", "pool_gru.", "router.", "experts.", "encoder.", "post_fc.")
NO_DECAY_MARKERS = ("bias", "ln", "norm", "pos", "a_att", "a_gru")


class KDMoETrainer(Trainer):
    """Hai giai đoạn: (1) L_front cho patch/khối lai/cổng/head phụ, (2) toàn bộ loss."""

    selection_metric: ClassVar[str] = "macro_acc"

    def __init__(
        self,
        model: BaseModel,
        config: KDMoETrainConfig,
        loss_config: KDMoELossConfig,
        membership: Tensor,
        device: torch.device,
        logger: logging.Logger,
        name: str,
        checkpoint_path: str | Path,
        stage: int = 2,
        resume_path: str | Path | None = None,
        drive_backup: Callable[[str | Path], None] | None = None,
        on_checkpoint: Callable[[Path, int, float], None] | None = None,
        separator: FRCRNSeparator | None = None,
    ) -> None:
        super().__init__(model, config, device, logger, name, checkpoint_path, on_checkpoint)
        if stage not in STAGES:
            raise ValueError(f"stage must be one of {STAGES}, got {stage}")
        if not isinstance(model, KDMoEModel):
            raise TypeError(f"KDMoETrainer expects a KDMoEModel, got {type(model).__name__}")
        self.stage = stage
        self.criterion = KDMoELoss(loss_config, membership)
        self.separator = separator.to(device).freeze() if separator is not None else None
        self.ema = EMA(model, config.ema_decay)
        self.selection_metric = "macro_acc" if stage == 2 else "front_score"
        self.resume_path = Path(resume_path) if resume_path else None
        self.drive_backup = drive_backup
        self._completed_epochs = 0
        self._current_epoch = 0
        self._in_fit = False
        self._history: list[dict[str, Any]] = []
        self._resume_state: dict[str, Any] | None = None
        self._resumed_best_state: dict[str, Tensor] | None = None
        self._apply_stage()
        self._reset_train_stats()
        self._peak_memory_reported = False

    @property
    def num_classes(self) -> int:
        return self.model.config.num_classes

    def _apply_stage(self) -> None:
        for name, parameter in self.model.named_parameters():
            frozen = self.stage == 1 and name.startswith(STAGE1_FROZEN_PREFIXES)
            parameter.requires_grad_(not frozen)

    def _reset_train_stats(self) -> None:
        self._sums: defaultdict[str, float] = defaultdict(float)
        self._seen = 0
        self._correct = 0
        self._router_correct = 0

    def configure_optimizer(self, steps_per_epoch: int) -> None:
        decay, no_decay = [], []
        for name, parameter in self.model.named_parameters():
            if not parameter.requires_grad:
                continue
            if any(marker in name for marker in NO_DECAY_MARKERS):
                no_decay.append(parameter)
            else:
                decay.append(parameter)
        self.optimizer = AdamW(
            [
                {"params": decay, "weight_decay": self.config.weight_decay},
                {"params": no_decay, "weight_decay": 0.0},
            ],
            lr=self.config.lr,
            betas=tuple(self.config.betas),
        )
        total_steps = self.config.epochs * steps_per_epoch
        self.scheduler = cosine_with_warmup(
            self.optimizer, total_steps, self.config.warmup_ratio, self.config.min_lr / self.config.lr
        )
        self._apply_resume_state()

    def _apply_resume_state(self) -> None:
        if self._resume_state is None:
            return
        self.optimizer.load_state_dict(self._resume_state["optimizer"])
        if self.scheduler is not None and self._resume_state.get("scheduler") is not None:
            self.scheduler.load_state_dict(self._resume_state["scheduler"])
        devices = {name: parameter.device for name, parameter in self.model.named_parameters()}
        for name, value in self._resume_state["ema"].items():
            if name in self.ema.shadow:
                self.ema.shadow[name] = value.to(devices.get(name, value.device))
        self._resume_state = None
        self.logger.info("resume: optimizer/scheduler/EMA khôi phục từ epoch %d", self._completed_epochs)

    def save_resume(self, path: str | Path | None = None) -> Path:
        if self.optimizer is None or self._current_epoch == 0:
            raise ValueError("configure_optimizer and run_epoch must run before save_resume")
        path = Path(path) if path else self.resume_path
        if path is None:
            raise ValueError("resume_path is not set")
        payload = {
            "stage": self.stage,
            "epoch": self._current_epoch,
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict() if self.scheduler is not None else None,
            "ema": self.ema.shadow,
            "best_score": self._best_score,
            "best_epoch": self._best_epoch,
            "best_state": copy.deepcopy(self._best_state) if self._best_state is not None else None,
            "history": self._history,
        }
        partial = path.with_name(path.name + ".part")
        torch.save(payload, partial)
        os.replace(partial, path)
        return path

    def load_resume(self, path: str | Path) -> None:
        data = torch.load(path, map_location="cpu", weights_only=False)
        if int(data.get("stage", self.stage)) != self.stage:
            raise ValueError(f"resume stage {data['stage']} does not match trainer stage {self.stage}")
        self.model.load_state_dict(data["model"])
        self._completed_epochs = int(data["epoch"])
        self._best_score = float(data.get("best_score", -float("inf")))
        self._best_epoch = int(data.get("best_epoch", 0))
        self._history = list(data.get("history", []))
        self._resumed_best_state = data.get("best_state")
        self._resume_state = data
        self.logger.info(
            "resume: stage %d từ epoch %d (best %.4f @ %d)",
            self.stage,
            self._completed_epochs,
            self._best_score,
            self._best_epoch,
        )

    @property
    def history(self) -> list[dict[str, Any]]:
        return self._history

    def snapshot(self) -> dict[str, Tensor]:
        if self._resumed_best_state is not None:
            state, self._resumed_best_state = copy.deepcopy(self._resumed_best_state), None
            return state
        state = super().snapshot()
        if self.stage == 2:
            for name, value in self.ema.shadow.items():
                if name in state:
                    state[name] = value.detach().clone()
        return state

    def run_epoch(self, loader: DataLoader, epoch: int) -> dict[str, float]:
        if epoch <= self._completed_epochs:
            return {}
        self._current_epoch = epoch
        self._in_fit = True
        return super().run_epoch(loader, epoch)

    def _persist_resume(self) -> None:
        if self.resume_path is None:
            return
        self.save_resume(self.resume_path)
        if self.drive_backup is not None and self._current_epoch % self.config.resume_every == 0:
            self.drive_backup(self.resume_path)

    def _autocast(self) -> contextlib.AbstractContextManager:
        if self.device.type == "cuda" and self.config.amp == "bf16":
            return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        return contextlib.nullcontext()

    def _to_device(self, batch: dict[str, Any]) -> dict[str, Any]:
        return {key: value.to(self.device) if isinstance(value, Tensor) else value for key, value in batch.items()}

    def _student_waveform(self, batch: dict[str, Any]) -> Tensor:
        waveform = batch["waveform"]
        if self.separator is not None:
            with torch.no_grad():
                waveform = self.separator(waveform)
                batch["snr_est"] = frcrn_snr_db(batch["waveform"], waveform)  # route_source frcrn_snr
        return waveform

    def begin_epoch(self) -> None:
        self._reset_train_stats()

    def training_step(self, batch: dict[str, Any]) -> StepOutput:
        batch = self._to_device(batch)
        noise = batch.get("noise")
        with self._autocast():
            waveform = self._student_waveform(batch)
            output = self.model(waveform, noise, batch.get("snr_est"))
        terms = self.criterion(output, batch)
        if self.stage == 1:
            if "front" not in terms:
                raise KeyError("stage 1 requires the front loss, load noise in the dataset")
            loss = terms["front"]
        else:
            loss = terms["total"]
        with torch.no_grad():
            labels = batch["label"]
            self._correct += (output.logits.argmax(dim=-1) == labels).sum().item()
            self._router_correct += (output.router_logits.argmax(dim=-1) == batch["group"]).sum().item()
            self._seen += labels.numel()
            self._sums["gate_mean"] += output.gate.mean().item() * labels.numel()
            self._sums["gate_low"] += (output.gate < 0.2).float().mean().item() * labels.numel()
            for key in ("ce", "kd", "route_ce", "at", "hint", "front", "cons"):
                if key in terms:
                    self._sums[key] += terms[key].item() * labels.numel()
        return StepOutput(loss=loss, size=labels.numel())

    def observe(self, step: StepOutput) -> None:
        self.ema.update(self.model)

    def epoch_summary(self) -> dict[str, float]:
        seen = max(self._seen, 1)
        summary = {
            "acc": self._correct / seen,
            "router_acc": self._router_correct / seen,
            "gate_mean": self._sums["gate_mean"] / seen,
            "gate_low": self._sums["gate_low"] / seen,
        }
        for key in ("ce", "kd", "route_ce", "at", "hint", "front", "cons"):
            if key in self._sums:
                summary[key] = self._sums[key] / seen
        if self.device.type == "cuda" and not self._peak_memory_reported:
            self.logger.info("peak memory: %.2f GB", torch.cuda.max_memory_allocated(self.device) / 2**30)
            self._peak_memory_reported = True
        return summary

    @torch.no_grad()
    def evaluate(self, loader: DataLoader, desc: str = "evaluate") -> dict[str, float]:
        self.model.eval()
        with self.ema.swap(self.model), self._autocast():
            probabilities, router_predictions = [], []
            labels, groups, snrs = [], [], []
            gate_means, gate_lows, entropies, kd_values, front_values = [], [], [], [], []
            for batch in loader:
                batch = self._to_device(batch)
                waveform = self._student_waveform(batch)
                output = self.model(waveform, batch.get("noise"), batch.get("snr_est"))
                probabilities.append(output.logits.softmax(dim=-1).float().cpu())
                router_predictions.append(output.router_logits.argmax(dim=-1).cpu())
                labels.append(batch["label"].cpu())
                groups.append(batch["group"].cpu())
                snrs.append(batch["snr_db"].cpu())
                gate_means.append(output.gate.mean().item())
                gate_lows.append((output.gate < 0.2).float().mean().item())
                probs = output.router_logits.softmax(dim=-1)
                entropies.append(-(probs * probs.clamp_min(1e-8).log()).sum(dim=-1).mean().item())
                if self.criterion.config.kd_weight:
                    kd_values.append(self.criterion.kd_term(output, batch["teacher_logits36"]).item())
                if output.noise_mel is not None and output.noise_target is not None:
                    front_values.append(self.criterion.front(output.noise_mel, output.noise_target).item())
        scores = self._metrics(
            torch.cat(probabilities).numpy(),
            torch.cat(router_predictions).numpy(),
            torch.cat(labels).numpy(),
            torch.cat(groups).numpy(),
            torch.cat(snrs).numpy(),
            gate_means,
            gate_lows,
            entropies,
            kd_values,
            front_values,
        )
        if self._in_fit and self._current_epoch > 0:
            self._history.append(
                {"epoch": self._current_epoch, **{f"val_{key}": value for key, value in scores.items()}}
            )
            self._persist_resume()
        return scores

    def _metrics(
        self,
        probabilities: np.ndarray,
        router_predictions: np.ndarray,
        labels: np.ndarray,
        groups: np.ndarray,
        snrs: np.ndarray,
        gate_means: list[float],
        gate_lows: list[float],
        entropies: list[float],
        kd_values: list[float],
        front_values: list[float],
    ) -> dict[str, float]:
        scores: dict[str, float] = {}
        overall = ClassificationMetrics(num_classes=self.num_classes)
        overall.update(labels, probabilities)
        scores.update(overall.compute().to_dict())
        scores["acc"] = scores["top1_accuracy"]
        balanced_per_snr, map_per_snr, router_per_snr = [], [], []
        for snr in np.unique(snrs):
            mask = snrs == snr
            per_snr = ClassificationMetrics(num_classes=self.num_classes)
            per_snr.update(labels[mask], probabilities[mask])
            result = per_snr.compute()
            scores[f"acc_snr_{int(snr)}"] = result.top1_accuracy
            scores[f"balanced_snr_{int(snr)}"] = result.balanced_accuracy
            scores[f"mAP_snr_{int(snr)}"] = result.mAP
            router_value = float((router_predictions[mask] == groups[mask]).mean())
            scores[f"router_acc_snr_{int(snr)}"] = router_value
            balanced_per_snr.append(result.balanced_accuracy)
            map_per_snr.append(result.mAP)
            router_per_snr.append(router_value)
        if balanced_per_snr:
            scores["macro_acc"] = float(np.mean(balanced_per_snr))
            scores["mAP_macro"] = float(np.mean(map_per_snr))
        else:
            scores["macro_acc"] = scores["balanced_accuracy"]
            scores["mAP_macro"] = scores["mAP"]
        scores["router_acc"] = float((router_predictions == groups).mean())
        scores["router_macro_acc"] = float(np.mean(router_per_snr)) if router_per_snr else scores["router_acc"]
        if kd_values:
            scores["kd"] = float(np.mean(kd_values))
        scores["gate_mean"] = float(np.mean(gate_means))
        scores["gate_low"] = float(np.mean(gate_lows))
        scores["router_entropy"] = float(np.mean(entropies))
        if front_values:
            scores["front"] = float(np.mean(front_values))
            scores["front_score"] = -scores["front"]
        return scores

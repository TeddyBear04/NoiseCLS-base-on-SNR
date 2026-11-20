from __future__ import annotations

import copy
import logging
import math
import os
from collections.abc import Callable
from pathlib import Path
from typing import Any, ClassVar

import numpy as np
import torch
from torch import Tensor
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from src.core import BaseModel
from src.loss import CrossEntropyLoss
from src.metrics import ClassificationMetrics
from src.model.frcrn import FRCRNSeparator

from .base import StepOutput, Trainer
from .config import BEATTrainConfig


def cosine_with_warmup(optimizer: AdamW, total_steps: int, warmup_ratio: float, min_lr_ratio: float = 0.0) -> LambdaLR:
    warmup_steps = max(int(total_steps * warmup_ratio), 1)

    def schedule(step: int) -> float:
        if step < warmup_steps:
            return (step + 1) / warmup_steps
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        return min_lr_ratio + (1.0 - min_lr_ratio) * 0.5 * (1.0 + math.cos(math.pi * progress))

    return LambdaLR(optimizer, schedule)


class BEATTrainer(Trainer):
    selection_metric: ClassVar[str] = "macro_f1"

    def __init__(
        self,
        model: BaseModel,
        config: BEATTrainConfig,
        device: torch.device,
        logger: logging.Logger,
        name: str,
        checkpoint_path: str | Path,
        separator: FRCRNSeparator | None = None,
        resume_path: str | Path | None = None,
        drive_backup: Callable[[str | Path], None] | None = None,
        on_checkpoint: Callable[[Path, int, float], None] | None = None,
    ) -> None:
        super().__init__(model, config, device, logger, name, checkpoint_path, on_checkpoint)
        self.separator = separator.to(device).freeze() if separator is not None else None
        self.criterion = CrossEntropyLoss(config.label_smoothing)
        self._train_metrics = ClassificationMetrics(num_classes=self.num_classes)
        self._labels: Tensor | None = None
        self._logits: Tensor | None = None
        self._correct = 0
        self._seen = 0
        self.resume_path = Path(resume_path) if resume_path else None
        self.drive_backup = drive_backup
        self._completed_epochs = 0
        self._current_epoch = 0
        self._in_fit = False
        self._history: list[dict[str, Any]] = []
        self._resume_state: dict[str, Any] | None = None
        self._resumed_best_state: dict[str, Tensor] | None = None

    @property
    def history(self) -> list[dict[str, Any]]:
        return self._history

    @property
    def num_classes(self) -> int:
        return self.model.config.num_classes

    def configure_optimizer(self, steps_per_epoch: int) -> None:
        self.optimizer = AdamW(
            self.trainable_parameters,
            lr=self.config.lr,
            weight_decay=self.config.weight_decay,
            betas=tuple(self.config.betas),
        )
        total_steps = self.config.epochs * steps_per_epoch
        self.scheduler = cosine_with_warmup(
            self.optimizer, total_steps, self.config.warmup_ratio, self.config.min_lr_ratio
        )
        self._apply_resume_state()

    def _apply_resume_state(self) -> None:
        if self._resume_state is None:
            return
        self.optimizer.load_state_dict(self._resume_state["optimizer"])
        if self.scheduler is not None and self._resume_state.get("scheduler") is not None:
            self.scheduler.load_state_dict(self._resume_state["scheduler"])
        self._resume_state = None
        self.logger.info("resume: optimizer/scheduler khôi phục từ epoch %d", self._completed_epochs)

    def save_resume(self, path: str | Path | None = None) -> Path:
        if self.optimizer is None or self._current_epoch == 0:
            raise ValueError("configure_optimizer and run_epoch must run before save_resume")
        path = Path(path) if path else self.resume_path
        if path is None:
            raise ValueError("resume_path is not set")
        payload = {
            "epoch": self._current_epoch,
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict() if self.scheduler is not None else None,
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
        self.model.load_state_dict(data["model"])
        self._completed_epochs = int(data["epoch"])
        self._best_score = float(data.get("best_score", -float("inf")))
        self._best_epoch = int(data.get("best_epoch", 0))
        self._history = list(data.get("history", []))
        self._resumed_best_state = data.get("best_state")
        self._resume_state = data
        self.logger.info(
            "resume: BEATs fine-tune từ epoch %d (best %.4f @ %d)",
            self._completed_epochs,
            self._best_score,
            self._best_epoch,
        )

    def snapshot(self) -> dict[str, Tensor]:
        if self._resumed_best_state is not None:
            state, self._resumed_best_state = copy.deepcopy(self._resumed_best_state), None
            return state
        return super().snapshot()

    def run_epoch(self, loader: DataLoader, epoch: int) -> dict[str, float]:
        if epoch <= self._completed_epochs:
            return {}
        self._current_epoch = epoch
        self._in_fit = True
        return super().run_epoch(loader, epoch)

    def prepare(self, batch: dict[str, Any]) -> tuple[Tensor, Tensor]:
        waveform = batch["waveform"].to(self.device)
        labels = batch["target"].argmax(dim=1).to(self.device)
        if self.separator is not None:
            with torch.no_grad():
                waveform = self.separator(waveform)
        return waveform, labels

    def begin_epoch(self) -> None:
        self._train_metrics.reset()
        self._correct, self._seen = 0, 0

    def training_step(self, batch: dict[str, Any]) -> StepOutput:
        waveform, labels = self.prepare(batch)
        logits = self.model(waveform).logits
        self._labels, self._logits = labels, logits.detach()
        return StepOutput(loss=self.criterion(logits, labels), size=len(labels))

    def observe(self, step: StepOutput) -> None:
        self._train_metrics.update(self._labels, self._logits.softmax(dim=-1))
        self._correct += (self._logits.argmax(dim=-1) == self._labels).sum().item()
        self._seen += step.size

    def epoch_summary(self) -> dict[str, float]:
        return {"acc": self._correct / max(self._seen, 1), "mAP": self._train_metrics.compute().mAP}

    def evaluate(self, loader: DataLoader, desc: str = "evaluate") -> dict[str, float]:
        self.model.eval()
        scores = self._evaluate_with(self.model.predict_proba, loader, desc)
        if self._in_fit and self._current_epoch > 0:
            self._history.append(
                {"epoch": self._current_epoch, **{f"val_{key}": value for key, value in scores.items()}}
            )
            if self.resume_path is not None:
                self.save_resume(self.resume_path)
                if self.drive_backup is not None:
                    self.drive_backup(self.resume_path)
        return scores

    def confusion_matrix(self, loader: DataLoader, desc: str = "confusion") -> np.ndarray:
        self.model.eval()
        return self._collect(self.model.predict_proba, loader, desc).confusion_matrix()

    def _evaluate_with(
        self, predict_proba: Callable[[Tensor], Tensor], loader: DataLoader, desc: str
    ) -> dict[str, float]:
        return self._collect(predict_proba, loader, desc).compute().to_dict()

    @torch.no_grad()
    def _collect(
        self, predict_proba: Callable[[Tensor], Tensor], loader: DataLoader, desc: str
    ) -> ClassificationMetrics:
        metrics = ClassificationMetrics(num_classes=self.num_classes)
        for batch in tqdm(loader, desc=desc, unit="batch", leave=False, dynamic_ncols=True):
            waveform, labels = self.prepare(batch)
            metrics.update(labels, predict_proba(waveform))
        return metrics

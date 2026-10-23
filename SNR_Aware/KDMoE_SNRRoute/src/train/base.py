from __future__ import annotations

import copy
import logging
from abc import ABC, abstractmethod
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, ClassVar

import torch
from torch import Tensor, nn
from torch.nn.utils import clip_grad_norm_
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from src.core import BaseModel
from src.utils.checkpoint import save_checkpoint

from .config import TrainConfig


@dataclass(frozen=True)
class StepOutput:
    loss: Tensor
    size: int


@dataclass
class TrainingResult:
    selection_metric: str
    best_score: float
    best_epoch: int
    epochs_run: int
    history: list[dict[str, float]] = field(default_factory=list)
    extras: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "selection_metric": self.selection_metric,
            "best_score": self.best_score,
            "best_epoch": self.best_epoch,
            "epochs_run": self.epochs_run,
            "history": self.history,
            **self.extras,
        }


class Trainer(ABC):
    selection_metric: ClassVar[str]

    def __init__(
        self,
        model: BaseModel,
        config: TrainConfig,
        device: torch.device,
        logger: logging.Logger,
        name: str,
        checkpoint_path: str | Path,
        on_checkpoint: Callable[[Path, int, float], None] | None = None,
    ) -> None:
        self.model = model
        self.config = config
        self.device = device
        self.logger = logger
        self.name = name
        self.checkpoint_path = Path(checkpoint_path)
        self.on_checkpoint = on_checkpoint
        self.optimizer: Optimizer | None = None
        self.scheduler: LRScheduler | None = None
        self._best_score = -float("inf")
        self._best_epoch = 0
        self._best_state: dict[str, Tensor] | None = None

    @property
    def checkpoint_module(self) -> nn.Module:
        return self.model

    @property
    def trainable_parameters(self) -> Iterable[nn.Parameter]:
        return self.checkpoint_module.parameters()

    @property
    def lr(self) -> float:
        return self.optimizer.param_groups[0]["lr"]

    @abstractmethod
    def configure_optimizer(self, steps_per_epoch: int) -> None: ...

    @abstractmethod
    def training_step(self, batch: dict[str, Any]) -> StepOutput: ...

    @abstractmethod
    def evaluate(self, loader: DataLoader, desc: str = "evaluate") -> dict[str, float]: ...

    def begin_epoch(self) -> None:
        pass

    def observe(self, step: StepOutput) -> None:
        pass

    def epoch_summary(self) -> dict[str, float]:
        return {}

    def baseline(self, loader: DataLoader) -> float | None:
        return None

    def on_no_improvement(self, epoch: int, stale: int) -> None:
        pass

    def end_epoch(self, epoch: int) -> None:
        pass

    def should_stop(self, epoch: int) -> bool:
        return False

    def snapshot(self) -> dict[str, Tensor]:
        return copy.deepcopy(self.checkpoint_module.state_dict())

    def restore_best(self) -> None:
        if self._best_state is not None:
            self.checkpoint_module.load_state_dict(self._best_state)

    def _record_best(self, score: float, epoch: int) -> None:
        self._best_score, self._best_epoch, self._best_state = score, epoch, self.snapshot()
        save_checkpoint(
            self.checkpoint_path,
            self.checkpoint_module,
            state_dict=self._best_state,
            epoch=epoch,
            **{self.selection_metric: score},
        )
        if self.on_checkpoint is not None:
            self.on_checkpoint(self.checkpoint_path, epoch, score)

    def run_epoch(self, loader: DataLoader, epoch: int) -> dict[str, float]:
        self.model.train()
        self.begin_epoch()
        total_loss, count = 0.0, 0
        progress = tqdm(
            loader,
            desc=f"{self.name} - epoch {epoch}/{self.config.epochs}",
            unit="batch",
            leave=False,
            dynamic_ncols=True,
        )
        for batch in progress:
            step = self.training_step(batch)
            if not torch.isfinite(step.loss):
                continue
            self.optimizer.zero_grad(set_to_none=True)
            step.loss.backward()
            clip_grad_norm_(self.trainable_parameters, self.config.grad_clip)
            self.optimizer.step()
            if self.scheduler is not None:
                self.scheduler.step()
            self.observe(step)
            total_loss += step.loss.item() * step.size
            count += step.size
            progress.set_postfix(loss=f"{total_loss / count:.3f}", lr=f"{self.lr:.1e}")
        return {"train_loss": total_loss / max(count, 1), **self.epoch_summary()}

    def fit(self, train_loader: DataLoader, val_loader: DataLoader) -> TrainingResult:
        self.model.to(self.device)
        self.configure_optimizer(len(train_loader))
        extras: dict[str, Any] = {}

        self._best_state = self.snapshot()
        baseline = self.baseline(val_loader)
        if baseline is not None:
            extras["baseline_score"] = baseline
            self._record_best(baseline, epoch=0)

        history: list[dict[str, float]] = []
        stale, epoch = 0, 0
        for epoch in range(1, self.config.epochs + 1):
            train_stats = self.run_epoch(train_loader, epoch)
            scores = self.evaluate(val_loader, desc=f"{self.name} - validation")
            history.append({"epoch": epoch, **train_stats, **{f"val_{k}": v for k, v in scores.items()}})

            if scores[self.selection_metric] > self._best_score:
                self._record_best(scores[self.selection_metric], epoch)
                stale = 0
            else:
                stale += 1
            self.logger.info(
                f"[{self.name}] epoch {epoch}/{self.config.epochs}"
                f" | train {self._format(train_stats)} | val {self._format(scores)}"
                f" | best {self._best_score:.3f} @ {self._best_epoch} | stale {stale}/{self.config.patience}"
            )
            if stale >= self.config.patience or self.should_stop(epoch):
                self.logger.info("%s early stop at epoch %d", self.name, epoch)
                break
            if stale:
                self.on_no_improvement(epoch, stale)
            self.end_epoch(epoch)

        self.restore_best()
        self.model.eval()
        return TrainingResult(
            selection_metric=self.selection_metric,
            best_score=self._best_score,
            best_epoch=self._best_epoch,
            epochs_run=epoch,
            history=history,
            extras=extras,
        )

    @staticmethod
    def _format(stats: dict[str, float]) -> str:
        return " ".join(f"{key} {value:.3f}" for key, value in stats.items())

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

from src.core import BaseConfig


@dataclass(frozen=True)
class TrainConfig(BaseConfig):
    section: ClassVar[str] = "train"
    epochs: int = 30
    patience: int = 5
    lr: float = 5e-5
    grad_clip: float = 5.0
    batch_size: int = 32
    eval_batch_size: int = 64


@dataclass(frozen=True)
class FRCRNTrainConfig(TrainConfig):
    epochs: int = 20
    lr: float = 1e-4
    grad_clip: float = 10.0
    batch_size: int = 12
    eval_batch_size: int = 32
    halve_lr_after: int = 3
    segment_seconds: float = 2.0
    eval_segment_seconds: float = 4.0
    snr_levels: tuple[float, ...] = (-5, 0, 5, 10, 15, 20)
    batch_candidates: tuple[int, ...] = ()
    vram_fraction: float = 0.9


@dataclass(frozen=True)
class BEATTrainConfig(TrainConfig):
    weight_decay: float = 0.01
    betas: tuple[float, float] = (0.9, 0.98)
    warmup_ratio: float = 0.1
    min_lr_ratio: float = 0.0
    label_smoothing: float = 0.0


@dataclass(frozen=True)
class KDMoETrainConfig(TrainConfig):
    lr: float = 5e-4
    weight_decay: float = 0.05
    betas: tuple[float, float] = (0.9, 0.999)
    warmup_ratio: float = 0.05
    min_lr: float = 1e-6
    grad_clip: float = 1.0
    batch_size: int = 64
    eval_batch_size: int = 64
    epochs: int = 120
    patience: int = 15
    stage1_epochs: int = 8
    ema_decay: float = 0.999
    amp: str = "bf16"
    seed: int = 42
    resume_every: int = 1

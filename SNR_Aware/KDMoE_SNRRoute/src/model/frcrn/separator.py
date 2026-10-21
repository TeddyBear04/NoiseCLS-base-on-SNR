from __future__ import annotations

from pathlib import Path
from typing import Any, ClassVar

import torch
import yaml
from torch import Tensor

from src.core import BaseModel
from src.loss.frcrn import FRCRNLoss

from .config import FRCRNConfig
from .model import FRCRN

_EPS = 1e-6
_DDP_PREFIX = "module."


def audio_norm(waveform: Tensor, target_level_db: float) -> tuple[Tensor, Tensor]:
    target = 10 ** (target_level_db / 20)
    rms = waveform.square().mean(dim=-1, keepdim=True).sqrt()
    first = target / (rms + _EPS)
    scaled = waveform * first
    power = scaled.square()
    loud = power > power.mean(dim=-1, keepdim=True)
    loud_power = (power * loud).sum(dim=-1, keepdim=True) / loud.sum(dim=-1, keepdim=True).clamp_min(1)
    second = target / (loud_power.sqrt() + _EPS)
    return scaled * second, 1.0 / (first * second + _EPS)


class FRCRNSeparator(BaseModel):
    weights_only: ClassVar[bool] = True

    def __init__(self, config: FRCRNConfig | None = None, criterion: FRCRNLoss | None = None) -> None:
        super().__init__()
        self.config = config or FRCRNConfig()
        self.network = FRCRN(self.config)
        self.criterion = criterion or FRCRNLoss(self.config)

    @classmethod
    def from_yaml(cls, path: str | Path) -> FRCRNSeparator:
        with open(path, encoding="utf-8") as f:
            raw = yaml.safe_load(f)
        config = FRCRNConfig.from_dict(raw[FRCRNConfig.section])
        return cls(config, FRCRNLoss(config, **raw.get("loss", {})))

    def forward(self, mixture: Tensor) -> Tensor:
        normalized, scale = audio_norm(mixture, self.config.target_level_db)
        estimate = self.network(normalized).estimate * scale
        return mixture - estimate if self.config.target == "speech" else estimate

    @torch.no_grad()
    def separate(self, mixture: Tensor) -> Tensor:
        return self(mixture)

    def training_target(self, mixture: Tensor, noise: Tensor) -> Tensor:
        return mixture - noise if self.config.target == "speech" else noise

    def loss(self, mixture: Tensor, noise: Tensor) -> Tensor:
        normalized, scale = audio_norm(mixture, self.config.target_level_db)
        output = self.network(normalized)
        target = self.training_target(normalized, noise / scale)
        return self.criterion(normalized, target, output.estimate, output.mask)

    def convert_checkpoint(self, checkpoint: dict[str, Any]) -> dict[str, Tensor]:
        source = checkpoint.get("model", checkpoint)
        source = {key.removeprefix(_DDP_PREFIX): value for key, value in source.items()}
        wanted = self.network.state_dict().keys()
        return {f"network.{key}": source[key] for key in wanted if key in source}

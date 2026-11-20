from __future__ import annotations

import torch
import torch.nn.functional as F

from .base import SupervisedLoss


class CrossEntropyLoss(SupervisedLoss):
    name = "cross_entropy"

    def __init__(self, label_smoothing: float = 0.0):
        super().__init__()
        self.label_smoothing = label_smoothing

    def forward(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return F.cross_entropy(prediction, target, label_smoothing=self.label_smoothing)

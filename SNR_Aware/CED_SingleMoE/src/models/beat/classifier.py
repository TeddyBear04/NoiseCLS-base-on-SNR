from __future__ import annotations

import torch
from torch import nn


class ClassifierHead(nn.Module):
    def __init__(self, dim: int, num_classes: int, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(dim, num_classes)

    def forward(self, patch_features: torch.Tensor) -> torch.Tensor:
        return self.fc(self.dropout(patch_features.mean(dim=1)))

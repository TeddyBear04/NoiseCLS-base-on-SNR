from __future__ import annotations

import torch
from torch import Tensor, nn


class MaskGate(nn.Module):
    """Cổng theo kênh từng token: g = eps + (1 - eps) * sigmoid(W x + b)."""

    def __init__(self, d: int, eps: float = 0.05, bias_init: float = 2.0) -> None:
        super().__init__()
        self.fc = nn.Linear(d, d)
        nn.init.constant_(self.fc.bias, bias_init)
        self.eps = eps

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        g = self.eps + (1.0 - self.eps) * torch.sigmoid(self.fc(x))
        return x * g, g


class NoiseHead(nn.Module):
    """Token (B, F', T', d) -> log-mel nhiễu (B, 1, F, T), chỉ dùng khi có đích L_front."""

    def __init__(self, d: int, patch_size: int = 16, freq_bins: int = 128, num_frames: int = 384) -> None:
        super().__init__()
        self.patch_size = patch_size
        self.grid_freq = freq_bins // patch_size
        self.grid_time = num_frames // patch_size
        self.fc = nn.Linear(d, patch_size * patch_size)

    def forward(self, x: Tensor) -> Tensor:
        batch, freq, time, _ = x.shape
        p = self.patch_size
        y = self.fc(x).reshape(batch, freq, time, p, p).permute(0, 1, 3, 2, 4)
        return y.reshape(batch, 1, freq * p, time * p)

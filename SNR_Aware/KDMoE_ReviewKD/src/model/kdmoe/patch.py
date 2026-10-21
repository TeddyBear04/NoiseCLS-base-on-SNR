from __future__ import annotations

import torch
from torch import Tensor, nn


class PatchEmbed(nn.Module):
    """Log-mel (B, 1, F, T) -> token (B, F', T', d), vị trí tần số và thời gian tách riêng."""

    def __init__(self, d: int, patch_size: int = 16, freq_bins: int = 128, num_frames: int = 384) -> None:
        super().__init__()
        self.grid_freq = freq_bins // patch_size
        self.grid_time = num_frames // patch_size
        self.proj = nn.Conv2d(1, d, kernel_size=patch_size, stride=patch_size)
        self.freq_pos = nn.Parameter(torch.zeros(self.grid_freq, d))
        self.time_pos = nn.Parameter(torch.zeros(self.grid_time, d))
        nn.init.normal_(self.freq_pos, std=0.02)
        nn.init.normal_(self.time_pos, std=0.02)

    def forward(self, x: Tensor) -> Tensor:
        x = self.proj(x).permute(0, 2, 3, 1)
        return x + self.freq_pos[None, :, None, :] + self.time_pos[None, None, :, :]

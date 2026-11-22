"""Masked Generative Distillation (Yang et al., ECCV 2022) on the token grid of a ViT student.

The student's patch tokens (freq x time grid) are projected to the teacher width, a random spatial mask hides part of
the grid, and a small convolutional generator must reconstruct the teacher's tokens on the same grid. Training only;
the module is not part of the student and is never saved with it.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


def student_tokens_to_map(tokens: Tensor, freq_patches: int, time_patches: int) -> Tensor:
    """[B, 1 + F*T, C] (class token first, frequency-major patches) -> [B, C, F, T]."""
    patches = tokens[:, 1:, :]
    batch, count, channels = patches.shape
    if count != freq_patches * time_patches:
        raise ValueError(f"expected {freq_patches * time_patches} patch tokens, got {count}")
    return patches.reshape(batch, freq_patches, time_patches, channels).permute(0, 3, 1, 2)


def teacher_tokens_to_map(tokens: Tensor, freq_patches: int, time_patches: int) -> Tensor:
    """BEATs tokens [B, T*Fb, C] (time-major, Fb frequency patches per frame) -> [B, C, F, T] with F pooled to freq_patches."""
    batch, count, channels = tokens.shape
    teacher_freq = count // time_patches
    if teacher_freq * time_patches != count or teacher_freq % freq_patches != 0:
        raise ValueError(f"cannot map {count} teacher tokens to a {freq_patches}x{time_patches} grid")
    grid = tokens.reshape(batch, time_patches, teacher_freq, channels).permute(0, 3, 2, 1)  # [B, C, Fb, T]
    return F.adaptive_avg_pool2d(grid, (freq_patches, time_patches))


class MGDHead(nn.Module):
    """L = alpha * sum((G(mask * align(F_S)) - F_T)^2) / B  with a spatial (per-location) mask, as in the original paper."""

    def __init__(self, student_dim: int, teacher_dim: int = 768, alpha: float = 7e-5, mask_ratio: float = 0.5) -> None:
        super().__init__()
        self.alpha = alpha
        self.mask_ratio = mask_ratio
        self.align = nn.Conv2d(student_dim, teacher_dim, kernel_size=1)
        self.generator = nn.Sequential(
            nn.Conv2d(teacher_dim, teacher_dim, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(teacher_dim, teacher_dim, kernel_size=3, padding=1),
        )

    def forward(self, student_map: Tensor, teacher_map: Tensor) -> Tensor:
        aligned = self.align(student_map.float())
        if self.training:
            keep = (torch.rand(aligned.shape[0], 1, aligned.shape[2], aligned.shape[3], device=aligned.device) >= self.mask_ratio).float()
            aligned = aligned * keep
        rebuilt = self.generator(aligned)
        return self.alpha * (rebuilt - teacher_map.detach().float()).pow(2).sum() / aligned.shape[0]

from __future__ import annotations

import torch
from torch import nn


class PatchEmbed(nn.Module):
    def __init__(self, patch_size: int = 16, embed_dim: int = 512):
        super().__init__()
        self.patch_embedding = nn.Conv2d(1, embed_dim, patch_size, stride=patch_size, bias=False)

    def forward(self, fbank: torch.Tensor) -> torch.Tensor:
        x = self.patch_embedding(fbank.unsqueeze(1))
        return x.reshape(x.size(0), x.size(1), -1).transpose(1, 2)

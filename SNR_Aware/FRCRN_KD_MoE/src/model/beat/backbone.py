from __future__ import annotations

import torch
from torch import nn

from .encoder import TransformerEncoder
from .patch_embed import PatchEmbed


class BEATsBackbone(nn.Module):
    """fbank (B, T, 128) -> patch features (B, N, 768), N = (T/16) * 8."""

    def __init__(self, config):
        super().__init__()
        self.patch_embed = PatchEmbed(config.input_patch_size, config.embed_dim)
        self.layer_norm = nn.LayerNorm(config.embed_dim)
        self.post_extract_proj = nn.Linear(config.embed_dim, config.encoder_embed_dim)
        self.dropout_input = nn.Dropout(config.dropout_input)
        self.encoder = TransformerEncoder(config)

    def forward(self, fbank: torch.Tensor) -> torch.Tensor:
        x = self.patch_embed(fbank)
        x = self.post_extract_proj(self.layer_norm(x))
        return self.encoder(self.dropout_input(x))

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import nn

from .encoder import TransformerEncoder
from .patch_embed import PatchEmbed


class BEATsBackbone(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.patch_embed = PatchEmbed(config.input_patch_size, config.embed_dim)
        self.layer_norm = nn.LayerNorm(config.embed_dim)
        self.post_extract_proj = nn.Linear(config.embed_dim, config.encoder_embed_dim)
        self.dropout_input = nn.Dropout(config.dropout_input)
        self.encoder = TransformerEncoder(config)

    def forward(self, fbank: torch.Tensor) -> torch.Tensor:
        return self.forward_with_hidden(fbank)[0]

    def forward_with_hidden(
        self, fbank: torch.Tensor, layers: Sequence[int] = ()
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, ...]]:
        x = self.patch_embed(fbank)
        x = self.post_extract_proj(self.layer_norm(x))
        return self.encoder.forward_with_hidden(self.dropout_input(x), layers)

from __future__ import annotations

import math
from collections.abc import Sequence

import torch
from torch import nn
from torch.nn.utils import weight_norm

from .attention import GatedRelPosAttention
from .ffn import FeedForward


class SamePad(nn.Module):
    def __init__(self, kernel_size: int):
        super().__init__()
        self.remove = 1 if kernel_size % 2 == 0 else 0

    def forward(self, x):
        return x[:, :, : -self.remove] if self.remove else x


class EncoderLayer(nn.Module):
    def __init__(self, config, layer_idx: int):
        super().__init__()
        dim = config.encoder_embed_dim
        self.self_attn = GatedRelPosAttention(
            dim,
            config.encoder_attention_heads,
            config.attention_dropout,
            has_relative_bias=config.relative_position_embedding and layer_idx == 0,
            num_buckets=config.num_buckets,
            max_distance=config.max_distance,
            gru_rel_pos=config.gru_rel_pos,
        )
        self.ffn = FeedForward(dim, config.encoder_ffn_embed_dim, config.activation_dropout)
        self.self_attn_layer_norm = nn.LayerNorm(dim)
        self.final_layer_norm = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(config.dropout)
        self.alpha = math.pow(2 * config.encoder_layers, 0.25) if config.deep_norm else 1.0

    def forward(self, x, position_bias=None):
        attn, position_bias = self.self_attn(x, position_bias)
        x = self.self_attn_layer_norm(x * self.alpha + self.dropout(attn))
        x = self.final_layer_norm(x * self.alpha + self.dropout(self.ffn(x)))
        return x, position_bias


class TransformerEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        dim = config.encoder_embed_dim
        conv = nn.Conv1d(dim, dim, config.conv_pos, padding=config.conv_pos // 2, groups=config.conv_pos_groups)
        self.pos_conv = nn.Sequential(weight_norm(conv, name="weight", dim=2), SamePad(config.conv_pos), nn.GELU())
        self.layers = nn.ModuleList(EncoderLayer(config, i) for i in range(config.encoder_layers))
        self.layer_norm = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward_with_hidden(x)[0]

    def forward_with_hidden(
        self, x: torch.Tensor, layers: Sequence[int] = ()
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, ...]]:
        invalid = [index for index in layers if not 1 <= index <= len(self.layers)]
        if invalid:
            raise ValueError(f"layers {invalid} are outside 1..{len(self.layers)}")
        x = x + self.pos_conv(x.transpose(1, 2)).transpose(1, 2)
        x = self.dropout(self.layer_norm(x))
        position_bias = None
        hidden: dict[int, torch.Tensor] = {}
        for index, layer in enumerate(self.layers, start=1):
            x, position_bias = layer(x, position_bias)
            if index in layers:
                hidden[index] = x
        return x, tuple(hidden[index] for index in layers)

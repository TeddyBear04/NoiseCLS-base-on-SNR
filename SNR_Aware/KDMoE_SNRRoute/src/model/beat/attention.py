from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn


def relative_position_bucket(relative_position, num_buckets=320, max_distance=800):
    num_buckets //= 2
    buckets = (relative_position > 0).long() * num_buckets
    relative_position = relative_position.abs()
    max_exact = num_buckets // 2
    is_small = relative_position < max_exact
    large = (
        max_exact
        + (
            torch.log(relative_position.float().clamp(min=1) / max_exact)
            / math.log(max_distance / max_exact)
            * (num_buckets - max_exact)
        ).long()
    )
    large = torch.clamp(large, max=num_buckets - 1)
    return buckets + torch.where(is_small, relative_position, large)


class GatedRelPosAttention(nn.Module):
    """Multi-head self-attention + gated relative position bias của BEATs.

    Chỉ layer 0 có `relative_attention_bias`; các layer sau dùng lại `position_bias`.
    """

    def __init__(
        self, dim, num_heads, dropout=0.0, has_relative_bias=False, num_buckets=320, max_distance=800, gru_rel_pos=True
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scaling = self.head_dim**-0.5
        self.num_buckets, self.max_distance = num_buckets, max_distance
        self.dropout = nn.Dropout(dropout)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.q_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)
        if has_relative_bias:
            self.relative_attention_bias = nn.Embedding(num_buckets, num_heads)
        self.gru_rel_pos = gru_rel_pos
        if gru_rel_pos:
            self.grep_linear = nn.Linear(self.head_dim, 8)
            self.grep_a = nn.Parameter(torch.ones(1, num_heads, 1, 1))

    def compute_bias(self, length: int, device) -> torch.Tensor:
        pos = torch.arange(length, device=device)
        bucket = relative_position_bucket(pos[None, :] - pos[:, None], self.num_buckets, self.max_distance)
        return self.relative_attention_bias(bucket).permute(2, 0, 1).unsqueeze(0)  # (1, H, T, T)

    def forward(self, x, position_bias=None):
        B, T, _ = x.shape
        H, d = self.num_heads, self.head_dim
        q, k, v = (p(x).view(B, T, H, d).transpose(1, 2) for p in (self.q_proj, self.k_proj, self.v_proj))

        if position_bias is None:
            position_bias = self.compute_bias(T, x.device)
        bias = position_bias
        if self.gru_rel_pos:
            gate = torch.sigmoid(self.grep_linear(q).view(B, H, T, 2, 4).sum(-1))
            gate_a, gate_b = gate.chunk(2, dim=-1)  # (B, H, T, 1)
            bias = (gate_a * (gate_b * self.grep_a - 1.0) + 2.0) * position_bias

        attn = (q * self.scaling) @ k.transpose(-2, -1) + bias
        attn = self.dropout(F.softmax(attn, dim=-1))
        out = (attn @ v).transpose(1, 2).reshape(B, T, H * d)
        return self.out_proj(out), position_bias

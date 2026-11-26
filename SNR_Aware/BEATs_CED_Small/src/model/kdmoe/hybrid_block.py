from __future__ import annotations

import torch
from torch import Tensor, nn

from .config import HYBRID_BRANCHES


class DropPath(nn.Module):
    """Stochastic depth: bỏ toàn bộ residual theo mẫu, khi training."""

    def __init__(self, p: float = 0.0) -> None:
        super().__init__()
        if not 0.0 <= p < 1.0:
            raise ValueError(f"drop probability must be in [0, 1), got {p}")
        self.p = p

    def forward(self, x: Tensor) -> Tensor:
        if self.p <= 0.0 or not self.training:
            return x
        keep = 1.0 - self.p
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        return x * x.new_empty(shape).bernoulli_(keep) / keep


class FeedForward(nn.Module):
    def __init__(self, d: int, ratio: int = 4, dropout: float = 0.1) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d, ratio * d),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ratio * d, d),
            nn.Dropout(dropout),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


class HybridBlock(nn.Module):
    """Residual song song MHSA trên N token và BiGRU theo thời gian từng dải, rồi FFN."""

    def __init__(
        self,
        d: int,
        heads: int,
        gru_hidden: int,
        ffn_ratio: int = 4,
        dropout: float = 0.1,
        drop_path: float = 0.0,
        branches: str = "both",
    ) -> None:
        super().__init__()
        if branches not in HYBRID_BRANCHES:
            raise ValueError(f"branches must be one of {HYBRID_BRANCHES}, got {branches!r}")
        self.branches = branches
        self.ln1 = nn.LayerNorm(d)
        self.ln2 = nn.LayerNorm(d)
        if branches in ("both", "attn"):
            self.attn = nn.MultiheadAttention(d, heads, dropout=dropout, batch_first=True)
            self.a_att = nn.Parameter(torch.ones(1))
        if branches in ("both", "gru"):
            self.gru = nn.GRU(d, gru_hidden, batch_first=True, bidirectional=True)
            self.proj = nn.Identity() if 2 * gru_hidden == d else nn.Linear(2 * gru_hidden, d)
            self.a_gru = nn.Parameter(torch.ones(1))
        self.ffn = FeedForward(d, ffn_ratio, dropout)
        self.drop_path = DropPath(drop_path)

    def forward(self, x: Tensor) -> Tensor:
        batch, freq, time, dim = x.shape
        u = self.ln1(x)
        delta = None
        if self.branches in ("both", "attn"):
            tokens = u.reshape(batch, freq * time, dim)
            attended, _ = self.attn(tokens, tokens, tokens, need_weights=False)
            delta = self.a_att * attended.reshape(batch, freq, time, dim)
        if self.branches in ("both", "gru"):
            sequence, _ = self.gru(u.reshape(batch * freq, time, dim))
            branch = self.a_gru * self.proj(sequence).reshape(batch, freq, time, dim)
            delta = branch if delta is None else delta + branch
        x = x + self.drop_path(delta)
        return x + self.drop_path(self.ffn(self.ln2(x)))

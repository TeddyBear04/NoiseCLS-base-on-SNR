from __future__ import annotations

import torch
from torch import Tensor, nn

from .config import FREQ_POOL_MODES


class FreqPoolGRU(nn.Module):
    """Gộp tần số -> BiGRU theo thời gian -> attention pooling -> embedding e (B, 4 * hidden)."""

    def __init__(self, d: int, hidden: int, attn_dim: int = 64, freq_pool: str = "mean") -> None:
        super().__init__()
        if freq_pool not in FREQ_POOL_MODES:
            raise ValueError(f"freq_pool must be one of {FREQ_POOL_MODES}, got {freq_pool!r}")
        self.freq_pool = freq_pool
        self.hidden = hidden
        self.gru = nn.GRU(d, hidden, batch_first=True, bidirectional=True)
        self.score = nn.Linear(2 * hidden, attn_dim)
        self.weight = nn.Linear(attn_dim, 1, bias=False)
        if freq_pool == "attn":
            self.freq_score = nn.Linear(d, attn_dim)
            self.freq_weight = nn.Linear(attn_dim, 1, bias=False)

    def _pool_freq(self, x: Tensor) -> Tensor:
        if self.freq_pool == "mean":
            return x.mean(dim=1)
        scores = self.freq_weight(torch.tanh(self.freq_score(x))).squeeze(-1)
        weights = scores.softmax(dim=1)
        return (x * weights.unsqueeze(-1)).sum(dim=1)

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        pooled = self._pool_freq(x)
        sequence, _ = self.gru(pooled)
        scores = self.weight(torch.tanh(self.score(sequence))).squeeze(-1)
        weights = scores.softmax(dim=-1)
        attended = (sequence * weights.unsqueeze(-1)).sum(dim=1)
        h = self.hidden
        embedding = torch.cat([sequence[:, -1, :h], sequence[:, 0, h:], attended], dim=-1)
        return embedding, sequence

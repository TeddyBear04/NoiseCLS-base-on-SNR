from __future__ import annotations

from torch import Tensor, nn

from .hybrid_block import DropPath, FeedForward


class ViTBlock(nn.Module):
    """Khối pre-LN: MHSA trên N = F' * T' token, rồi FFN, mỗi nhánh một residual."""

    def __init__(self, d: int, heads: int, ffn_ratio: int = 4, dropout: float = 0.1, drop_path: float = 0.0) -> None:
        super().__init__()
        self.ln1 = nn.LayerNorm(d)
        self.ln2 = nn.LayerNorm(d)
        self.attn = nn.MultiheadAttention(d, heads, dropout=dropout, batch_first=True)
        self.ffn = FeedForward(d, ffn_ratio, dropout)
        self.drop_path = DropPath(drop_path)

    def forward(self, x: Tensor) -> Tensor:
        batch, freq, time, dim = x.shape
        u = self.ln1(x).reshape(batch, freq * time, dim)
        attended, _ = self.attn(u, u, u, need_weights=False)
        x = x + self.drop_path(attended.reshape(batch, freq, time, dim))
        return x + self.drop_path(self.ffn(self.ln2(x)))

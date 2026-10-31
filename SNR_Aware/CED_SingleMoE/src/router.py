"""
Self-Contained Spectral-Token Router for Single-Backbone MoE (Zero-Oracle SNR).

This module estimates the continuous expert gating weight alpha in (0, 1) dynamically
from the combination of:
1. 6 intrinsic acoustic/spectral physical features (f_spec in R^6) extracted from waveform.
2. High-level intermediate CLS token representation (h_9 in R^192) extracted after Block 9.

References:
    - Soft MoE: Puigcerver et al., "From Sparse to Soft Mixtures of Experts",
      ICLR 2024. arXiv:2308.00951.
    - DeepSeekMoE: Dai et al., "DeepSeekMoE: Towards Ultimate Expert Specialization
      in Mixture-of-Experts Language Models", ACL 2024. arXiv:2401.06066.
    - V-MoE: Riquelme et al., "Scaling Vision with Sparse Mixture of Experts",
      NeurIPS 2021. arXiv:2106.05974.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


def snr_band_targets(snr_db: Tensor, edges_db: tuple[float, ...], tau: float) -> Tensor:
    """Nhãn router mềm [B, K] trên K = len(edges_db) + 1 khoảng SNR.

    c_k = sigmoid((SNR - e_k) / tau) là xác suất "SNR trên ngưỡng e_k"; khoảng 0 nhận 1 - c_0, khoảng k nhận
    c_{k-1} - c_k, khoảng cuối nhận c_{K-2}. Ngưỡng tăng dần nên mọi phần >= 0 và tổng bằng 1.
    Một ngưỡng [15] cho [1 - w, w] với w = sigmoid((SNR - 15) / tau): đúng nhãn alpha của bản gốc 2 expert.
    Ngưỡng [2.5, 12.5]: -5/0 dB -> low, 5/10 dB -> mid, 15/20 dB -> high (cùng 3 khoảng của KD-MoE)."""
    above = torch.sigmoid((snr_db.float().unsqueeze(-1) - snr_db.new_tensor(edges_db).float()) / tau)  # [B, K-1]
    ones, zeros = torch.ones_like(above[:, :1]), torch.zeros_like(above[:, :1])
    return torch.cat([ones, above], dim=-1) - torch.cat([above, zeros], dim=-1)


def snr_band_index(snr_db: Tensor, edges_db: tuple[float, ...]) -> Tensor:
    """Chỉ số khoảng SNR cứng [B]: số ngưỡng <= SNR (ngưỡng [2.5, 12.5]: -5/0 -> 0, 5/10 -> 1, 15/20 -> 2)."""
    return (snr_db.float().unsqueeze(-1) >= snr_db.new_tensor(edges_db).float()).sum(dim=-1)


class SelfContainedSpectralTokenRouter(nn.Module):
    """
    Router nội tại: CLS token sau Block 9 (nén 192 -> 16) ghép 6 đặc trưng âm học -> MLP 22 -> 32 -> K
    -> softmax -> gates [B, K] trên các expert SNR (low, [mid,] high). Không cần nhãn SNR lúc suy luận;
    lúc train gates được giám sát bằng nhãn mềm `snr_band_targets`.

    Outputs:
        - alpha: [B, 1] gate của expert cao nhất (high), dùng cho log và trộn teacher lúc suy luận
        - gates: [B, K] xác suất trên các expert, tổng bằng 1
    """
    def __init__(
        self,
        spectral_dim: int = 6,
        token_dim: int = 192,
        token_proj_dim: int = 16,
        hidden_dim: int = 32,
        num_experts: int = 2,
    ) -> None:
        super().__init__()
        self.num_experts = num_experts

        # Branch 1: Token projection layer (192 -> 16)
        self.token_proj = nn.Sequential(
            nn.LayerNorm(token_dim),
            nn.Linear(token_dim, token_proj_dim),
            nn.GELU(),
        )

        # Branch 2: Cross-level interaction MLP (22 -> 32 -> K)
        self.mlp = nn.Sequential(
            nn.Linear(spectral_dim + token_proj_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, num_experts),
        )

    def forward(self, spectral_feat: Tensor, intermediate_token: Tensor) -> tuple[Tensor, Tensor]:
        token_h = self.token_proj(intermediate_token)  # [B, 16]
        gates = F.softmax(self.mlp(torch.cat([spectral_feat, token_h], dim=-1)), dim=-1)  # [B, K]
        return gates[:, -1:], gates


def router_loss(gates: Tensor, targets: Tensor) -> Tensor:
    """CE mềm giữa gates và nhãn khoảng SNR; với K = 2 đúng bằng BCE(alpha, w) của bản gốc."""
    return -(targets.float() * gates.float().clamp_min(1e-6).log()).sum(dim=-1).mean()

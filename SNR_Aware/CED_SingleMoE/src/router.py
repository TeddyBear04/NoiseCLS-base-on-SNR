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


class SelfContainedSpectralTokenRouter(nn.Module):
    """
    Self-Contained Dynamic Routing Module estimating continuous soft gating weight alpha.

    Dataflow Architecture:
        1. CLS token h_9 [B, 192] after Block 9
           -> LayerNorm(192) -> Linear(192, 16) -> GELU -> h_tilde_9 [B, 16]
        2. Concat with f_spec [B, 6] from frontend:
           -> u = [f_spec || h_tilde_9] [B, 22]
        3. Cross-feature MLP interaction:
           -> Linear(22, 32) -> LayerNorm(32) -> GELU -> Linear(32, 2)
           -> Logits [r_low, r_high] [B, 2]
        4. Softmax normalization:
           -> alpha = exp(r_high) / (exp(r_low) + exp(r_high)) in (0, 1)

    Outputs:
        - alpha: [B, 1] soft routing weight for High-Expert (1 - alpha for Low-Expert)
        - gates: [B, 2] normalized probabilities over [Low, High]
    """
    def __init__(
        self,
        spectral_dim: int = 6,
        token_dim: int = 192,
        token_proj_dim: int = 16,
        hidden_dim: int = 32,
        transition_center_db: float = 15.0,
        transition_tau: float = 2.5,
    ) -> None:
        super().__init__()
        self.transition_center_db = transition_center_db
        self.transition_tau = transition_tau

        # Branch 1: Token projection layer (192 -> 16)
        self.token_proj = nn.Sequential(
            nn.LayerNorm(token_dim),
            nn.Linear(token_dim, token_proj_dim),
            nn.GELU(),
        )

        # Branch 2: Cross-level interaction MLP (22 -> 32 -> 2)
        in_total = spectral_dim + token_proj_dim
        self.mlp = nn.Sequential(
            nn.Linear(in_total, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 2),  # [r_low, r_high]
        )

    def forward(
        self,
        spectral_feat: Tensor,
        intermediate_token: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """
        Args:
            spectral_feat: [B, 6] from AcousticFeatureExtractor
            intermediate_token: [B, 192] CLS token after Block 9
        Returns:
            alpha: [B, 1] soft routing weight for High-SNR Expert
            gates: [B, 2] normalized softmax probabilities [p_low, p_high]
        """
        # Compress 192-dim token to 16-dim representation
        token_h = self.token_proj(intermediate_token)  # [B, 16]

        # Concatenate physical and semantic representations
        fused_feat = torch.cat([spectral_feat, token_h], dim=-1)  # [B, 22]

        # Compute routing logits
        logits = self.mlp(fused_feat)  # [B, 2]

        # Softmax probability distribution
        gates = F.softmax(logits, dim=-1)  # [B, 2]
        alpha = gates[:, 1:2]              # [B, 1] High-Expert weight
        return alpha, gates

    def compute_target_soft_label(self, snr_db: Tensor) -> Tensor:
        """
        Compute continuous ground-truth routing label from SNR during training:
        y_target = Sigmoid((SNR - 15.0) / 2.5)

        Reference behavior:
            SNR = -5 dB -> y_target = 0.0003  (Fully Low-SNR mode)
            SNR =  0 dB -> y_target = 0.0025
            SNR =  5 dB -> y_target = 0.0180
            SNR = 10 dB -> y_target = 0.1192
            SNR = 15 dB -> y_target = 0.5000  (Smooth transition zone)
            SNR = 20 dB -> y_target = 0.8808  (High-SNR speech leakage mode)
        """
        z = (snr_db.float() - self.transition_center_db) / self.transition_tau
        return torch.sigmoid(z).unsqueeze(-1)

    def compute_router_loss(self, alpha: Tensor, snr_db: Tensor) -> Tensor:
        """
        Supervised Binary Cross Entropy loss for routing weight during training.
        """
        target = self.compute_target_soft_label(snr_db)
        return F.binary_cross_entropy(alpha.float().clamp(1e-6, 1.0 - 1e-6), target.float())

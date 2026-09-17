"""Three-view BEATs encoding and quality-gated fusion classification."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from models.separator import NoiseSeparator
from utils.training36 import mixed_precision_context


class FusionClassifier(nn.Module):
    """Keep mixture evidence and add quality-gated auxiliary residuals.

    The mixture embedding is an unconditional skip connection.  The learned
    softmax gate estimates reliability for mixture, separated noise, and the
    speech-conditioned view.  Only auxiliary residuals are scaled by that gate,
    so the model starts as an exact mixture-only classifier.
    """

    def __init__(
        self,
        num_classes: int,
        embed_dim: int = 768,
        dropout: float = 0.1,
        noise_dropout: float = 0.2,
        conditioned_dropout: float = 0.2,
        gate_hidden_dim: int = 256,
        contrastive_dim: int = 128,
    ) -> None:
        super().__init__()
        self.mix_gate_norm = nn.LayerNorm(embed_dim)
        self.noise_norm = nn.LayerNorm(embed_dim)
        self.conditioned_norm = nn.LayerNorm(embed_dim)
        self.quality_gate = nn.Sequential(
            nn.Linear(3 * embed_dim + 1, gate_hidden_dim),
            nn.GELU(),
            nn.Linear(gate_hidden_dim, 3),
        )
        self.noise_dropout = nn.Dropout(noise_dropout)
        self.conditioned_dropout = nn.Dropout(conditioned_dropout)
        self.fusion = nn.Linear(3 * embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(embed_dim, num_classes)
        self.noise_aux_head = nn.Linear(embed_dim, num_classes)
        self.noise_projection = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(embed_dim, contrastive_dim),
        )
        nn.init.zeros_(self.fusion.weight)
        nn.init.zeros_(self.fusion.bias)
        with torch.no_grad():
            # Start with mixture as the trusted view.  Zero fusion still makes
            # the initial classifier exactly mixture-only.
            self.quality_gate[-1].weight.zero_()
            self.quality_gate[-1].bias.copy_(torch.tensor([2.0, -2.0, -2.0]))

    def forward(
        self,
        z_mix: torch.Tensor,
        z_noise: torch.Tensor,
        z_conditioned: torch.Tensor,
        noise_ratio_db: torch.Tensor,
        speech_ratio_db: torch.Tensor,
        return_details: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, dict[str, torch.Tensor]]:
        # Keep this argument for cache/checkpoint API compatibility.  The gate
        # deliberately uses [z_mix, z_noise, z_conditioned, r] as designed.
        del speech_ratio_db
        gate_input = torch.cat(
            [
                self.mix_gate_norm(z_mix),
                self.noise_norm(z_noise),
                self.conditioned_norm(z_conditioned),
                noise_ratio_db.float().unsqueeze(-1) / 10,
            ],
            dim=-1,
        )
        weights = torch.softmax(self.quality_gate(gate_input), dim=-1)
        mixture_weight = weights[:, :1].clamp_min(1e-4)
        noise = (weights[:, 1:2] / mixture_weight) * self.noise_dropout(self.noise_norm(z_noise))
        conditioned = (weights[:, 2:3] / mixture_weight) * self.conditioned_dropout(
            self.conditioned_norm(z_conditioned)
        )
        fused = z_mix + self.fusion(torch.cat([z_mix, noise, conditioned], dim=-1))
        logits = self.head(self.dropout(fused))
        if return_details:
            return logits, {
                "branch_weights": weights,
                "noise_logits": self.noise_aux_head(self.noise_norm(z_noise)),
                "mix_logits": self.head(z_mix),
                "conditioned_logits": self.head(z_conditioned),
                "noise_projection": self.noise_projection(self.noise_norm(z_noise)),
            }
        return logits


def supervised_contrastive_loss(
    projections: torch.Tensor, targets: torch.Tensor, temperature: float = 0.1
) -> torch.Tensor:
    """Supervised contrastive loss over noise-view projections in one batch."""
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    features = F.normalize(projections.float(), dim=-1)
    similarities = features @ features.T / temperature
    diagonal = torch.eye(len(targets), device=targets.device, dtype=torch.bool)
    similarities = similarities.masked_fill(diagonal, float("-inf"))
    positives = targets[:, None].eq(targets[None, :]) & ~diagonal
    valid = positives.any(dim=1)
    if not valid.any():
        return projections.new_zeros(())
    log_prob = similarities - torch.logsumexp(similarities, dim=1, keepdim=True)
    positive_log_prob = log_prob.masked_fill(~positives, 0).sum(dim=1)
    return -(positive_log_prob[valid] / positives.sum(dim=1)[valid]).mean()


def noise_energy_ratio_db(noise: torch.Tensor, mixture: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Per-clip 10*log10(E[n_hat^2] / E[mixture^2])."""
    noise_energy = noise.float().pow(2).mean(dim=-1)
    mixture_energy = mixture.float().pow(2).mean(dim=-1)
    return 10 * torch.log10((noise_energy + eps) / (mixture_energy + eps))


def encode_branches(
    encoder,
    separator: NoiseSeparator,
    mixture: torch.Tensor,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Encode mixture, amplified noise, and WDRC-conditioned residual views."""
    noise = separator(mixture)
    amplified = separator.amplify(noise)
    speech = mixture - noise
    conditioned = separator.compress_and_amplify_speech(speech) + noise
    with mixed_precision_context(device):
        sequence, _ = encoder.extract_features(torch.cat([mixture, amplified, conditioned], dim=0))
        z_mix, z_noise, z_conditioned = sequence.mean(dim=1).split(mixture.shape[0])
    return (
        z_mix,
        z_noise,
        z_conditioned,
        noise,
        noise_energy_ratio_db(noise, mixture).detach(),
        noise_energy_ratio_db(speech, mixture).detach(),
    )

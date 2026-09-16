"""Two-branch BEATs encoding and the mixture/noise fusion classifier."""

from __future__ import annotations

import torch
from torch import nn

from models.separator import NoiseSeparator
from utils.training36 import mixed_precision_context


class FusionClassifier(nn.Module):
    """fused = z_mix + Linear(1536, 768)(Concat(z_mix, z_noise)) → Linear(768, classes).

    The fusion projection starts at zero, so the untrained model is exactly the
    mixture-only BEATs head (as in ``audio_noise_capstone``). Weight decay pulls
    the projection back toward that baseline, so the noise branch only adds what
    the validation data supports instead of replacing the mixture evidence.
    """

    def __init__(
        self,
        num_classes: int,
        embed_dim: int = 768,
        dropout: float = 0.1,
        noise_dropout: float = 0.2,
        conditioned_dropout: float = 0.2,
    ) -> None:
        super().__init__()
        # Separated-noise embeddings are out of distribution for BEATs, so
        # normalise them and drop features to stop the head memorising them.
        self.noise_norm = nn.LayerNorm(embed_dim)
        self.conditioned_norm = nn.LayerNorm(embed_dim)
        # Confidence gate from the separator's own energy estimate: near 0 dB the
        # clip is noise-dominated and n_hat is reliable; at very negative values
        # (high SNR) n_hat is mostly leaked speech, so the gate closes.
        self.gate = nn.Linear(1, 1)
        self.conditioned_gate = nn.Linear(1, 1)
        self.noise_dropout = nn.Dropout(noise_dropout)
        self.conditioned_dropout = nn.Dropout(conditioned_dropout)
        self.fusion = nn.Linear(3 * embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(embed_dim, num_classes)
        nn.init.zeros_(self.fusion.weight)
        nn.init.zeros_(self.fusion.bias)
        with torch.no_grad():
            self.gate.weight.fill_(1.0)
            self.gate.bias.fill_(1.0)
            self.conditioned_gate.weight.fill_(1.0)
            self.conditioned_gate.bias.fill_(0.0)

    def forward(
        self,
        z_mix: torch.Tensor,
        z_noise: torch.Tensor,
        z_conditioned: torch.Tensor,
        noise_ratio_db: torch.Tensor,
        speech_ratio_db: torch.Tensor,
    ) -> torch.Tensor:
        gate = torch.sigmoid(self.gate(noise_ratio_db.float().unsqueeze(-1) / 10))
        noise = gate * self.noise_dropout(self.noise_norm(z_noise))
        conditioned_gate = torch.sigmoid(
            self.conditioned_gate(speech_ratio_db.float().unsqueeze(-1) / 10)
        )
        conditioned = conditioned_gate * self.conditioned_dropout(
            self.conditioned_norm(z_conditioned)
        )
        fused = z_mix + self.fusion(torch.cat([z_mix, noise, conditioned], dim=-1))
        return self.head(self.dropout(fused))


def noise_energy_ratio_db(noise: torch.Tensor, mixture: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Per-clip ``10·log10(E[n_hat²] / E[mixture²])``, a label-free SNR estimate."""
    noise_energy = noise.float().pow(2).mean(dim=-1)
    mixture_energy = mixture.float().pow(2).mean(dim=-1)
    return 10 * torch.log10((noise_energy + eps) / (mixture_energy + eps))


def encode_branches(
    encoder,
    separator: NoiseSeparator,
    mixture: torch.Tensor,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return ``z_mix``, ``z_noise``, the raw noise estimate ``n_hat`` and its
    detached noise-to-mixture energy ratio in dB (input of the fusion gate).

    Both branches go through the same BEATs encoder in one doubled batch, so the
    transformer weights are shared by construction.
    """
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

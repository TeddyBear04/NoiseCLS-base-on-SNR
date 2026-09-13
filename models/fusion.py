"""Two-branch BEATs encoding and the mixture/noise fusion classifier."""

from __future__ import annotations

import torch
from torch import nn

from models.separator import NoiseSeparator
from utils.training36 import mixed_precision_context


class FusionClassifier(nn.Module):
    """Concat(z_mix, z_noise) → Linear(1536, 768) → Linear(768, classes)."""

    def __init__(self, num_classes: int, embed_dim: int = 768, dropout: float = 0.1) -> None:
        super().__init__()
        self.fusion = nn.Linear(2 * embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(embed_dim, num_classes)
        with torch.no_grad():
            # Start as the average of both embeddings so the AudioSet-initialised
            # head sees features on the scale it was trained for.
            identity = torch.eye(embed_dim)
            self.fusion.weight.copy_(0.5 * torch.cat([identity, identity], dim=1))
            self.fusion.bias.zero_()

    def forward(self, z_mix: torch.Tensor, z_noise: torch.Tensor) -> torch.Tensor:
        fused = self.fusion(torch.cat([z_mix, z_noise], dim=-1))
        return self.head(self.dropout(fused))


def encode_branches(
    encoder,
    separator: NoiseSeparator,
    mixture: torch.Tensor,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return ``z_mix``, ``z_noise`` and the raw noise estimate ``n_hat``.

    Both branches go through the same BEATs encoder in one doubled batch, so the
    transformer weights are shared by construction.
    """
    noise = separator(mixture)
    amplified = separator.amplify(noise)
    with mixed_precision_context(device):
        sequence, _ = encoder.extract_features(torch.cat([mixture, amplified], dim=0))
        z_mix, z_noise = sequence.mean(dim=1).split(mixture.shape[0])
    return z_mix, z_noise, noise

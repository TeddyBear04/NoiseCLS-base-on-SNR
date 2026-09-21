"""Concatenate frozen branch embeddings, then predict label and SNR."""

from __future__ import annotations

import torch
from torch import Tensor, nn


class FusionHead(nn.Module):
    """One LayerNorm per branch, then a shared MLP with two output heads.

    The same class serves the ablation rows: pass a one-entry ``branch_dims``
    and it becomes a single-branch probe with identical capacity per input unit.

    ``forward`` slices the incoming feature vector by walking ``self.branch_dims``
    (a dict, so insertion order is preserved) and advancing an offset. This is
    correct only because the caller builds ``branch_dims`` in the same order the
    cache concatenated its columns in (see ``BRANCH_ORDER`` in
    ``dataset.embedding_cache``). Passing the same branches in a different order
    silently slices every branch's LayerNorm over the wrong columns -- nothing
    raises, the model just trains on scrambled features.
    """

    def __init__(
        self,
        branch_dims: dict[str, int],
        hidden: tuple[int, int] = (512, 256),
        classes_num: int = 36,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        self.branch_dims = dict(branch_dims)
        self.norms = nn.ModuleDict(
            {name: nn.LayerNorm(dim) for name, dim in self.branch_dims.items()}
        )
        total = sum(self.branch_dims.values())
        self.trunk = nn.Sequential(
            nn.Linear(total, hidden[0]), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden[0], hidden[1]), nn.GELU(),
        )
        self.label_head = nn.Linear(hidden[1], classes_num)
        self.snr_head = nn.Linear(hidden[1], 1)

    def forward(self, features: Tensor) -> dict[str, Tensor]:
        pieces, start = [], 0
        for name, dim in self.branch_dims.items():
            pieces.append(self.norms[name](features[:, start:start + dim]))
            start += dim
        hidden = self.trunk(torch.cat(pieces, dim=1))
        return {"logits": self.label_head(hidden), "snr": self.snr_head(hidden).squeeze(-1)}

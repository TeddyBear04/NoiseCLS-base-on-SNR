from __future__ import annotations

import torch
from torch import Tensor, nn


def _init_linear(linear: nn.Linear, generator: torch.Generator) -> None:
    bound = (1.0 / linear.in_features) ** 0.5
    with torch.no_grad():
        linear.weight.uniform_(-bound, bound, generator=generator)
        linear.bias.uniform_(-bound, bound, generator=generator)


class Expert(nn.Module):
    """MLP xuất đủ logit cho mọi lớp; mỗi expert khởi tạo với seed riêng."""

    def __init__(self, embed_dim: int, hidden: int, num_classes: int, dropout: float = 0.1, seed: int = 0) -> None:
        super().__init__()
        self.fc1 = nn.Linear(embed_dim, hidden)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden, num_classes)
        generator = torch.Generator().manual_seed(seed)
        _init_linear(self.fc1, generator)
        _init_linear(self.fc2, generator)

    def forward(self, embedding: Tensor) -> Tensor:
        return self.fc2(self.dropout(self.act(self.fc1(embedding))))


class Experts(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        hidden: int,
        num_classes: int,
        num_groups: int = 4,
        dropout: float = 0.1,
        seed: int = 0,
    ) -> None:
        super().__init__()
        self.experts = nn.ModuleList(
            Expert(embed_dim, hidden, num_classes, dropout, seed + index) for index in range(num_groups)
        )

    def forward(self, embedding: Tensor) -> Tensor:
        return torch.stack([expert(embedding) for expert in self.experts], dim=1)

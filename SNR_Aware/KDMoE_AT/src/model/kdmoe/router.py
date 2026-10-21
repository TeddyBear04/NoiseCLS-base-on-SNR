from __future__ import annotations

from torch import Tensor, nn


class Router(nn.Module):
    """Embedding clip -> logit 4 nhóm; bias lớp cuối khởi tạo 0."""

    def __init__(self, embed_dim: int, hidden: int = 64, num_groups: int = 4) -> None:
        super().__init__()
        self.fc1 = nn.Linear(embed_dim, hidden)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden, num_groups)
        nn.init.zeros_(self.fc2.bias)

    def forward(self, embedding: Tensor) -> Tensor:
        return self.fc2(self.act(self.fc1(embedding)))

from __future__ import annotations

from torch import nn


class FeedForward(nn.Module):
    def __init__(self, dim=768, ffn_dim=3072, activation_dropout=0.0):
        super().__init__()
        self.fc1 = nn.Linear(dim, ffn_dim)
        self.fc2 = nn.Linear(ffn_dim, dim)
        self.act = nn.GELU()
        self.drop = nn.Dropout(activation_dropout)

    def forward(self, x):
        return self.fc2(self.drop(self.act(self.fc1(x))))

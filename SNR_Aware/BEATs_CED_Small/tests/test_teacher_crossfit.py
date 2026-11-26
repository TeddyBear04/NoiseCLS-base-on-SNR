from __future__ import annotations

import sys
from pathlib import Path

import torch
from torch import nn

HERE = Path(__file__).resolve().parents[1]
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from teacher_crossfit import crossfit_folds, rollout_hooks  # noqa: E402


class FakeAttention(nn.Module):
    """Same call contract as BEATs MultiheadAttention: returns (x, head-averaged weights or None, bias)."""

    def __init__(self, weights: torch.Tensor) -> None:
        super().__init__()
        self.weights = weights

    def forward(self, query, need_weights=False, need_head_weights=False):
        return query, (self.weights if need_weights else None), None


class FakeBlock(nn.Module):
    def __init__(self, weights: torch.Tensor) -> None:
        super().__init__()
        self.self_attn = FakeAttention(weights)

    def forward(self, x):
        return self.self_attn(query=x, need_weights=False)


def test_crossfit_folds_keep_every_clip_in_one_fold():
    rows = [{"noise_ytid": f"clip{i % 7}"} for i in range(42)]
    folds = crossfit_folds(rows, 5, seed=2026)
    assert set(folds) == {f"clip{i}" for i in range(7)}
    assert set(folds.values()) == set(range(5))
    assert crossfit_folds(rows, 5, seed=2026) == folds


def test_rollout_matches_hand_computation():
    torch.manual_seed(0)
    n = 6
    attentions = [torch.softmax(torch.randn(2, n, n), dim=-1) for _ in range(3)]
    blocks = [FakeBlock(a) for a in attentions]
    captured: list = []
    hooks = rollout_hooks(blocks, [2, 3], captured)
    x = torch.randn(n, 2, 4)
    for block in blocks:
        block(x)
    for hook in hooks:
        hook.remove()
    rollout, expected = torch.eye(n).expand(2, -1, -1), []
    for index, attention in enumerate(attentions, start=1):
        step = 0.5 * attention + 0.5 * torch.eye(n)
        rollout = (step / step.sum(-1, keepdim=True)) @ rollout
        if index in (2, 3):
            expected.append(rollout.mean(1).T)
    assert len(captured) == 2
    for got, want in zip(captured, expected):
        assert got.shape == (n, 2)
        assert torch.allclose(got, want, atol=1e-6)
        assert torch.allclose(got.sum(0), torch.ones(2), atol=1e-6)

from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parents[1]
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from reviewkd import (  # noqa: E402
    ReviewKDLoss,
    ReviewKDMoE,
    ReviewModule,
    hcl,
    to_teacher_grid,
    tokens_to_grid,
)
from src.loss import KDMoELossConfig  # noqa: E402
from src.loss.kdmoe import load_membership  # noqa: E402

YAML = HERE / "config" / "kdmoe_panns_cnn10_reviewkd.yaml"
# block CNN10 của PANNs cho 400 frame x 64 mel, bố cục (C, F', T')
PANNS_BLOCKS = [(64, 32, 200), (128, 16, 100), (256, 8, 50), (512, 4, 25)]


def reference_hcl(fstudent, fteacher):
    """github.com/dvlab-research/ReviewKD, CIFAR-100/model/reviewkd.py, verbatim."""
    loss_all = 0.0
    for fs, ft in zip(fstudent, fteacher):
        n, c, h, w = fs.shape
        loss = F.mse_loss(fs, ft, reduction="mean")
        cnt = 1.0
        tot = 1.0
        for l in [4, 2, 1]:
            if l >= h:
                continue
            tmpfs = F.adaptive_avg_pool2d(fs, (l, l))
            tmpft = F.adaptive_avg_pool2d(ft, (l, l))
            cnt /= 2.0
            loss += F.mse_loss(tmpfs, tmpft, reduction="mean") * cnt
            tot += cnt
        loss = loss / tot
        loss_all = loss_all + loss
    return loss_all


def test_teacher_grid_keeps_constant_maps_and_crops_low_band():
    for _, f, t in PANNS_BLOCKS:
        out = to_teacher_grid(torch.full((1, 3, f, t), 2.0), (8, 24), 0.845, 0.96)
        assert out.shape == (1, 3, 8, 24)
        assert torch.allclose(out, torch.full_like(out, 2.0))
    ramp = torch.arange(32, dtype=torch.float32).view(1, 1, 32, 1).expand(1, 1, 32, 200)
    out = to_teacher_grid(ramp, (8, 24), 0.845, 0.96)
    assert out.max() < 0.845 * 32  # dải mel trên 8 kHz bị cắt bỏ
    assert torch.all(out[0, 0, 1:] > out[0, 0, :-1])  # thứ tự tần số giữ nguyên


def test_review_module_maps_every_panns_block_to_teacher_grid():
    stages = [torch.randn(2, c, f, t) for c, f, t in PANNS_BLOCKS]
    outputs = ReviewModule()(stages)
    assert [tuple(o.shape) for o in outputs] == [(2, 768, 8, 24)] * 4


def test_hcl_matches_reference_code():
    torch.manual_seed(0)
    student = [torch.randn(3, 768, 8, 24) for _ in range(4)]
    teacher = [torch.randn(3, 768, 8, 24) for _ in range(4)]
    assert torch.allclose(hcl(student, teacher), reference_hcl(student, teacher), atol=1e-6)
    assert hcl(teacher, teacher).item() == 0.0


def test_tokens_to_grid_follows_time_major_token_order():
    tokens = torch.arange(24 * 8, dtype=torch.float32).view(1, 24 * 8, 1).expand(1, -1, 3)  # index t * 8 + f
    grid = tokens_to_grid(tokens.reshape(1, 24, 8, 3))
    assert grid.shape == (1, 3, 8, 24)
    assert grid[0, 0, 5, 7].item() == 7 * 8 + 5


def test_model_returns_review_maps_only_in_training():
    model = ReviewKDMoE.from_yaml(YAML)
    model.train()
    output = model(torch.randn(2, 64000) * 0.1, torch.randn(2, 64000) * 0.05)
    assert output.logits.shape == (2, 36) and output.router_logits.shape == (2, 3)
    assert output.noise_mel is not None  # stage 1 (L_front) của KD-MoE vẫn chạy
    assert [tuple(r.shape) for r in output.review] == [(2, 768, 8, 24)] * 4
    model.eval()
    assert model(torch.randn(1, 64000) * 0.1).review is None


def test_review_loss_adds_scaled_hcl_and_skips_without_teacher():
    torch.manual_seed(1)
    model = ReviewKDMoE.from_yaml(YAML).train()
    output = model(torch.randn(2, 64000) * 0.1)
    loss = ReviewKDLoss(KDMoELossConfig(kd_weight=0.0, route_ce_weight=1.0), load_membership(), 2.0)
    batch = {"label": torch.tensor([0, 5]), "group": torch.tensor([0, 2]), "noise_ytid": ["a", "b"]}
    plain = loss(output, batch)
    assert "review" not in plain and loss.last_review is None
    teacher = torch.randn(2, 4, 24, 8, 768).half()
    loss.review_scale = 0.5
    terms = loss(output, {**batch, "teacher_review": teacher})
    expected = hcl(output.review, [tokens_to_grid(teacher[:, i]) for i in range(4)])
    assert torch.allclose(terms["review"], expected)
    assert torch.allclose(terms["total"], plain["total"] + 2.0 * 0.5 * expected, atol=1e-5)

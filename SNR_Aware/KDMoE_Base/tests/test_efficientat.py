from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch
import yaml

HERE = Path(__file__).resolve().parents[1]
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from efficientat import EfficientATKDMoE  # noqa: E402
from src.model.kdmoe import KDMoEConfig  # noqa: E402

YAML = HERE / "config" / "kdmoe_mn10.yaml"
# repo EfficientAT clone ở đây khi chạy test (ngoài repo); EFFICIENTAT_REPO ghi đè
REPO = Path(os.environ.get("EFFICIENTAT_REPO", Path.home() / ".cache" / "efficientat_test_repo"))


@pytest.fixture(scope="module")
def model() -> EfficientATKDMoE:
    raw = yaml.safe_load(YAML.read_text(encoding="utf-8"))
    student = {**raw["student"], "repo_dir": str(REPO)}
    try:
        return EfficientATKDMoE(KDMoEConfig.from_dict(raw["model"]), raw.get("frontend"), student)
    except (subprocess.CalledProcessError, FileNotFoundError, OSError) as error:
        pytest.skip(f"EfficientAT repo unavailable: {error}")


def test_mn10_forward_shapes_and_front_target(model: EfficientATKDMoE):
    model.train()
    output = model(torch.randn(2, 64000) * 0.1, torch.randn(2, 64000) * 0.05)
    assert output.logits.shape == (2, 36) and output.router_logits.shape == (2, 3)
    assert output.e.shape == (2, 1280)
    assert output.gate.shape == (2, 4, 12, 960)  # lưới feature mn10: 128 mel / 32 x 384 frame / 32
    assert output.noise_mel.shape == output.noise_target.shape == (2, 1, 128, 384)


def test_mixture_and_noise_share_one_mel_draw(model: EfficientATKDMoE):
    model.train()
    wave = torch.randn(2, 64000) * 0.1
    torch.manual_seed(0)
    spec, target = model.frontend(wave, wave.clone())
    assert torch.allclose(spec, target)  # cùng fmin/fmax ngẫu nhiên cho mixture và nhiễu
    model.eval()
    spec_eval, none = model.frontend(wave)
    assert none is None and spec_eval.shape == (2, 1, 128, 384)


def test_mn10_size_matches_published_count(model: EfficientATKDMoE):
    encoder = sum(p.numel() for p in model.encoder.parameters()) + sum(p.numel() for p in model.post_fc.parameters())
    # README EfficientAT: mn10_as 4.88M gồm cả lớp 527 lớp AudioSet (1280 x 527 + 527) đã bỏ ở đây
    assert abs(encoder + 1280 * 527 + 527 - 4.88e6) / 4.88e6 < 0.02


def _variant(name: str):
    from src.loss import KDMoELossConfig

    path = HERE / "config" / f"kdmoe_mn10_{name}.yaml"
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    student = {**raw["student"], "repo_dir": str(REPO)}
    if "reviewkd" in raw:
        from efficientat import EfficientATReviewKDMoE as cls
    else:
        cls = EfficientATKDMoE
    try:
        model = cls(KDMoEConfig.from_dict(raw["model"]), raw.get("frontend"), student)
    except (subprocess.CalledProcessError, FileNotFoundError, OSError) as error:
        pytest.skip(f"EfficientAT repo unavailable: {error}")
    return model, KDMoELossConfig.from_yaml(path)

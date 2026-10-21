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

# repo EfficientAT clone ở đây khi chạy test (ngoài repo); EFFICIENTAT_REPO ghi đè
REPO = Path(os.environ.get("EFFICIENTAT_REPO", Path.home() / ".cache" / "efficientat_test_repo"))


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


def test_mn10_review_maps_only_in_training():
    model, _ = _variant("reviewkd")
    output = model.train()(torch.randn(2, 64000) * 0.1)
    assert [tuple(r.shape) for r in output.review] == [(2, 768, 8, 24)] * 4
    assert model.eval()(torch.randn(1, 64000) * 0.1).review is None

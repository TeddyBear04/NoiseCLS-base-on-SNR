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


def test_frcrn_snr_db_reads_energy_before_rms():
    from src.model.kdmoe import frcrn_snr_db

    n_hat = torch.randn(2, 64000) * 0.1
    speech = torch.randn(2, 64000) * 0.1
    mixture = torch.stack([speech[0] + n_hat[0], 10 ** 0.5 * speech[1] + n_hat[1]])  # 0 dB, +10 dB
    snr = frcrn_snr_db(mixture, n_hat)
    assert torch.allclose(snr, torch.tensor([0.0, 10.0]), atol=0.2)
    assert torch.allclose(frcrn_snr_db(3 * mixture, 3 * n_hat), snr, atol=1e-4)  # không phụ thuộc mức tuyệt đối


def test_mn10_snrroute_picks_one_expert_per_frcrn_snr_band():
    raw = yaml.safe_load((HERE / "config" / "kdmoe_mn10_snrroute.yaml").read_text(encoding="utf-8"))
    try:
        model = EfficientATKDMoE(KDMoEConfig.from_dict(raw["model"]), raw.get("frontend"),
                                 {**raw["student"], "repo_dir": str(REPO)})
    except (subprocess.CalledProcessError, FileNotFoundError, OSError) as error:
        pytest.skip(f"EfficientAT repo unavailable: {error}")
    assert model.router is None
    model.eval()
    wave = torch.randn(4, 64000) * 0.1
    output = model(wave, None, torch.tensor([-5.0, 4.9, 5.0, 20.0]))
    assert output.router_logits.argmax(-1).tolist() == [0, 0, 1, 2]
    with torch.no_grad():
        experts = model.experts(output.e)
    assert torch.allclose(output.logits, experts[torch.arange(4), torch.tensor([0, 0, 1, 2])], atol=1e-5)
    with pytest.raises(ValueError, match="snr_est"):
        model(wave)
    with pytest.raises(ValueError, match="route_by snr"):
        KDMoEConfig.from_dict({**raw["model"], "route_by": "class", "num_groups": 4})

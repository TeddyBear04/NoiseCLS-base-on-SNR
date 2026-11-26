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

    path = HERE / "config" / "unified" / f"kdmoe_mn10_{name}.yaml"
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


def test_mn10_attention_maps_and_hint_land_on_beats_grid():
    from src.loss.kdmoe import KDMoELoss, load_membership

    model, loss_config = _variant("at")
    output = model.train()(torch.randn(2, 64000) * 0.1)
    assert output.at_maps.shape == (2, 2, 24, 8)
    batch = {"label": torch.tensor([0, 5]), "group": torch.tensor([0, 2]), "noise_ytid": ["a", "b"],
             "teacher_logits36": torch.randn(2, 36), "teacher_at": torch.rand(2, 2, 24, 8)}
    assert KDMoELoss(loss_config, load_membership())(output, batch)["at"].item() > 0
    model, _ = _variant("fitnets")
    assert model.train()(torch.randn(2, 64000) * 0.1).hint.shape == (2, 24, 8, 768)


def test_mn10_review_maps_only_in_training():
    model, _ = _variant("reviewkd")
    output = model.train()(torch.randn(2, 64000) * 0.1)
    assert [tuple(r.shape) for r in output.review] == [(2, 768, 8, 24)] * 4
    assert model.eval()(torch.randn(1, 64000) * 0.1).review is None


def test_frcrn_snr_db_reads_energy_before_rms():
    from src.model.kdmoe import frcrn_snr_db

    n_hat = torch.randn(2, 64000) * 0.1
    speech = torch.randn(2, 64000) * 0.1
    mixture = torch.stack([speech[0] + n_hat[0], 10 ** 0.5 * speech[1] + n_hat[1]])  # 0 dB, +10 dB
    snr = frcrn_snr_db(mixture, n_hat)
    assert torch.allclose(snr, torch.tensor([0.0, 10.0]), atol=0.2)
    assert torch.allclose(frcrn_snr_db(3 * mixture, 3 * n_hat), snr, atol=1e-4)  # không phụ thuộc mức tuyệt đối


def test_mn10_snrroute_picks_one_expert_per_frcrn_snr_band():
    raw = yaml.safe_load((HERE / "config" / "unified" / "kdmoe_mn10_snrroute.yaml").read_text(encoding="utf-8"))
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

from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parents[1]
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

def test_rms_normalize_equalises_input_level_and_scales_noise_alike():
    import yaml
    from src.model.kdmoe import KDMoEConfig, KDMoEModel

    raw = yaml.safe_load((HERE / "config" / "kdmoe_panns_cnn10.yaml").read_text(encoding="utf-8"))
    torch.manual_seed(0)
    quiet = torch.randn(2, 64000) * 0.01
    model = KDMoEModel(KDMoEConfig.from_dict({**raw["model"], "rms_normalize": 0.07925}), raw.get("frontend")).eval()
    loud = model(quiet * 30.0)
    same = model(quiet)
    assert torch.allclose(loud.logits, same.logits, atol=1e-4)  # mức âm đầu vào không còn ảnh hưởng
    off = KDMoEModel(KDMoEConfig.from_dict({**raw["model"], "rms_normalize": 0.0}), raw.get("frontend")).eval()
    off.load_state_dict(model.state_dict())
    assert not torch.allclose(off(quiet * 30.0).logits, off(quiet).logits, atol=1e-4)
    model.train()
    out = model(quiet * 30.0, quiet * 15.0)
    ref = model.frontend((quiet * 30.0) * (0.07925 / (quiet * 30.0).square().mean(-1, keepdim=True).sqrt()),
                         (quiet * 15.0) * (0.07925 / (quiet * 30.0).square().mean(-1, keepdim=True).sqrt()))[1]
    assert torch.allclose(out.noise_target, ref, atol=1e-4)

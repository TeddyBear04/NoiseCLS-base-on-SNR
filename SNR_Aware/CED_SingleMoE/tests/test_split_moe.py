"""
Unit tests for SplitMoEFFN and CEDSingleMoESplitFFN Architecture.

Verifies:
1. SplitMoEFFN mathematical identity warm-start:
   At initialization, for ANY alpha in [0, 1], SplitMoEFFN(x, alpha) == BaseMLP(x).
2. Deep Blocks 10 & 11 use SplitMoEFFN.
3. Dual Classifier Head blending: z = (1 - alpha) * z_low + alpha * z_high.
"""

import pytest
import torch
import torch.nn as nn
from src.models.split_moe import CEDSingleMoESplitFFN, SplitMoEFFN
from src.models.vit_grep import ViTSBias


def test_split_moe_ffn_identity_warm_start():
    # Construct base MLP: 192 -> 768 -> 192 with GELU
    base_mlp = nn.Sequential(
        nn.Linear(192, 768),
        nn.GELU(),
        nn.Linear(768, 192),
    )
    x = torch.randn(4, 97, 192)
    y_base = base_mlp(x)

    # Đầu ra đúng bằng MLP gốc với mọi gates, cả 2 lẫn 3 expert
    for names in (("low", "high"), ("low", "mid", "high")):
        moe = SplitMoEFFN(base_mlp, names)
        for _ in range(5):
            gates = torch.softmax(torch.randn(4, len(names)) * 3, dim=-1)
            assert torch.allclose(moe(x, gates), y_base, atol=1e-5), f"Failed for {names}"


def test_model_architecture_and_deep_moe_blocks():
    model = CEDSingleMoESplitFFN(num_classes=36, split_blocks=(10, 11), expert_names=("low", "mid", "high"))

    # Blocks 0 through 9 are standard timm/ViT MLPs
    for i in range(10):
        assert not isinstance(model.vit_base.vit.blocks[i].mlp, SplitMoEFFN)

    # Deep Blocks 10 and 11 MUST be SplitMoEFFN with 3 routed experts
    for i in (10, 11):
        moe = model.vit_base.vit.blocks[i].mlp
        assert isinstance(moe, SplitMoEFFN)
        assert all(hasattr(moe, f"ffn_{name}") for name in ("shared", "low", "mid", "high"))

    # Một head mỗi expert
    for name in ("low", "mid", "high"):
        head = getattr(model, f"head_{name}")
        assert isinstance(head, nn.Linear) and head.out_features == 36
    assert model.router.mlp[-1].out_features == 3


def test_head_blending_follows_gates(sample_waveform):
    model = CEDSingleMoESplitFFN(num_classes=36, expert_names=("low", "mid", "high")).eval()
    with torch.no_grad():
        out = model(sample_waveform, return_dict=True)
    fused = (out["gates"].unsqueeze(-1) * out["logits_experts"]).sum(dim=1)
    assert torch.allclose(out["logits"], fused, atol=1e-5)
    assert torch.allclose(out["alpha"], out["gates"][:, -1:])


def test_router_input_raw_keeps_level_but_rms_does_not(sample_waveform):
    """router_input='raw': log-RMS của router theo biên độ n_hat (gợi ý SNR); 'rms' (bản gốc): log-RMS là hằng số."""
    quiet = sample_waveform * 0.01
    for router_input, level_kept in (("raw", True), ("rms", False)):
        model = CEDSingleMoESplitFFN(num_classes=36, expert_names=("low", "mid", "high"), router_input=router_input).eval()
        with torch.no_grad():
            loud_out, quiet_out = model(sample_waveform, return_dict=True), model(quiet, return_dict=True)
        diff = (loud_out["spectral_feat"][:, 0] - quiet_out["spectral_feat"][:, 0]).abs()
        assert bool((diff > 1.0).all()) == level_kept, router_input
        # Đặc trưng không phụ thuộc biên độ (crest, kurtosis, flatness, tỉ lệ dải) giữ nguyên
        assert torch.allclose(loud_out["spectral_feat"][:, 1:], quiet_out["spectral_feat"][:, 1:], atol=1e-3)
        # Nhánh mel vẫn chuẩn hoá RMS: CLS sau block 9 không đổi theo biên độ
        assert torch.allclose(loud_out["block9_token"], quiet_out["block9_token"], atol=1e-3)
    assert isinstance(model.router.feature_norm, nn.Identity)

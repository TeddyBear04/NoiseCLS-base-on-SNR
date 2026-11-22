"""
End-to-End Integration Tests for CEDSingleMoESplitFFN Pipeline.

Verifies:
1. Complete forward pass from raw waveform [B, 64000] to 36-class logits [B, 36].
2. Zero-Oracle operation: inference executes purely from waveform without SNR inputs.
3. Full gradient flow through entire computational graph.
"""

import pytest
import torch
from src.models.split_moe import CEDSingleMoESplitFFN


def test_end_to_end_forward_pass(sample_waveform):
    model = CEDSingleMoESplitFFN(num_classes=36, freq_bins=64, frames=398, expert_names=("low", "mid", "high"))
    model.eval()

    with torch.no_grad():
        out = model(sample_waveform, return_dict=True)

    # 1. Output dict fields
    assert "logits" in out
    assert "logits_experts" in out
    assert "alpha" in out
    assert "gates" in out
    assert "spectral_feat" in out
    assert "block9_token" in out

    # 2. Output shapes
    assert out["logits"].shape == (2, 36)
    assert out["logits_experts"].shape == (2, 3, 36)
    assert out["alpha"].shape == (2, 1)
    assert out["gates"].shape == (2, 3)
    assert out["spectral_feat"].shape == (2, 6)
    assert out["block9_token"].shape == (2, 192)

    # 3. Numeric validity
    assert not torch.isnan(out["logits"]).any()
    assert not torch.isinf(out["logits"]).any()
    assert (out["alpha"] >= 0.0).all() and (out["alpha"] <= 1.0).all()


def test_full_backward_gradient_flow(sample_waveform):
    model = CEDSingleMoESplitFFN(num_classes=36, freq_bins=64, frames=398, expert_names=("low", "mid", "high"))
    model.train()

    out = model(sample_waveform, return_dict=True)
    loss = out["logits"].sum() + out["alpha"].sum()
    loss.backward()

    # 1. Gradient in Early Backbone (Patch Embedding & Block 0)
    assert model.vit_base.vit.patch_embed.proj.weight.grad is not None
    assert model.vit_base.vit.blocks[0].mlp.fc1.weight.grad is not None

    # 2. Gradient in Router
    assert model.router.token_proj[1].weight.grad is not None
    assert model.router.mlp[0].weight.grad is not None
    assert model.router.mlp[3].weight.grad is not None

    # 3. Gradient in Deep MoE Experts (Blocks 10 & 11)
    moe10 = model.vit_base.vit.blocks[10].mlp
    assert moe10.ffn_shared.fc1.weight.grad is not None
    for name in ("low", "mid", "high"):
        assert getattr(moe10, f"ffn_{name}").fc1.weight.grad is not None

    # 4. Gradient in expert heads
    for name in ("low", "mid", "high"):
        assert getattr(model, f"head_{name}").weight.grad is not None

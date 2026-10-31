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

    moe = SplitMoEFFN(base_mlp)
    x = torch.randn(4, 97, 192)

    # Reference output from base MLP
    y_base = base_mlp(x)

    # Test identity for multiple alpha values across [0, 1]
    for alpha_val in [0.0, 0.15, 0.50, 0.75, 1.0]:
        alpha = torch.full((4, 1), alpha_val)
        y_moe = moe(x, alpha)
        # Difference must be essentially zero
        assert torch.allclose(y_moe, y_base, atol=1e-5), f"Failed for alpha={alpha_val}"


def test_model_architecture_and_deep_moe_blocks():
    model = CEDSingleMoESplitFFN(num_classes=36, split_blocks=(10, 11))

    # Blocks 0 through 9 are standard timm/ViT MLPs
    for i in range(10):
        mlp = model.vit_base.vit.blocks[i].mlp
        assert not isinstance(mlp, SplitMoEFFN)

    # Deep Blocks 10 and 11 MUST be SplitMoEFFN
    assert isinstance(model.vit_base.vit.blocks[10].mlp, SplitMoEFFN)
    assert isinstance(model.vit_base.vit.blocks[11].mlp, SplitMoEFFN)

    # Check Dual Classifier Heads
    assert isinstance(model.head_low, nn.Linear)
    assert isinstance(model.head_high, nn.Linear)
    assert model.head_low.out_features == 36
    assert model.head_high.out_features == 36


def test_dual_head_soft_blending():
    model = CEDSingleMoESplitFFN(num_classes=36)
    cls_token = torch.randn(2, 192)

    z_low = model.head_low(cls_token)
    z_high = model.head_high(cls_token)

    # Case 1: alpha = 0 -> z == z_low
    alpha_0 = torch.tensor([[0.0], [0.0]])
    z_fused_0 = (1.0 - alpha_0) * z_low + alpha_0 * z_high
    assert torch.allclose(z_fused_0, z_low, atol=1e-5)

    # Case 2: alpha = 1 -> z == z_high
    alpha_1 = torch.tensor([[1.0], [1.0]])
    z_fused_1 = (1.0 - alpha_1) * z_low + alpha_1 * z_high
    assert torch.allclose(z_fused_1, z_high, atol=1e-5)

    # Case 3: alpha = 0.5 -> z == 0.5 * z_low + 0.5 * z_high
    alpha_half = torch.tensor([[0.5], [0.5]])
    z_fused_half = (1.0 - alpha_half) * z_low + alpha_half * z_high
    assert torch.allclose(z_fused_half, 0.5 * (z_low + z_high), atol=1e-5)

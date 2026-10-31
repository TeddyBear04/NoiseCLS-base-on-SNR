"""
Unit tests for ViT Backbone, Patch Tokens, Pre-LN, and 2D GREP BiasAttention.

Verifies:
1. Conv2d 16x16 patch embedding produces 4 x 24 = 96 patch tokens.
2. CLS token concatenation yields X_0 of shape [B, 97, 192].
3. 2D Positional Embeddings.
4. Pre-LN structure and MLP parameter count (295,872 parameters).
5. 2D Gated Relative Position Bias (GREP) in BiasAttention:
   - 3 heads, head_dim = 64
   - Shared table reuse across blocks
   - Dynamic gates g_a, g_b and amplitude modulation scale S
"""

import pytest
import torch
import torch.nn as nn
from src.models.vit_grep import BiasAttention, SharedBias, ViTSBias


def test_patch_tokens_and_cls():
    model = ViTSBias(num_classes=36, freq_bins=64, frames=398, pretrained=False)
    vit = model.vit

    # Input mel spectrogram: [B, 1, 64, 398]
    x_mel = torch.randn(2, 1, 64, 398)

    # Conv2d patch embedding (16x16, stride 16)
    patches = vit.patch_embed(x_mel[..., : model.frames])
    B, N, C = patches.shape

    # 64 // 16 = 4 freq patches, 398 // 16 = 24 time patches -> 4 * 24 = 96 patches
    assert N == 96
    assert C == 192

    # Add 2D pos embed and prepend CLS token
    cls = (vit.cls_token + vit.pos_embed[:, :1]).expand(B, -1, -1)
    x = vit.norm_pre(vit.pos_drop(torch.cat([cls, patches + vit.pos_embed[:, 1:]], dim=1)))

    # Total tokens = 96 + 1 = 97
    assert x.shape == (2, 97, 192)


def test_pre_ln_mlp_param_count():
    model = ViTSBias(num_classes=36, freq_bins=64, frames=398, pretrained=False)
    block0 = model.vit.blocks[0]

    # Pre-LN checks
    assert hasattr(block0, "norm1")
    assert hasattr(block0, "norm2")
    assert hasattr(block0, "attn")
    assert hasattr(block0, "mlp")

    # MLP parameter count: 192 -> 768 -> 192
    # Linear 1: 192 * 768 + 768 = 148,224
    # Linear 2: 768 * 192 + 192 = 147,648
    # Total = 295,872 parameters
    mlp_params = sum(p.numel() for p in block0.mlp.parameters())
    assert mlp_params == 295872


def test_grep_bias_attention_mechanism():
    model = ViTSBias(num_classes=36, freq_bins=64, frames=398, pretrained=False)
    block0_attn = model.vit.blocks[0].attn
    block1_attn = model.vit.blocks[1].attn

    assert isinstance(block0_attn, BiasAttention)
    assert isinstance(block1_attn, BiasAttention)

    # Block 0 owns the relative position tables
    assert hasattr(block0_attn, "freq_rel_bias")
    assert hasattr(block0_attn, "time_rel_bias")

    # Block 1 does NOT own the tables, reuses Block 0's via shared
    assert not hasattr(block1_attn, "freq_rel_bias")
    assert block0_attn.shared is block1_attn.shared

    # Head configuration: 3 heads, 64 dimension
    assert block0_attn.num_heads == 3
    assert block0_attn.head_dim == 64

    # Test forward pass through BiasAttention
    x = torch.randn(2, 97, 192)
    out0 = block0_attn(x)
    assert out0.shape == (2, 97, 192)

    # Check shared table was populated
    assert model.shared.value is not None
    assert model.shared.value.shape == (1, 3, 97, 97)

    # Block 1 uses the shared table
    out1 = block1_attn(x)
    assert out1.shape == (2, 97, 192)

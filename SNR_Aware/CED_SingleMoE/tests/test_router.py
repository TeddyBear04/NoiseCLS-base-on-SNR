"""
Unit tests for Self-Contained Spectral-Token Router (Zero-Oracle SNR).

Verifies:
1. Router outputs K gates (softmax, sum 1) and alpha = gate of the highest-SNR expert.
2. Soft SNR-band targets: each row sums to 1; 3 bands put -5/0 dB on low, 5/10 dB on mid, 15/20 dB on high.
3. With one edge at 15 dB the targets and loss reduce to the original 2-expert alpha label and BCE.
"""

import torch
import torch.nn.functional as F
from src.router import SelfContainedSpectralTokenRouter, router_loss, snr_band_targets

SNRS = torch.tensor([-5.0, 0.0, 5.0, 10.0, 15.0, 20.0])


def test_router_forward_shapes_and_values():
    B = 4
    for k in (2, 3):
        router = SelfContainedSpectralTokenRouter(spectral_dim=6, token_dim=192, token_proj_dim=16, hidden_dim=32, num_experts=k)
        alpha, gates = router(torch.randn(B, 6), torch.randn(B, 192))
        assert alpha.shape == (B, 1)
        assert gates.shape == (B, k)
        assert (alpha > 0.0).all() and (alpha < 1.0).all()
        assert torch.allclose(gates.sum(dim=-1), torch.ones(B), atol=1e-5)
        assert torch.allclose(alpha, gates[:, -1:])


def test_three_band_targets():
    targets = snr_band_targets(SNRS, (2.5, 12.5), 2.5)
    assert targets.shape == (6, 3)
    assert torch.allclose(targets.sum(dim=-1), torch.ones(6), atol=1e-6)
    assert (targets >= 0).all()
    assert targets.argmax(dim=-1).tolist() == [0, 0, 1, 1, 2, 2]


def test_two_band_matches_original_alpha_label_and_bce():
    targets = snr_band_targets(SNRS, (15.0,), 2.5)
    w = torch.sigmoid((SNRS - 15.0) / 2.5)
    assert torch.allclose(targets[:, 1], w, atol=1e-6)
    alpha = torch.rand(6, 1) * 0.98 + 0.01
    gates = torch.cat([1.0 - alpha, alpha], dim=-1)
    assert torch.allclose(router_loss(gates, targets), F.binary_cross_entropy(alpha, w.unsqueeze(-1)), atol=1e-5)

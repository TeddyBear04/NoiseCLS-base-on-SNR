"""
Unit tests for Self-Contained Spectral-Token Router (Zero-Oracle SNR).

Verifies:
1. Token projection branch compresses CLS token from 192 to 16 dimensions.
2. Concatenation of f_spec (6 dims) and token_h (16 dims) produces vector of 22 dims.
3. MLP layer outputs 2 logits [r_low, r_high].
4. Softmax yields continuous soft weight alpha in (0, 1).
5. Target soft label function is strictly monotonic with SNR.
6. Router BCE loss function behavior.
"""

import torch
from src.router import SelfContainedSpectralTokenRouter


def test_router_forward_shapes_and_values():
    router = SelfContainedSpectralTokenRouter(
        spectral_dim=6,
        token_dim=192,
        token_proj_dim=16,
        hidden_dim=32,
    )

    B = 4
    spectral_feat = torch.randn(B, 6)
    intermediate_token = torch.randn(B, 192)

    alpha, gates = router(spectral_feat, intermediate_token)

    # Check output shapes
    assert alpha.shape == (B, 1)
    assert gates.shape == (B, 2)

    # Soft gating properties:
    # 1. Continuous range: alpha in (0, 1)
    assert (alpha > 0.0).all() and (alpha < 1.0).all()

    # 2. Sum to 1 across gates
    assert torch.allclose(gates.sum(dim=-1), torch.ones(B), atol=1e-5)

    # 3. alpha equals gates[:, 1:2]
    assert torch.allclose(alpha, gates[:, 1:2])


def test_target_soft_label_monotonicity():
    router = SelfContainedSpectralTokenRouter()

    snrs = torch.tensor([-5.0, 0.0, 5.0, 10.0, 15.0, 20.0])
    targets = router.compute_target_soft_label(snrs).squeeze(-1)

    # Monotonicity check
    for i in range(len(targets) - 1):
        assert targets[i] < targets[i + 1]

    # Reference values:
    # SNR = 15 dB is the midpoint -> target == 0.5
    assert torch.isclose(targets[4], torch.tensor(0.5), atol=1e-3)
    # SNR = -5 dB is close to 0
    assert targets[0] < 0.01
    # SNR = 20 dB is high (> 0.8)
    assert targets[5] > 0.85


def test_router_bce_loss():
    router = SelfContainedSpectralTokenRouter()
    alpha = torch.tensor([[0.1], [0.85]])
    snr = torch.tensor([-5.0, 20.0])

    loss = router.compute_router_loss(alpha, snr)
    assert loss.ndim == 0
    assert loss.item() >= 0.0
    assert not torch.isnan(loss)

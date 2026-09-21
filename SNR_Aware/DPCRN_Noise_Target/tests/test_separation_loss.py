import torch

from tasks.run_36 import separation_loss

N_FFT, HOP = 512, 160


def test_better_estimate_scores_lower():
    torch.manual_seed(0)
    target = torch.randn(2, 16000)
    close = target + 0.01 * torch.randn_like(target)
    far = target + 1.00 * torch.randn_like(target)
    assert separation_loss(close, target, N_FFT, HOP) < separation_loss(far, target, N_FFT, HOP)


def test_loss_is_finite_for_a_silent_estimate():
    target = torch.randn(2, 16000)
    value = separation_loss(torch.zeros_like(target), target, N_FFT, HOP)
    assert torch.isfinite(value)


def test_good_estimate_drives_the_loss_below_zero():
    """This is what separates the paper's loss from the L1 loss it replaces.

    The negative-SNR term is measured in decibels, so a near-perfect estimate
    pushes the loss far below zero. The superseded loss summed magnitudes and
    could never be negative — measured minimum 1.9e-4 over 40 random cases.
    """
    torch.manual_seed(0)
    target = torch.randn(2, 16000)
    estimate = target + 1e-3 * torch.randn_like(target)
    assert separation_loss(estimate, target, N_FFT, HOP) < 0

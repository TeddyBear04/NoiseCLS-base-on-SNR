from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from torch import Tensor
from torch.nn import functional as F

from .base import SeparationLoss

if TYPE_CHECKING:
    from src.model.frcrn.config import FRCRNConfig


def si_snr(reference: Tensor, estimate: Tensor, eps: float = 1e-6) -> Tensor:
    reference = reference - reference.mean(dim=-1, keepdim=True)
    estimate = estimate - estimate.mean(dim=-1, keepdim=True)
    energy = reference.square().sum(dim=-1, keepdim=True) + eps
    projection = (reference * estimate).sum(dim=-1, keepdim=True) * reference / energy
    residual = estimate - projection
    ratio = projection.square().sum(dim=-1) / (residual.square().sum(dim=-1) + eps)
    return 10.0 * torch.log10(ratio + eps)


def complex_ideal_ratio_mask(
    target: Tensor,
    mixture: Tensor,
    config: FRCRNConfig,
    mask_bound: float = 2.0,
    eps: float = 1e-8,
) -> Tensor:
    window = torch.hann_window(config.win_len, periodic=False, device=mixture.device, dtype=mixture.dtype)

    def analyze(waveform: Tensor) -> Tensor:
        return torch.stft(
            waveform,
            config.fft_len,
            config.win_inc,
            config.win_len,
            window=window,
            center=False,
            return_complex=True,
        )

    target_spectrum, mixture_spectrum = analyze(target), analyze(mixture)
    ratio = target_spectrum * mixture_spectrum.conj() / (mixture_spectrum.abs().square() + eps)
    mask = torch.cat([ratio.real, ratio.imag], dim=1)
    mask = torch.where(mask > mask_bound, torch.ones_like(mask), mask)
    return torch.where(mask < -mask_bound, -torch.ones_like(mask), mask)


class FRCRNLoss(SeparationLoss):
    name = "frcrn"

    def __init__(self, config: FRCRNConfig, mask_bound: float = 2.0, eps: float = 1e-6, mask_eps: float = 1e-8) -> None:
        super().__init__()
        self.config = config
        self.mask_bound = mask_bound
        self.eps = eps
        self.mask_eps = mask_eps

    def forward(self, mixture: Tensor, target: Tensor, estimate: Tensor, mask: Tensor) -> Tensor:
        ideal = complex_ideal_ratio_mask(target, mixture, self.config, self.mask_bound, self.mask_eps)
        bins = self.config.num_bins
        mask_loss = (
            F.mse_loss(mask[:, :bins], ideal[:, :bins]) + F.mse_loss(mask[:, bins:], ideal[:, bins:])
        ) * self.config.fft_len
        return mask_loss - si_snr(target, estimate, self.eps).mean()

"""STFT-mask noise separator, noise amplification, and SI-SDR loss."""

from __future__ import annotations

import torch
from torch import nn


class DilatedBlock(nn.Module):
    """Residual depthwise-separable dilated convolution over STFT frames."""

    def __init__(self, channels: int, kernel_size: int, dilation: int) -> None:
        super().__init__()
        padding = dilation * (kernel_size - 1) // 2
        self.net = nn.Sequential(
            nn.Conv1d(
                channels, channels, kernel_size,
                padding=padding, dilation=dilation, groups=channels,
            ),
            nn.PReLU(),
            nn.GroupNorm(1, channels),
            nn.Conv1d(channels, channels, 1),
        )

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return hidden + self.net(hidden)


class NoiseSeparator(nn.Module):
    """Estimate the noise waveform ``n_hat`` from a mixture ``[batch, samples]``.

    A small temporal convolution network predicts a sigmoid mask over the
    mixture STFT magnitude; the masked STFT is inverted with the mixture phase.
    """

    def __init__(
        self,
        n_fft: int = 512,
        hop_length: int = 160,
        win_length: int = 400,
        channels: int = 256,
        blocks: int = 8,
        kernel_size: int = 3,
        target_rms: float = 0.1,
        max_gain_db: float = 40.0,
    ) -> None:
        super().__init__()
        self.config = {
            "n_fft": n_fft,
            "hop_length": hop_length,
            "win_length": win_length,
            "channels": channels,
            "blocks": blocks,
            "kernel_size": kernel_size,
            "target_rms": target_rms,
            "max_gain_db": max_gain_db,
        }
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.target_rms = target_rms
        self.max_gain = 10 ** (max_gain_db / 20)
        self.register_buffer("window", torch.hann_window(win_length), persistent=False)
        frequency_bins = n_fft // 2 + 1
        self.input = nn.Sequential(
            nn.Conv1d(frequency_bins, channels, 1),
            nn.PReLU(),
            nn.GroupNorm(1, channels),
        )
        self.blocks = nn.Sequential(
            *(DilatedBlock(channels, kernel_size, 2**index) for index in range(blocks))
        )
        self.mask = nn.Conv1d(channels, frequency_bins, 1)

    def forward(self, mixture: torch.Tensor) -> torch.Tensor:
        # STFT/ISTFT do not support BF16; keep the whole separator in FP32.
        with torch.autocast(device_type=mixture.device.type, enabled=False):
            mixture = mixture.float()
            spectrum = torch.stft(
                mixture, self.n_fft, self.hop_length, self.win_length,
                window=self.window, center=True, return_complex=True,
            )
            features = torch.log(spectrum.abs().clamp_min(1e-5))
            mask = torch.sigmoid(self.mask(self.blocks(self.input(features))))
            return torch.istft(
                spectrum * mask, self.n_fft, self.hop_length, self.win_length,
                window=self.window, center=True, length=mixture.shape[-1],
            )

    def set_amplification(self, target_rms: float | None = None, max_gain_db: float | None = None) -> None:
        """Change the parameter-free loudness normalisation without retraining."""
        if target_rms is not None:
            self.target_rms = target_rms
            self.config["target_rms"] = target_rms
        if max_gain_db is not None:
            self.max_gain = 10 ** (max_gain_db / 20)
            self.config["max_gain_db"] = max_gain_db

    def amplify(self, noise: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
        """Bring every estimated noise clip to ``target_rms`` before BEATs.

        Quiet high-SNR residuals are boosted by at most ``max_gain_db`` so that
        near-silent estimates are not blown up to pure artefacts. The gain is
        detached so it acts as a fixed loudness normalisation, and the result
        is peak-limited to [-1, 1] like a regular waveform.
        """
        rms = noise.float().pow(2).mean(dim=-1, keepdim=True).add(eps).sqrt()
        gain = (self.target_rms / rms).clamp(max=self.max_gain).detach()
        amplified = noise * gain
        peak = amplified.abs().amax(dim=-1, keepdim=True).detach()
        return amplified / peak.clamp(min=1.0)


def si_sdr(estimate: torch.Tensor, reference: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Scale-invariant SDR in dB for every clip, shaped ``[batch]``."""
    estimate = estimate.float()
    reference = reference.float()
    estimate = estimate - estimate.mean(dim=-1, keepdim=True)
    reference = reference - reference.mean(dim=-1, keepdim=True)
    scale = (estimate * reference).sum(dim=-1, keepdim=True) / (
        reference.pow(2).sum(dim=-1, keepdim=True) + eps
    )
    target = scale * reference
    residual = estimate - target
    return 10 * torch.log10(
        (target.pow(2).sum(dim=-1) + eps) / (residual.pow(2).sum(dim=-1) + eps)
    )


def separation_loss(estimate: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
    """Negative mean SI-SDR between ``n_hat`` and the oracle noise."""
    return -si_sdr(estimate, reference).mean()


def separator_payload(separator: NoiseSeparator) -> dict:
    """Serializable architecture and weights for embedding in any checkpoint."""
    return {
        "config": dict(separator.config),
        "state": {
            key: value.detach().cpu().clone()
            for key, value in separator.state_dict().items()
        },
    }


def build_separator(payload: dict) -> NoiseSeparator:
    separator = NoiseSeparator(**payload["config"])
    separator.load_state_dict(payload["state"])
    return separator

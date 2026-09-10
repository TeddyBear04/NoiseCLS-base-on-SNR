"""Pure-PyTorch 16 kHz log-mel frontend with SpecAugment."""

from __future__ import annotations

import logging
from typing import Optional

import torch
import torch.nn as nn

from config import AudioFeaturesConfig as AudioFrontendConfig

logger = logging.getLogger(__name__)


def init_bn(batch_norm: nn.BatchNorm2d) -> None:
    if batch_norm.bias is not None:
        nn.init.zeros_(batch_norm.bias)
    if batch_norm.weight is not None:
        nn.init.ones_(batch_norm.weight)


def _hz_to_mel(frequency: torch.Tensor) -> torch.Tensor:
    return 2595.0 * torch.log10(1.0 + frequency / 700.0)


def _mel_to_hz(mel: torch.Tensor) -> torch.Tensor:
    return 700.0 * (torch.pow(10.0, mel / 2595.0) - 1.0)


def create_mel_filter(
    sample_rate: int,
    n_fft: int,
    n_mels: int,
    fmin: float,
    fmax: float,
) -> torch.Tensor:
    """Return triangular mel filters shaped [frequency_bins, mel_bins]."""
    min_mel = _hz_to_mel(torch.tensor(float(fmin)))
    max_mel = _hz_to_mel(torch.tensor(float(fmax)))
    mel_points = torch.linspace(min_mel, max_mel, n_mels + 2)
    hz_points = _mel_to_hz(mel_points)
    fft_frequencies = torch.linspace(0.0, sample_rate / 2.0, n_fft // 2 + 1)
    filters = torch.zeros(n_fft // 2 + 1, n_mels)
    for mel_index in range(n_mels):
        left, center, right = hz_points[mel_index : mel_index + 3]
        up_slope = (fft_frequencies - left) / max(float(center - left), 1e-12)
        down_slope = (right - fft_frequencies) / max(float(right - center), 1e-12)
        filters[:, mel_index] = torch.clamp(torch.minimum(up_slope, down_slope), min=0.0)
    # Area normalization keeps magnitudes comparable across different mel widths.
    enorm = 2.0 / torch.clamp(hz_points[2 : n_mels + 2] - hz_points[:n_mels], min=1e-12)
    return filters * enorm.unsqueeze(0)


class AudioFrontend(nn.Module):
    def __init__(self, config: Optional[AudioFrontendConfig] = None) -> None:
        super().__init__()
        self.config = config or AudioFrontendConfig()
        self.mel_bins = self.config.mel_bins
        self.register_buffer("window", torch.hann_window(self.config.window_size), persistent=False)
        self.register_buffer(
            "mel_filter",
            create_mel_filter(
                self.config.sample_rate,
                self.config.window_size,
                self.config.mel_bins,
                self.config.fmin,
                self.config.fmax,
            ),
            persistent=False,
        )
        self.bn0 = nn.BatchNorm2d(self.mel_bins)
        init_bn(self.bn0)
        logger.info(
            "Audio frontend: sr=%d n_fft=%d hop=%d mels=%d fmin=%d fmax=%d",
            self.config.sample_rate,
            self.config.window_size,
            self.config.hop_size,
            self.config.mel_bins,
            self.config.fmin,
            self.config.fmax,
        )

    @staticmethod
    def _mask_axis(x: torch.Tensor, axis: int, max_width: int, stripe_count: int) -> torch.Tensor:
        if max_width <= 0 or stripe_count <= 0:
            return x
        axis_size = x.size(axis)
        for _ in range(stripe_count):
            width = int(torch.randint(0, min(max_width, axis_size) + 1, (1,), device=x.device))
            if width == 0:
                continue
            start = int(torch.randint(0, axis_size - width + 1, (1,), device=x.device))
            slices = [slice(None)] * x.ndim
            slices[axis] = slice(start, start + width)
            x[tuple(slices)] = 0.0
        return x

    def _spec_augment(self, x: torch.Tensor) -> torch.Tensor:
        x = x.clone()
        x = self._mask_axis(
            x, axis=2, max_width=self.config.time_drop_width, stripe_count=self.config.time_stripes_num
        )
        x = self._mask_axis(
            x, axis=3, max_width=self.config.freq_drop_width, stripe_count=self.config.freq_stripes_num
        )
        return x

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        if input_tensor.ndim != 2:
            raise ValueError(f"Expected waveform [batch, samples], got {tuple(input_tensor.shape)}")
        stft = torch.stft(
            input_tensor,
            n_fft=self.config.window_size,
            hop_length=self.config.hop_size,
            win_length=self.config.window_size,
            window=self.window,
            center=True,
            pad_mode="reflect",
            normalized=False,
            onesided=True,
            return_complex=True,
        )
        power = stft.abs().square().transpose(1, 2)  # [batch, time, frequency]
        mel = torch.matmul(power, self.mel_filter)
        logmel = 10.0 * torch.log10(mel.clamp_min(1e-10))
        x = logmel.unsqueeze(1)  # [batch, 1, time, mel]
        x = nn.functional.pad(x, (0, 0, 2, 0))
        x = self.bn0(x.transpose(1, 3)).transpose(1, 3)
        if self.training:
            x = self._spec_augment(x)
        return x

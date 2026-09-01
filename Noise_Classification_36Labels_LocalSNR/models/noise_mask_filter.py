"""U-Net complex-ratio-mask filter that pulls the noise out of a mixture.

The classifier's label comes from the noise stem alone, so handing it the extracted
noise instead of the mixture removes the speech it would otherwise have to learn to
ignore. Everything downstream of the mask - the speech residual, the clip-level SNR,
the gain, the per-segment SNR - is arithmetic on the mask's own output, not a second
network.
"""

from __future__ import annotations

from typing import Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as F

from config import AudioFeaturesConfig, FilterConfig, LocalSNRConfig

EPS = 1e-8


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.SiLU(),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.SiLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class NoiseMaskUNet(nn.Module):
    """Three-level U-Net predicting a bounded complex ratio mask."""

    def __init__(self, base_channels: int = 16) -> None:
        super().__init__()
        c1, c2, c3 = base_channels, base_channels * 2, base_channels * 4
        self.e1, self.e2, self.e3 = ConvBlock(2, c1), ConvBlock(c1, c2), ConvBlock(c2, c3)
        self.pool = nn.MaxPool2d(2)
        self.up2, self.d2 = nn.ConvTranspose2d(c3, c2, 2, 2), ConvBlock(c3, c2)
        self.up1, self.d1 = nn.ConvTranspose2d(c2, c1, 2, 2), ConvBlock(c2, c1)
        self.out = nn.Conv2d(c1, 2, 1)

    @staticmethod
    def _fit(x: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
        return F.interpolate(x, size=reference.shape[-2:], mode="bilinear", align_corners=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        a = self.e1(x)
        b = self.e2(self.pool(a))
        z = self.e3(self.pool(b))
        z = self.d2(torch.cat([self._fit(self.up2(z), b), b], dim=1))
        z = self.d1(torch.cat([self._fit(self.up1(z), a), a], dim=1))
        return torch.tanh(self.out(z))


class NoiseMaskFilter(nn.Module):
    """Mixture in, extracted noise out, plus everything derived from it."""

    def __init__(self, config: FilterConfig, audio_config: AudioFeaturesConfig) -> None:
        super().__init__()
        self.config = config
        self.audio_config = audio_config
        self.n_fft = audio_config.window_size
        self.hop_length = audio_config.hop_size
        self.gain_max = 10.0 ** (config.gain_max_db / 20.0)
        self.unet = NoiseMaskUNet(config.base_channels)
        self.register_buffer("window", torch.hann_window(self.n_fft), persistent=False)

    # ------------------------------------------------------------------ transforms

    def stft(self, waveform: torch.Tensor) -> torch.Tensor:
        return torch.stft(
            waveform,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.n_fft,
            window=self.window.to(waveform.dtype),
            center=True,
            pad_mode="reflect",
            return_complex=True,
        )

    def istft(self, spectrum: torch.Tensor, length: int) -> torch.Tensor:
        return torch.istft(
            spectrum,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.n_fft,
            window=self.window.to(spectrum.real.dtype),
            center=True,
            length=length,
        )

    def predict_mask(self, spectrum: torch.Tensor) -> torch.Tensor:
        """Bounded complex ratio mask, same shape as the spectrum."""
        stacked = torch.stack([spectrum.real, spectrum.imag], dim=1)
        mask = self.unet(stacked)
        return torch.complex(mask[:, 0], mask[:, 1])

    # ----------------------------------------------------------------------- gain

    def estimate_gain(
        self, noise_spec: torch.Tensor, mixture_spec: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """How far to lift the extracted noise, and the clip SNR that decided it.

        Both come back detached on purpose. With a live gradient the filter can shrink
        the noise it reports purely to earn a larger gain, which is not what the mask
        is supposed to learn.
        """
        dims = tuple(range(1, noise_spec.ndim))
        noise_energy = noise_spec.abs().square().mean(dims, keepdim=True)
        speech_energy = (mixture_spec - noise_spec).abs().square().mean(dims, keepdim=True)
        snr_db = 10.0 * torch.log10((speech_energy + EPS) / (noise_energy + EPS))

        if not self.config.adaptive_gain:
            return torch.ones_like(snr_db), snr_db.detach()

        threshold = self.config.snr_threshold_db
        if self.config.gain_mode == "sigmoid":
            gate = torch.sigmoid((snr_db - threshold) / self.config.snr_temperature_db)
            gain = 1.0 + (self.gain_max - 1.0) * gate
        else:
            # Bring anything above the threshold down to the noise level it would have
            # had at the threshold, so every high-SNR clip arrives at the same loudness.
            target_energy = speech_energy / (10.0 ** (threshold / 10.0))
            requested = torch.sqrt(target_energy / (noise_energy + EPS)).clamp(1.0, self.gain_max)
            gain = torch.where(snr_db > threshold, requested, torch.ones_like(requested))
        return gain.detach(), snr_db.detach()

    # --------------------------------------------------------------- analytic SNR

    @staticmethod
    def segment_snr_db(
        speech: torch.Tensor,
        noise: torch.Tensor,
        sample_rate: int,
        config: LocalSNRConfig,
    ) -> torch.Tensor:
        """Per-segment SNR in dB, matching how the dataloader builds its targets.

        Differentiable, so the local-SNR loss still reaches the mask even though no
        regression head sits in between.
        """
        from dataset import local_snr_segment_starts  # circular at module import time

        segment_samples = int(round(sample_rate * config.segment_seconds))
        starts: List[int] = local_snr_segment_starts(speech.shape[-1], sample_rate, config)
        speech_power = torch.stack(
            [speech[..., s : s + segment_samples].square().mean(-1) for s in starts], dim=-1
        )
        noise_power = torch.stack(
            [noise[..., s : s + segment_samples].square().mean(-1) for s in starts], dim=-1
        )
        return 10.0 * torch.log10((speech_power + 1e-12) / (noise_power + 1e-12))

    # -------------------------------------------------------------------- forward

    def forward(self, waveform: torch.Tensor) -> Dict[str, torch.Tensor]:
        length = waveform.shape[-1]
        mixture_spec = self.stft(waveform)
        noise_spec = self.predict_mask(mixture_spec) * mixture_spec
        gain, snr_db = self.estimate_gain(noise_spec, mixture_spec)

        est_noise = self.istft(noise_spec, length=length)
        amplified = self.istft(noise_spec * gain, length=length)
        condition = torch.cat(
            [
                torch.log(gain.clamp_min(EPS)).flatten(1),
                (snr_db / 20.0).clamp(-2.0, 2.0).flatten(1),
            ],
            dim=1,
        )
        return {
            "est_noise": est_noise,
            "est_noise_amplified": amplified,
            "est_speech": waveform - est_noise,
            "gain": gain.flatten(1),
            "snr_db": snr_db.flatten(1),
            "condition": condition,
        }

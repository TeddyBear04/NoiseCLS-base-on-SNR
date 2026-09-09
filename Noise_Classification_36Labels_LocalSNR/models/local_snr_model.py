"""BlackFeather-inspired multi-task model for supervised noise extraction.

The deployable path only needs the mixture.  Oracle noise is accepted during
training solely as optional teacher forcing for the noise encoder; extraction
and the physics-based SNR estimate always come from the predicted noise.
"""

from __future__ import annotations

import math
from typing import Dict, Literal

import torch
import torch.nn as nn
import torch.nn.functional as F

from config import AudioFeaturesConfig, ModelConfig
from features import AudioFrontend


class DepthwiseResidual2d(nn.Module):
    def __init__(self, channels: int, dilation: int = 1) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.GroupNorm(1, channels),
            nn.GELU(),
            nn.Conv2d(
                channels,
                channels,
                kernel_size=3,
                padding=dilation,
                dilation=dilation,
                groups=channels,
                bias=False,
            ),
            nn.Conv2d(channels, channels, kernel_size=1, bias=False),
        )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return value + self.block(value)


class SpectralNoiseExtractor(nn.Module):
    """Predict a complex ratio mask and reconstruct a noise waveform."""

    def __init__(
        self,
        audio_config: AudioFeaturesConfig,
        channels: int = 32,
        depth: int = 4,
    ) -> None:
        super().__init__()
        self.n_fft = audio_config.window_size
        self.hop_length = audio_config.hop_size
        self.register_buffer("window", torch.hann_window(self.n_fft), persistent=False)
        layers: list[nn.Module] = [
            nn.Conv2d(2, channels, kernel_size=3, padding=1, bias=False)
        ]
        layers.extend(
            DepthwiseResidual2d(channels, dilation=2 ** (index % 3))
            for index in range(depth)
        )
        layers.extend(
            [
                nn.GroupNorm(1, channels),
                nn.GELU(),
                nn.Conv2d(channels, 2, kernel_size=1),
            ]
        )
        self.mask_network = nn.Sequential(*layers)

    def forward(self, mixture: torch.Tensor) -> torch.Tensor:
        if mixture.ndim != 2:
            raise ValueError(f"Expected [batch, samples], got {tuple(mixture.shape)}")
        spectrum = torch.stft(
            mixture,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.n_fft,
            window=self.window,
            center=True,
            normalized=False,
            onesided=True,
            return_complex=True,
        )
        network_input = torch.stack((spectrum.real, spectrum.imag), dim=1)
        raw_mask = torch.tanh(self.mask_network(network_input))
        complex_mask = torch.complex(raw_mask[:, 0], raw_mask[:, 1])
        estimate_spectrum = spectrum * complex_mask
        return torch.istft(
            estimate_spectrum,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.n_fft,
            window=self.window,
            center=True,
            normalized=False,
            onesided=True,
            length=mixture.shape[-1],
        )


class LogMelSequenceEncoder(nn.Module):
    """Preserve the time axis while compressing frequency into embeddings."""

    def __init__(
        self,
        audio_config: AudioFeaturesConfig,
        channels: int,
        embedding_dim: int,
    ) -> None:
        super().__init__()
        self.frontend = AudioFrontend(audio_config)
        self.convolutions = nn.Sequential(
            nn.Conv2d(1, channels, 3, padding=1, bias=False),
            nn.GroupNorm(1, channels),
            nn.GELU(),
            nn.Conv2d(channels, channels * 2, 3, stride=(1, 2), padding=1, bias=False),
            nn.GroupNorm(1, channels * 2),
            nn.GELU(),
            nn.Conv2d(
                channels * 2,
                channels * 2,
                3,
                stride=(1, 2),
                padding=1,
                groups=channels * 2,
                bias=False,
            ),
            nn.Conv2d(channels * 2, channels * 4, 1, bias=False),
            nn.GroupNorm(1, channels * 4),
            nn.GELU(),
            nn.Conv2d(
                channels * 4,
                channels * 4,
                3,
                stride=(1, 2),
                padding=1,
                groups=channels * 4,
                bias=False,
            ),
            nn.GELU(),
        )
        self.projection = nn.Linear(channels * 4, embedding_dim)
        self.normalization = nn.LayerNorm(embedding_dim)

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        value = self.convolutions(self.frontend(waveform))
        value = value.mean(dim=-1).transpose(1, 2)
        return self.normalization(self.projection(value))


class GatedFusion(nn.Module):
    """Noise-dominant fusion prevents the mixture branch from bypassing extraction."""

    def __init__(self, embedding_dim: int) -> None:
        super().__init__()
        self.mixture_projection = nn.Linear(embedding_dim, embedding_dim)
        self.gate = nn.Linear(embedding_dim * 2, embedding_dim)
        self.normalization = nn.LayerNorm(embedding_dim)

    def forward(
        self, mixture_features: torch.Tensor, noise_features: torch.Tensor
    ) -> torch.Tensor:
        gate = torch.sigmoid(self.gate(torch.cat((mixture_features, noise_features), dim=-1)))
        return self.normalization(
            noise_features + gate * self.mixture_projection(mixture_features)
        )


def physical_local_snr(
    speech: torch.Tensor,
    noise: torch.Tensor,
    frame_count: int,
    minimum_db: float = -30.0,
    maximum_db: float = 30.0,
) -> torch.Tensor:
    """Differentiable local speech-to-noise ratio from estimated components."""
    speech_power = F.adaptive_avg_pool1d(speech.square().unsqueeze(1), frame_count).squeeze(1)
    noise_power = F.adaptive_avg_pool1d(noise.square().unsqueeze(1), frame_count).squeeze(1)
    eps = torch.finfo(speech.dtype).eps
    value = 10.0 * torch.log10((speech_power + eps) / (noise_power + eps))
    return value.clamp(minimum_db, maximum_db)


class BlackFeatherLocalSNR(nn.Module):
    """Noise extraction + two encoders + gated classification + Local-SNR."""

    def __init__(
        self,
        audio_config: AudioFeaturesConfig,
        model_config: ModelConfig,
    ) -> None:
        super().__init__()
        self.local_snr_frames = max(
            1,
            math.ceil(
                audio_config.clip_seconds / audio_config.local_snr_segment_seconds
            ),
        )
        self.max_snr_correction_db = model_config.max_snr_correction_db
        self.mixture_branch_dropout = model_config.mixture_branch_dropout
        self.noise_extractor = SpectralNoiseExtractor(
            audio_config,
            channels=model_config.extractor_channels,
            depth=model_config.extractor_depth,
        )
        self.mixture_encoder = LogMelSequenceEncoder(
            audio_config, model_config.encoder_channels, model_config.embedding_dim
        )
        self.noise_encoder = LogMelSequenceEncoder(
            audio_config, model_config.encoder_channels, model_config.embedding_dim
        )
        self.fusion = GatedFusion(model_config.embedding_dim)
        self.classifier = nn.Sequential(
            nn.Linear(model_config.embedding_dim * 2, model_config.embedding_dim),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(model_config.embedding_dim, model_config.classes_num),
        )
        self.snr_correction = nn.Sequential(
            nn.Linear(model_config.embedding_dim * 2, model_config.embedding_dim),
            nn.GELU(),
            nn.Linear(model_config.embedding_dim, 1),
        )

    @staticmethod
    def _pool_for_classification(features: torch.Tensor) -> torch.Tensor:
        return torch.cat((features.mean(dim=1), features.amax(dim=1)), dim=-1)

    @staticmethod
    def _pool_frames(features: torch.Tensor, frame_count: int) -> torch.Tensor:
        return F.adaptive_avg_pool1d(features.transpose(1, 2), frame_count).transpose(1, 2)

    def set_stage(self, stage: Literal["extractor", "heads", "joint"]) -> None:
        if stage not in {"extractor", "heads", "joint"}:
            raise ValueError(f"Unknown training stage: {stage}")
        for parameter in self.parameters():
            parameter.requires_grad_(stage == "joint")
        if stage == "extractor":
            for parameter in self.noise_extractor.parameters():
                parameter.requires_grad_(True)
        elif stage == "heads":
            for name, parameter in self.named_parameters():
                if not name.startswith("noise_extractor."):
                    parameter.requires_grad_(True)

    def forward(
        self,
        mixture: torch.Tensor,
        noise_reference: torch.Tensor | None = None,
        fusion_mode: Literal["fusion", "mixture", "noise"] = "fusion",
    ) -> Dict[str, torch.Tensor]:
        estimated_noise = self.noise_extractor(mixture)
        estimated_speech = mixture - estimated_noise
        noise_for_features = estimated_noise if noise_reference is None else noise_reference

        mixture_features = self.mixture_encoder(mixture)
        noise_features = self.noise_encoder(noise_for_features)
        if mixture_features.shape[1] != noise_features.shape[1]:
            mixture_features = F.interpolate(
                mixture_features.transpose(1, 2),
                size=noise_features.shape[1],
                mode="linear",
                align_corners=False,
            ).transpose(1, 2)
        mixture_features = F.dropout(
            mixture_features, p=self.mixture_branch_dropout, training=self.training
        )

        if fusion_mode == "fusion":
            classifier_features = self.fusion(mixture_features, noise_features)
        elif fusion_mode == "mixture":
            classifier_features = mixture_features
        elif fusion_mode == "noise":
            classifier_features = noise_features
        else:
            raise ValueError(f"Unknown fusion mode: {fusion_mode}")
        logits = self.classifier(self._pool_for_classification(classifier_features))

        local_mixture = self._pool_frames(mixture_features, self.local_snr_frames)
        local_noise = self._pool_frames(noise_features, self.local_snr_frames)
        correction = self.max_snr_correction_db * torch.tanh(
            self.snr_correction(torch.cat((local_mixture, local_noise), dim=-1)).squeeze(-1)
        )
        physical_snr = physical_local_snr(
            estimated_speech, estimated_noise, self.local_snr_frames
        )
        local_snr = (physical_snr + correction).clamp(-30.0, 30.0)

        return {
            "clipwise_output": logits,
            "estimated_noise": estimated_noise,
            "estimated_speech": estimated_speech,
            "local_snr_output": local_snr,
            "physical_local_snr": physical_snr,
            "mixture_features": mixture_features,
            "noise_features": noise_features,
        }


def build_local_snr_model(
    audio_config: AudioFeaturesConfig,
    model_config: ModelConfig,
) -> BlackFeatherLocalSNR:
    return BlackFeatherLocalSNR(audio_config, model_config)

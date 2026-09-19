"""DPCRN-style complex-mask extractor with a 36-label noise head."""

from __future__ import annotations

import torch
from torch import Tensor, nn
import torch.nn.functional as functional


class DualPathBlock(nn.Module):
    """Model spectral patterns within a frame and temporal patterns per bin."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.intra = nn.LSTM(channels, channels // 2, batch_first=True, bidirectional=True)
        self.intra_projection = nn.Linear(channels, channels)
        self.intra_norm = nn.LayerNorm(channels)
        self.inter = nn.LSTM(channels, channels, batch_first=True)
        self.inter_projection = nn.Linear(channels, channels)
        self.inter_norm = nn.LayerNorm(channels)

    def forward(self, features: Tensor) -> Tensor:
        # [B, C, F, T] -> intra-RNN processes F for every time frame.
        batch, channels, frequencies, frames = features.shape
        intra_input = features.permute(0, 3, 2, 1).reshape(batch * frames, frequencies, channels)
        intra_output, _ = self.intra(intra_input)
        intra_output = self.intra_norm(self.intra_projection(intra_output))
        intra_output = intra_output.reshape(batch, frames, frequencies, channels).permute(0, 3, 2, 1)
        features = features + intra_output

        # Inter-RNN processes T independently for each frequency bin.
        inter_input = features.permute(0, 2, 3, 1).reshape(batch * frequencies, frames, channels)
        inter_output, _ = self.inter(inter_input)
        inter_output = self.inter_norm(self.inter_projection(inter_output))
        inter_output = inter_output.reshape(batch, frequencies, frames, channels).permute(0, 3, 1, 2)
        return features + inter_output


class DPCRNNoiseClassifier(nn.Module):
    """Extract noise directly from a mixture, then classify it."""

    def __init__(
        self,
        classes_num: int,
        n_fft: int = 512,
        hop_length: int = 160,
        encoder_channels: tuple[int, ...] = (32, 64, 96),
        dprnn_blocks: int = 2,
        embedding_dim: int = 192,
    ) -> None:
        super().__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.register_buffer("window", torch.hann_window(n_fft), persistent=False)

        layers: list[nn.Module] = []
        in_channels = 2
        for out_channels in encoder_channels:
            layers.extend((
                nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=(2, 1), padding=1),
                nn.BatchNorm2d(out_channels),
                nn.PReLU(out_channels),
            ))
            in_channels = out_channels
        self.encoder = nn.Sequential(*layers)
        self.dual_path = nn.Sequential(*[DualPathBlock(in_channels) for _ in range(dprnn_blocks)])
        self.mask_head = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),
            nn.PReLU(in_channels),
            nn.Conv2d(in_channels, 2, kernel_size=1),
        )
        frequency_bins = n_fft // 2 + 1
        self.classifier = nn.Sequential(
            nn.Linear(frequency_bins, embedding_dim),
            nn.PReLU(embedding_dim),
            nn.Dropout(0.2),
            nn.Linear(embedding_dim, classes_num),
        )

    def _stft(self, waveform: Tensor) -> Tensor:
        spectrum = torch.stft(
            waveform,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            window=self.window,
            center=True,
            return_complex=True,
        )
        return torch.view_as_real(spectrum).permute(0, 3, 1, 2)

    def forward(self, mixture: Tensor) -> dict[str, Tensor]:
        mixture_parts = self._stft(mixture)
        encoded = self.encoder(mixture_parts)
        encoded = self.dual_path(encoded)
        mask = self.mask_head(encoded)
        mask = functional.interpolate(mask, size=mixture_parts.shape[-2:], mode="bilinear", align_corners=False)

        real = mixture_parts[:, 0] * mask[:, 0] - mixture_parts[:, 1] * mask[:, 1]
        imaginary = mixture_parts[:, 0] * mask[:, 1] + mixture_parts[:, 1] * mask[:, 0]
        noise_spectrum = torch.complex(real, imaginary)
        estimated_noise = torch.istft(
            noise_spectrum,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            window=self.window,
            center=True,
            length=mixture.shape[-1],
        )
        log_magnitude = torch.log1p(noise_spectrum.abs()).mean(dim=-1)
        return {"noise": estimated_noise, "logits": self.classifier(log_magnitude), "mask": mask}

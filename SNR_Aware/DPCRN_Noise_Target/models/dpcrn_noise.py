"""DPCRN-style complex-mask extractor with a 36-label noise head."""

from __future__ import annotations

import torch
from torch import Tensor, nn
import torch.nn.functional as functional


class DualPathBlock(nn.Module):
    """Model spectral patterns within a frame and temporal patterns per bin."""

    def __init__(self, channels: int, bidirectional_time: bool = True) -> None:
        super().__init__()
        self.intra = nn.LSTM(channels, channels // 2, batch_first=True, bidirectional=True)
        self.intra_projection = nn.Linear(channels, channels)
        self.intra_norm = nn.LayerNorm(channels)
        hidden = channels // 2 if bidirectional_time else channels
        self.inter = nn.LSTM(channels, hidden, batch_first=True, bidirectional=bidirectional_time)
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


ENCODER_KERNELS = ((5, 2), (3, 2), (3, 2), (3, 2), (3, 2))
ENCODER_STRIDES = ((2, 1), (2, 1), (1, 1), (1, 1), (1, 1))


def _causal_time_pad(features: Tensor) -> Tensor:
    """Kernels are 2 frames wide; pad the past so the frame count survives."""
    return functional.pad(features, (1, 0))


class DPCRNEncoder(nn.Module):
    """Five Conv2d layers, frequency 257 -> 129 -> 65, time length preserved."""

    def __init__(
        self,
        in_channels: int = 2,
        channels: tuple[int, ...] = (32, 32, 32, 64, 128),
    ) -> None:
        super().__init__()
        self.layers = nn.ModuleList()
        previous = in_channels
        for out_channels, kernel, stride in zip(channels, ENCODER_KERNELS, ENCODER_STRIDES):
            self.layers.append(nn.Sequential(
                nn.Conv2d(previous, out_channels, kernel, stride, padding=(kernel[0] // 2, 0)),
                nn.BatchNorm2d(out_channels),
                nn.PReLU(out_channels),
            ))
            previous = out_channels

    def forward(self, spectrum: Tensor) -> tuple[Tensor, list[Tensor]]:
        skips: list[Tensor] = []
        features = spectrum
        for layer in self.layers:
            features = layer(_causal_time_pad(features))
            skips.append(features)
        return features, skips


class DPCRNDecoder(nn.Module):
    """Mirror of the encoder. Each stage sees the matching encoder output."""

    def __init__(self, channels: tuple[int, ...] = (32, 32, 32, 64, 128)) -> None:
        super().__init__()
        outputs = (*channels[:-1][::-1], 2)          # 64, 32, 32, 32, 2
        inputs = channels[::-1]                       # 128, 64, 32, 32, 32
        kernels = ENCODER_KERNELS[::-1]
        strides = ENCODER_STRIDES[::-1]
        self.layers = nn.ModuleList()
        for index, (in_ch, out_ch, kernel, stride) in enumerate(
            zip(inputs, outputs, kernels, strides)
        ):
            last = index == len(inputs) - 1
            block: list[nn.Module] = [nn.ConvTranspose2d(
                in_ch * 2, out_ch, kernel, stride, padding=(kernel[0] // 2, 0)
            )]
            if not last:
                block += [nn.BatchNorm2d(out_ch), nn.PReLU(out_ch)]
            self.layers.append(nn.Sequential(*block))

    def forward(self, features: Tensor, skips: list[Tensor]) -> Tensor:
        frames = features.shape[-1]
        for layer, skip in zip(self.layers, skips[::-1]):
            features = layer(torch.cat((features, skip), dim=1))
            features = features[..., :frames]         # transposed conv adds one frame
        return features


class AttentionPooling(nn.Module):
    """Weight frames by learned relevance instead of averaging them."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.score = nn.Linear(dim, 1)

    def forward(self, sequence: Tensor) -> Tensor:      # [B, T, D] -> [B, D]
        weights = self.score(sequence).softmax(dim=1)
        return (sequence * weights).sum(dim=1)


class NoiseClassifierHead(nn.Module):
    """Read the dual-path bottleneck and the separated-noise spectrum together."""

    def __init__(
        self,
        bottleneck_channels: int = 128,
        frequency_bins: int = 257,
        embedding_dim: int = 256,
        classes_num: int = 36,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        # Strided convs fold frequency into channels. Averaging it away costs as
        # much accuracy as averaging time away (see spec 2.2).
        self.frequency = nn.Sequential(
            nn.Conv2d(bottleneck_channels, 64, 3, stride=(2, 1), padding=1),
            nn.BatchNorm2d(64), nn.PReLU(64),
            nn.Conv2d(64, 64, 3, stride=(2, 1), padding=1),
            nn.BatchNorm2d(64), nn.PReLU(64),
        )
        self.bottleneck_projection = nn.Conv1d(64 * 17, embedding_dim, kernel_size=1)
        self.noise_projection = nn.Sequential(
            nn.Conv1d(frequency_bins, 128, kernel_size=5, padding=2),
            nn.BatchNorm1d(128), nn.PReLU(128),
        )
        self.recurrent = nn.GRU(
            embedding_dim + 128, embedding_dim // 2,
            batch_first=True, bidirectional=True,
        )
        self.pooling = AttentionPooling(embedding_dim)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(embedding_dim, classes_num)

    def forward(self, bottleneck: Tensor, noise_spectrum_magnitude: Tensor) -> tuple[Tensor, Tensor]:
        batch, _, _, frames = bottleneck.shape
        folded = self.frequency(bottleneck).reshape(batch, -1, frames)
        spectral = self.bottleneck_projection(folded)
        separated = self.noise_projection(torch.log1p(noise_spectrum_magnitude))
        sequence, _ = self.recurrent(torch.cat((spectral, separated), dim=1).transpose(1, 2))
        embedding = self.pooling(sequence)
        return embedding, self.classifier(self.dropout(embedding))


class DPCRNNoiseClassifier(nn.Module):
    """Extract noise directly from a mixture, then classify it."""

    def __init__(
        self,
        classes_num: int,
        n_fft: int = 512,
        hop_length: int = 160,
        encoder_channels: tuple[int, ...] = (32, 32, 32, 64, 128),
        dprnn_blocks: int = 2,
        embedding_dim: int = 256,
        classifier_dropout: float = 0.2,
        bidirectional_time: bool = True,
    ) -> None:
        super().__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.register_buffer("window", torch.hann_window(n_fft), persistent=False)
        self.encoder = DPCRNEncoder(2, encoder_channels)
        self.dual_path = nn.Sequential(*[
            DualPathBlock(encoder_channels[-1], bidirectional_time)
            for _ in range(dprnn_blocks)
        ])
        self.decoder = DPCRNDecoder(encoder_channels)
        self.head = NoiseClassifierHead(
            bottleneck_channels=encoder_channels[-1],
            frequency_bins=n_fft // 2 + 1,
            embedding_dim=embedding_dim,
            classes_num=classes_num,
            dropout=classifier_dropout,
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
        bottleneck, skips = self.encoder(mixture_parts)
        bottleneck = self.dual_path(bottleneck)
        mask = self.decoder(bottleneck, skips)

        real = mixture_parts[:, 0] * mask[:, 0] - mixture_parts[:, 1] * mask[:, 1]
        imaginary = mixture_parts[:, 0] * mask[:, 1] + mixture_parts[:, 1] * mask[:, 0]
        noise_spectrum = torch.complex(real, imaginary)
        estimated_noise = torch.istft(
            noise_spectrum, n_fft=self.n_fft, hop_length=self.hop_length,
            window=self.window, center=True, length=mixture.shape[-1],
        )
        embedding, logits = self.head(bottleneck, noise_spectrum.abs())
        return {
            "noise": estimated_noise, "logits": logits,
            "mask": mask, "embedding": embedding,
        }

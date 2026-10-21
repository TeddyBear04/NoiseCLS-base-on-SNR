from typing import NamedTuple

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .config import FRCRNConfig
from .stft import ConviSTFT, ConvSTFT
from .unet import UNET_BINS, UNet


def fit_length(waveform: Tensor, length: int) -> Tensor:
    if waveform.shape[-1] < length:
        return F.pad(waveform, (0, length - waveform.shape[-1]))
    return waveform[..., :length]


class FRCRNOutput(NamedTuple):
    estimate: Tensor
    mask: Tensor


class FRCRN(nn.Module):
    def __init__(self, config: FRCRNConfig = FRCRNConfig()) -> None:
        super().__init__()
        if config.num_bins != UNET_BINS:
            raise ValueError(f"FRCRN UNet needs {UNET_BINS} frequency bins (fft_len=640), got fft_len={config.fft_len}")
        self.num_bins = config.num_bins
        self.stft = ConvSTFT(config)
        self.istft = ConviSTFT(config)
        self.unet = UNet()
        self.unet2 = UNet()

    def forward(self, waveform: Tensor) -> FRCRNOutput:
        spectrum = self.stft(waveform)
        real, imag = spectrum[:, : self.num_bins], spectrum[:, self.num_bins :]
        first = self.unet(torch.stack((real, imag), dim=-1).unsqueeze(1))
        mask = (torch.tanh(first) + torch.tanh(self.unet2(first))).squeeze(1)
        mask_re, mask_im = mask[..., 0], mask[..., 1]
        estimate = torch.cat([real * mask_re - imag * mask_im, real * mask_im + imag * mask_re], dim=1)
        masked = fit_length(self.istft(estimate), waveform.shape[-1])
        return FRCRNOutput(masked, torch.cat([mask_re, mask_im], dim=1))

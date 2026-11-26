import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .config import FRCRNConfig

_OVERLAP_EPS = 1e-8


def fourier_kernels(config: FRCRNConfig, inverse: bool) -> tuple[Tensor, Tensor]:
    window = torch.hann_window(config.win_len, periodic=True, dtype=torch.float64).sqrt()
    basis = torch.fft.rfft(torch.eye(config.fft_len, dtype=torch.float64))[: config.win_len]
    kernel = torch.cat([basis.real, basis.imag], dim=1).T
    if inverse:
        kernel = torch.linalg.pinv(kernel).T
    kernel = (kernel * window).unsqueeze(1)
    return kernel.float(), window.view(1, -1, 1).float()


class ConvSTFT(nn.Module):
    def __init__(self, config: FRCRNConfig) -> None:
        super().__init__()
        self.stride = config.win_inc
        kernel, _ = fourier_kernels(config, inverse=False)
        self.register_buffer("weight", kernel)

    def forward(self, waveform: Tensor) -> Tensor:
        return F.conv1d(waveform.unsqueeze(1), self.weight, stride=self.stride)


class ConviSTFT(nn.Module):
    def __init__(self, config: FRCRNConfig) -> None:
        super().__init__()
        self.stride = config.win_inc
        kernel, window = fourier_kernels(config, inverse=True)
        self.register_buffer("weight", kernel)
        self.register_buffer("window", window)
        self.register_buffer("enframe", torch.eye(config.win_len).unsqueeze(1))

    def forward(self, spectrum: Tensor) -> Tensor:
        frames = F.conv_transpose1d(spectrum, self.weight, stride=self.stride)
        squared_window = self.window.repeat(1, 1, spectrum.shape[-1]).square()
        overlap = F.conv_transpose1d(squared_window, self.enframe, stride=self.stride)
        return (frames / (overlap + _OVERLAP_EPS)).squeeze(1)

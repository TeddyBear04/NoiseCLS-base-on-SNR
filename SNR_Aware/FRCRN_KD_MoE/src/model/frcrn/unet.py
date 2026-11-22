import torch
from torch import Tensor, nn

from .complex_nn import (
    ComplexBatchNorm2d,
    ComplexConv2d,
    ComplexConvTranspose2d,
    ComplexSqueezeExcitation,
    ComplexUniDeepFsmn,
    ComplexUniDeepFsmn_L1,
)

UNET_BINS = 321
_CHANNELS = 128
_ENC_CHANNELS = (1, 128, 128, 128, 128, 128, 128, 128)
_ENC_KERNELS = ((5, 2),) * 6 + ((2, 2),)
_DEC_CHANNELS = (64, 128, 128, 128, 128, 128, 128, 1)
_DEC_KERNELS = ((2, 2), (5, 2), (5, 2), (5, 2), (6, 2), (5, 2), (5, 2))
_STRIDE = (2, 1)
_PADDING = (0, 1)
_SE_REDUCTION = 8


class Encoder(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: tuple[int, int]) -> None:
        super().__init__()
        self.conv = ComplexConv2d(in_channels, out_channels, kernel_size, _STRIDE, _PADDING)
        self.bn = ComplexBatchNorm2d(out_channels)
        self.relu = nn.LeakyReLU(inplace=True)

    def forward(self, x: Tensor) -> Tensor:
        return self.relu(self.bn(self.conv(x)))


class Decoder(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: tuple[int, int]) -> None:
        super().__init__()
        self.transconv = ComplexConvTranspose2d(in_channels, out_channels, kernel_size, _STRIDE, _PADDING)
        self.bn = ComplexBatchNorm2d(out_channels)
        self.relu = nn.LeakyReLU(inplace=True)

    def forward(self, x: Tensor) -> Tensor:
        return self.relu(self.bn(self.transconv(x)))


class UNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        depth = len(_ENC_KERNELS)
        self.depth = depth
        self.fsmn = ComplexUniDeepFsmn(_CHANNELS, _CHANNELS, _CHANNELS)
        self.encoders = nn.ModuleList(
            Encoder(_ENC_CHANNELS[i], _ENC_CHANNELS[i + 1], _ENC_KERNELS[i]) for i in range(depth)
        )
        self.fsmn_enc = nn.ModuleList(ComplexUniDeepFsmn_L1(_CHANNELS, _CHANNELS, _CHANNELS) for _ in range(depth))
        self.se_layers_enc = nn.ModuleList(
            ComplexSqueezeExcitation(_ENC_CHANNELS[i + 1], _SE_REDUCTION) for i in range(depth)
        )
        self.decoders = nn.ModuleList(
            Decoder(_DEC_CHANNELS[i] * 2, _DEC_CHANNELS[i + 1], _DEC_KERNELS[i]) for i in range(depth)
        )
        self.fsmn_dec = nn.ModuleList(ComplexUniDeepFsmn_L1(_CHANNELS, _CHANNELS, _CHANNELS) for _ in range(depth))
        self.se_layers_dec = nn.ModuleList(
            ComplexSqueezeExcitation(_DEC_CHANNELS[i + 1], _SE_REDUCTION) for i in range(depth - 1)
        )
        self.linear = ComplexConv2d(_DEC_CHANNELS[-1], 1, 1)

    def forward(self, x: Tensor) -> Tensor:
        skips = [x]
        for i, encoder in enumerate(self.encoders):
            if i > 0:
                x = self.fsmn_enc[i](x)
            x = encoder(x)
            skips.append(self.se_layers_enc[i](x))
        x = self.fsmn(x)
        for i, decoder in enumerate(self.decoders):
            x = decoder(x)
            if i == self.depth - 1:
                break
            x = self.fsmn_dec[i](x)
            if i < self.depth - 2:
                x = self.se_layers_dec[i](x)
            x = torch.cat([x, skips[self.depth - 1 - i]], dim=1)
        return self.linear(x)

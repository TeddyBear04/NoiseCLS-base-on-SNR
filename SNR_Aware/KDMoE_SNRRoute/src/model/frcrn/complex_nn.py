from collections.abc import Callable

import torch
from torch import Tensor, nn
from torch.nn import functional as F

_FSMN_ORDER = 20


def _complex(real_fn: Callable[[Tensor], Tensor], imag_fn: Callable[[Tensor], Tensor], x: Tensor) -> Tensor:
    real = real_fn(x[..., 0]) - imag_fn(x[..., 1])
    imaginary = real_fn(x[..., 1]) + imag_fn(x[..., 0])
    return torch.stack((real, imaginary), dim=-1)


class UniDeepFsmn(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, lorder: int, hidden_size: int) -> None:
        super().__init__()
        self.lorder = lorder
        self.linear = nn.Linear(input_dim, hidden_size)
        self.project = nn.Linear(hidden_size, output_dim, bias=False)
        self.conv1 = nn.Conv2d(output_dim, output_dim, (lorder, 1), groups=output_dim, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        projected = self.project(F.relu(self.linear(x)))
        per_channel = projected.transpose(1, 2).unsqueeze(-1)
        memory = self.conv1(F.pad(per_channel, (0, 0, self.lorder - 1, 0)))
        return x + (per_channel + memory).squeeze(-1).transpose(1, 2)


class ComplexUniDeepFsmn(nn.Module):
    def __init__(self, num_in: int, num_hidden: int = 128, num_out: int = 128) -> None:
        super().__init__()
        self.fsmn_re_L1 = UniDeepFsmn(num_in, num_hidden, _FSMN_ORDER, num_hidden)
        self.fsmn_im_L1 = UniDeepFsmn(num_in, num_hidden, _FSMN_ORDER, num_hidden)
        self.fsmn_re_L2 = UniDeepFsmn(num_hidden, num_out, _FSMN_ORDER, num_hidden)
        self.fsmn_im_L2 = UniDeepFsmn(num_hidden, num_out, _FSMN_ORDER, num_hidden)

    def forward(self, x: Tensor) -> Tensor:
        b, c, h, t, d = x.shape
        sequence = x.reshape(b, c * h, t, d).transpose(1, 2)
        hidden = _complex(self.fsmn_re_L1, self.fsmn_im_L1, sequence)
        output = _complex(self.fsmn_re_L2, self.fsmn_im_L2, hidden)
        return output.transpose(1, 2).reshape(b, c, h, t, d)


class ComplexUniDeepFsmn_L1(nn.Module):
    def __init__(self, num_in: int, num_hidden: int = 128, num_out: int = 128) -> None:
        super().__init__()
        self.fsmn_re_L1 = UniDeepFsmn(num_in, num_hidden, _FSMN_ORDER, num_hidden)
        self.fsmn_im_L1 = UniDeepFsmn(num_in, num_hidden, _FSMN_ORDER, num_hidden)

    def forward(self, x: Tensor) -> Tensor:
        b, c, h, t, d = x.shape
        sequence = x.transpose(1, 3).reshape(b * t, h, c, d)
        output = _complex(self.fsmn_re_L1, self.fsmn_im_L1, sequence)
        return output.reshape(b, t, h, c, d).transpose(1, 3)


class ComplexConv2d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size, stride=1, padding=0, **kwargs) -> None:
        super().__init__()
        self.conv_re = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, **kwargs)
        self.conv_im = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, **kwargs)

    def forward(self, x: Tensor) -> Tensor:
        return _complex(self.conv_re, self.conv_im, x)


class ComplexConvTranspose2d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size, stride=1, padding=0) -> None:
        super().__init__()
        self.tconv_re = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding)
        self.tconv_im = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding)

    def forward(self, x: Tensor) -> Tensor:
        return _complex(self.tconv_re, self.tconv_im, x)


class ComplexBatchNorm2d(nn.Module):
    def __init__(self, num_features: int) -> None:
        super().__init__()
        self.bn_re = nn.BatchNorm2d(num_features)
        self.bn_im = nn.BatchNorm2d(num_features)

    def forward(self, x: Tensor) -> Tensor:
        return torch.stack((self.bn_re(x[..., 0]), self.bn_im(x[..., 1])), dim=-1)


def _excitation(channels: int, reduction: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(channels, channels // reduction),
        nn.ReLU(inplace=True),
        nn.Linear(channels // reduction, channels),
        nn.Sigmoid(),
    )


class ComplexSqueezeExcitation(nn.Module):
    def __init__(self, channels: int, reduction: int = 8) -> None:
        super().__init__()
        self.fc_r = _excitation(channels, reduction)
        self.fc_i = _excitation(channels, reduction)

    def forward(self, x: Tensor) -> Tensor:
        b, c = x.shape[:2]
        pooled_re = x[..., 0].mean(dim=(-2, -1))
        pooled_im = x[..., 1].mean(dim=(-2, -1))
        gate_re = self.fc_r(pooled_re) - self.fc_i(pooled_im)
        gate_im = self.fc_r(pooled_im) + self.fc_i(pooled_re)
        return x * torch.stack((gate_re, gate_im), dim=-1).view(b, c, 1, 1, 2)

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.base_backbone import BaseBackbone


def init_layer(layer: nn.Module) -> None:
    nn.init.xavier_uniform_(layer.weight)
    if hasattr(layer, "bias") and layer.bias is not None:
        layer.bias.data.fill_(0.0)


def init_bn(bn: nn.BatchNorm2d) -> None:
    if bn.bias is not None:
        bn.bias.data.fill_(0.0)
    if bn.weight is not None:
        bn.weight.data.fill_(1.0)


class InvertedResidual(nn.Module):
    """
    PANNs-style MobileNetV2 inverted residual block.

    Downsampling follows the original PANNs implementation: the depthwise
    convolution uses stride 1 and AvgPool2d performs spatial reduction.
    """
    def __init__(self, inp: int, oup: int, stride: int, expand_ratio: int) -> None:
        super(InvertedResidual, self).__init__()
        if stride not in [1, 2]:
            raise ValueError(f"MobileNetV2 stride must be 1 or 2, got {stride}.")

        hidden_dim = round(inp * expand_ratio)
        self.use_res_connect = stride == 1 and inp == oup

        if expand_ratio == 1:
            layers = [
                nn.Conv2d(hidden_dim, hidden_dim, 3, 1, 1, groups=hidden_dim, bias=False),
                nn.AvgPool2d(stride),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            ]
            init_layer(layers[0])
            init_bn(layers[2])
            init_layer(layers[4])
            init_bn(layers[5])
        else:
            layers = [
                nn.Conv2d(inp, hidden_dim, 1, 1, 0, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
                nn.Conv2d(hidden_dim, hidden_dim, 3, 1, 1, groups=hidden_dim, bias=False),
                nn.AvgPool2d(stride),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            ]
            init_layer(layers[0])
            init_bn(layers[1])
            init_layer(layers[3])
            init_bn(layers[5])
            init_layer(layers[7])
            init_bn(layers[8])

        self.conv = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_res_connect:
            return x + self.conv(x)
        return self.conv(x)


class MobileNetV2(BaseBackbone):
    """
    MobileNetV2 audio backbone adapted from PANNs/U-FFIA.

    This class intentionally excludes Spectrogram, Logmel, SpecAugment, and
    bn0 because those steps are handled by AudioFrontend in the current codebase.
    """
    def __init__(self, classes_num: int = 4, width_mult: float = 1.0) -> None:
        super(MobileNetV2, self).__init__()
        self.model_name = "mobilenet_v2"

        block = InvertedResidual
        input_channel = 32
        last_channel = 1280
        inverted_residual_setting = [
            # expand_ratio, channels, num_blocks, stride
            [1, 16, 1, 1],
            [6, 24, 2, 2],
            [6, 32, 3, 2],
            [6, 64, 4, 2],
            [6, 96, 3, 2],
            [6, 160, 3, 1],
            [6, 320, 1, 1],
        ]

        def conv_bn(inp: int, oup: int, stride: int) -> nn.Sequential:
            layers = [
                nn.Conv2d(inp, oup, 3, 1, 1, bias=False),
                nn.AvgPool2d(stride),
                nn.BatchNorm2d(oup),
                nn.ReLU6(inplace=True),
            ]
            init_layer(layers[0])
            init_bn(layers[2])
            return nn.Sequential(*layers)

        def conv_1x1_bn(inp: int, oup: int) -> nn.Sequential:
            layers = [
                nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
                nn.ReLU6(inplace=True),
            ]
            init_layer(layers[0])
            init_bn(layers[1])
            return nn.Sequential(*layers)

        input_channel = int(input_channel * width_mult)
        self.last_channel = int(last_channel * width_mult) if width_mult > 1.0 else last_channel

        features = [conv_bn(1, input_channel, 2)]
        for expand_ratio, channels, num_blocks, stride in inverted_residual_setting:
            output_channel = int(channels * width_mult)
            for block_index in range(num_blocks):
                block_stride = stride if block_index == 0 else 1
                features.append(block(input_channel, output_channel, block_stride, expand_ratio=expand_ratio))
                input_channel = output_channel
        features.append(conv_1x1_bn(input_channel, self.last_channel))
        self.features = nn.Sequential(*features)

        self.fc1 = nn.Linear(self.last_channel, 1024, bias=True)
        self.fc_audioset = nn.Linear(1024, classes_num, bias=True)

        self.init_weight()

    def init_weight(self) -> None:
        init_layer(self.fc1)
        init_layer(self.fc_audioset)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = torch.mean(x, dim=3)

        x1, _ = torch.max(x, dim=2)
        x2 = torch.mean(x, dim=2)
        x = x1 + x2

        x = F.relu_(self.fc1(x))
        return self.fc_audioset(x)

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


class Mobilev2Block(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, r: int = 1) -> None:
        super(Mobilev2Block, self).__init__()

        size = 3
        pad = size // 2

        self.conv1a = nn.Conv2d(in_channels, out_channels * r, kernel_size=1, bias=False)
        self.conv1b = nn.Conv2d(
            out_channels * r,
            out_channels * r,
            kernel_size=size,
            padding=pad,
            groups=out_channels * r,
            bias=False,
        )
        self.conv1c = nn.Conv2d(out_channels * r, out_channels, kernel_size=1, bias=False)

        self.conv2a = nn.Conv2d(out_channels, out_channels * r, kernel_size=1, bias=False)
        self.conv2b = nn.Conv2d(
            out_channels * r,
            out_channels * r,
            kernel_size=size,
            padding=pad,
            groups=out_channels * r,
            bias=False,
        )
        self.conv2c = nn.Conv2d(out_channels * r, out_channels, kernel_size=1, bias=False)

        self.bn1a = nn.BatchNorm2d(in_channels)
        self.bn1b = nn.BatchNorm2d(out_channels * r)
        self.bn1c = nn.BatchNorm2d(out_channels)
        self.bn2a = nn.BatchNorm2d(out_channels)
        self.bn2b = nn.BatchNorm2d(out_channels * r)
        self.bn2c = nn.BatchNorm2d(out_channels)
        self.gelu = nn.GELU()

        if in_channels != out_channels:
            self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1)
            self.is_shortcut = True
        else:
            self.is_shortcut = False

        self.init_weights()

    def init_weights(self) -> None:
        init_layer(self.conv1a)
        init_layer(self.conv1b)
        init_layer(self.conv1c)
        init_layer(self.conv2a)
        init_layer(self.conv2b)
        init_layer(self.conv2c)
        init_bn(self.bn1a)
        init_bn(self.bn1b)
        init_bn(self.bn1c)
        init_bn(self.bn2a)
        init_bn(self.bn2b)
        init_bn(self.bn2c)
        if self.is_shortcut:
            init_layer(self.shortcut)

    def forward(self, input_tensor: torch.Tensor, pool_size: tuple[int, int] = (2, 2)) -> torch.Tensor:
        origin = input_tensor

        x = self.conv1a(self.gelu(self.bn1a(origin)))
        x = self.conv1b(self.gelu(self.bn1b(x)))
        x = self.conv1c(self.gelu(self.bn1c(x)))

        if self.is_shortcut:
            origin = self.shortcut(origin) + x
        else:
            origin = origin + x

        x = self.conv2a(self.gelu(self.bn2a(origin)))
        x = self.conv2b(self.gelu(self.bn2b(x)))
        x = self.conv2c(self.gelu(self.bn2c(x)))

        x = origin + x
        return F.avg_pool2d(x, kernel_size=pool_size, stride=pool_size)


class Cnn14MobileV2_1P9M(BaseBackbone):
    """
    Cnn14-MobileV2 variant adapted from U-FFIA conv_mv2/CBAM_mobilenet.
    Approximate parameter count: 1.96M for classes_num=4.
    """
    def __init__(self, classes_num: int = 4) -> None:
        super(Cnn14MobileV2_1P9M, self).__init__()
        self.model_name = "cnn14_mobilev2_1p9m"

        self.conv_block1 = Mobilev2Block(in_channels=1, out_channels=16)
        self.conv_block2 = Mobilev2Block(in_channels=16, out_channels=32)
        self.conv_block3 = Mobilev2Block(in_channels=32, out_channels=64)
        self.conv_block4 = Mobilev2Block(in_channels=64, out_channels=128)
        self.conv_block5 = Mobilev2Block(in_channels=128, out_channels=256)
        self.conv_block6 = Mobilev2Block(in_channels=256, out_channels=512)

        self.fc1 = nn.Linear(512, 1024, bias=True)
        self.fc_audioset = nn.Linear(1024, classes_num, bias=True)
        self.gelu = nn.GELU()

        self.init_weight()

    def init_weight(self) -> None:
        init_layer(self.fc1)
        init_layer(self.fc_audioset)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv_block1(x, pool_size=(2, 2))
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block2(x, pool_size=(2, 2))
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block3(x, pool_size=(2, 2))
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block4(x, pool_size=(2, 2))
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block5(x, pool_size=(2, 2))
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block6(x, pool_size=(2, 2))
        x = F.dropout(x, p=0.2, training=self.training)

        x = torch.mean(x, dim=3)
        x1, _ = torch.max(x, dim=2)
        x2 = torch.mean(x, dim=2)
        x = x1 + x2

        x = F.dropout(x, p=0.5, training=self.training)
        x = self.gelu(self.fc1(x))
        x = F.dropout(x, p=0.2, training=self.training)
        return self.fc_audioset(x)

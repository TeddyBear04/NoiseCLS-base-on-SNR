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


class MobileNetV1(BaseBackbone):
    """
    PANNs/U-FFIA MobileNetV1 adapted for the current audio pipeline.

    AudioFrontend handles Spectrogram, Logmel, SpecAugment, and bn0. This
    backbone receives [Batch, 1, Time, Mel] features and returns raw logits.
    """
    def __init__(self, classes_num: int = 4) -> None:
        super(MobileNetV1, self).__init__()
        self.model_name = "mobilenet_v1"

        def conv_bn(inp: int, oup: int, stride: int) -> nn.Sequential:
            layers = [
                nn.Conv2d(inp, oup, 3, 1, 1, bias=False),
                nn.AvgPool2d(stride),
                nn.BatchNorm2d(oup),
                nn.ReLU(inplace=True),
            ]
            init_layer(layers[0])
            init_bn(layers[2])
            return nn.Sequential(*layers)

        def conv_dw(inp: int, oup: int, stride: int) -> nn.Sequential:
            layers = [
                nn.Conv2d(inp, inp, 3, 1, 1, groups=inp, bias=False),
                nn.AvgPool2d(stride),
                nn.BatchNorm2d(inp),
                nn.ReLU(inplace=True),
                nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
                nn.ReLU(inplace=True),
            ]
            init_layer(layers[0])
            init_bn(layers[2])
            init_layer(layers[4])
            init_bn(layers[5])
            return nn.Sequential(*layers)

        self.features = nn.Sequential(
            conv_bn(1, 32, 2),
            conv_dw(32, 64, 1),
            conv_dw(64, 128, 2),
            conv_dw(128, 128, 1),
            conv_dw(128, 256, 2),
            conv_dw(256, 256, 1),
            conv_dw(256, 512, 2),
            conv_dw(512, 512, 1),
            conv_dw(512, 512, 1),
            conv_dw(512, 512, 1),
            conv_dw(512, 512, 1),
            conv_dw(512, 512, 1),
            conv_dw(512, 1024, 2),
            conv_dw(1024, 1024, 1),
        )

        self.fc1 = nn.Linear(1024, 1024, bias=True)
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

        x = F.dropout(x, p=0.5, training=self.training)
        x = F.relu_(self.fc1(x))
        return self.fc_audioset(x)

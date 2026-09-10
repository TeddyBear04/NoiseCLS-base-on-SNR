import torch.nn as nn

from models.torchvision_adapter import TorchvisionBackbone, adapt_first_conv


class ResNet34(TorchvisionBackbone):
    """Torchvision ResNet-18 (~11.2M parameters) on Mel-spectrograms.

    The mid-size residual baseline: deep enough to separate 36 noise textures,
    an order of magnitude smaller than the PANNs ResNet22 already in the
    registry, and cheap enough to sweep hyper-parameters against.
    """

    def __init__(self, classes_num: int = 4, pretrained: bool = False, dropout: float = 0.2) -> None:
        try:
            from torchvision.models import ResNet34_Weights, resnet34
        except ImportError as exc:
            raise ImportError(
                "torchvision is required for ResNet34. "
                "Install project requirements before using this backbone."
            ) from exc

        model = resnet34(weights=ResNet34_Weights.DEFAULT if pretrained else None)
        model.conv1 = adapt_first_conv(model.conv1, pretrained=pretrained)

        # Everything up to (but not including) the global average pool: the
        # base class does its own audio-specific pooling instead.
        features = nn.Sequential(
            model.conv1,
            model.bn1,
            model.relu,
            model.maxpool,
            model.layer1,
            model.layer2,
            model.layer3,
            model.layer4,
        )
        super().__init__("resnet34", features, model.fc.in_features, classes_num, dropout)

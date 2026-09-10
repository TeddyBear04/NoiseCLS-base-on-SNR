import torch.nn as nn

from models.torchvision_adapter import TorchvisionBackbone, adapt_first_conv


class DenseNet121(TorchvisionBackbone):
    """Torchvision DenseNet-121 (~7.0M parameters) on Mel-spectrograms.

    Dense connectivity reuses every earlier feature map, so the classifier
    still sees fine-grained early-layer detail - the narrowband tones and
    clicks that separate several of the 36 noise classes - at a parameter
    count below ResNet18.
    """

    def __init__(self, classes_num: int = 4, pretrained: bool = False, dropout: float = 0.2) -> None:
        try:
            from torchvision.models import DenseNet121_Weights, densenet121
        except ImportError as exc:
            raise ImportError(
                "torchvision is required for DenseNet121. "
                "Install project requirements before using this backbone."
            ) from exc

        model = densenet121(weights=DenseNet121_Weights.DEFAULT if pretrained else None)
        model.features.conv0 = adapt_first_conv(model.features.conv0, pretrained=pretrained)

        # torchvision applies the final ReLU in DenseNet.forward rather than
        # inside .features, so it has to be re-attached here.
        features = nn.Sequential(model.features, nn.ReLU(inplace=True))
        super().__init__("densenet121", features, model.classifier.in_features, classes_num, dropout)

from models.torchvision_adapter import TorchvisionBackbone, adapt_first_conv


class EfficientNetB2(TorchvisionBackbone):
    """Torchvision EfficientNet-B2 (~7.8M parameters) on Mel-spectrograms.

    One compound-scaling step up from the EfficientNetB0 already registered:
    wider and deeper at similar cost per parameter, so the two together show
    whether this task is capacity-bound before reaching for a large model.
    """

    def __init__(self, classes_num: int = 4, pretrained: bool = False, dropout: float = 0.3) -> None:
        try:
            from torchvision.models import EfficientNet_B2_Weights, efficientnet_b2
        except ImportError as exc:
            raise ImportError(
                "torchvision is required for EfficientNetB2. "
                "Install project requirements before using this backbone."
            ) from exc

        model = efficientnet_b2(weights=EfficientNet_B2_Weights.DEFAULT if pretrained else None)
        model.features[0][0] = adapt_first_conv(model.features[0][0], pretrained=pretrained)

        # The last stage's BatchNorm carries the output width; reading it keeps
        # the head correct if torchvision retunes the architecture.
        feature_dim = model.features[-1][1].num_features
        super().__init__("efficientnet_b2", model.features, feature_dim, classes_num, dropout)

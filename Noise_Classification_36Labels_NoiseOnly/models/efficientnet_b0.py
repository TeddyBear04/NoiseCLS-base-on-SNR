from models.torchvision_adapter import TorchvisionBackbone, adapt_first_conv


class EfficientNetB0(TorchvisionBackbone):
    """Torchvision EfficientNet-B0 (~4.0M parameters) on Mel-spectrograms.

    AudioFrontend handles Spectrogram, Logmel, SpecAugment, and bn0. This
    backbone receives [Batch, 1, Time, Mel] features and returns raw logits.
    """

    def __init__(self, classes_num: int = 4, pretrained: bool = False, dropout: float = 0.2) -> None:
        try:
            from torchvision.models import EfficientNet_B0_Weights, efficientnet_b0
        except ImportError as exc:
            raise ImportError(
                "torchvision is required for EfficientNetB0. "
                "Install project requirements before using this backbone."
            ) from exc

        model = efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT if pretrained else None)
        model.features[0][0] = adapt_first_conv(model.features[0][0], pretrained=pretrained)

        feature_dim = model.features[-1][1].num_features
        super().__init__("efficientnet_b0", model.features, feature_dim, classes_num, dropout)

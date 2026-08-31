import torch
import torch.nn as nn
import torch.nn.functional as F

from models.base_backbone import BaseBackbone


class EfficientNetB0(BaseBackbone):
    """
    Torchvision EfficientNet-B0 adapted for audio Mel-spectrogram classification.

    AudioFrontend handles Spectrogram, Logmel, SpecAugment, and bn0. This
    backbone receives [Batch, 1, Time, Mel] features and returns raw logits.
    """
    def __init__(
        self,
        classes_num: int = 4,
        pretrained: bool = False,
        dropout: float = 0.2,
    ) -> None:
        super(EfficientNetB0, self).__init__()
        self.model_name = "efficientnet_b0"
        self.feature_channels = 1280

        try:
            from torchvision.models import EfficientNet_B0_Weights, efficientnet_b0
        except ImportError as exc:
            raise ImportError(
                "torchvision is required for EfficientNetB0. "
                "Install project requirements before using this backbone."
            ) from exc

        weights = EfficientNet_B0_Weights.DEFAULT if pretrained else None
        model = efficientnet_b0(weights=weights)

        self.features = model.features
        self._replace_first_conv(pretrained=pretrained)
        self._preserve_final_time_resolution()
        self.dropout = nn.Dropout(p=dropout)
        self.fc_audioset = nn.Linear(1280, classes_num, bias=True)

        nn.init.xavier_uniform_(self.fc_audioset.weight)
        if self.fc_audioset.bias is not None:
            self.fc_audioset.bias.data.fill_(0.0)

    def _replace_first_conv(self, pretrained: bool) -> None:
        first_conv = self.features[0][0]
        if not isinstance(first_conv, nn.Conv2d):
            raise TypeError(
                "Unexpected torchvision EfficientNet-B0 stem layout: "
                "features[0][0] is not Conv2d."
            )

        new_conv = nn.Conv2d(
            in_channels=1,
            out_channels=first_conv.out_channels,
            kernel_size=first_conv.kernel_size,
            stride=first_conv.stride,
            padding=first_conv.padding,
            dilation=first_conv.dilation,
            groups=first_conv.groups,
            bias=first_conv.bias is not None,
            padding_mode=first_conv.padding_mode,
        )

        if pretrained:
            with torch.no_grad():
                new_conv.weight.copy_(first_conv.weight.mean(dim=1, keepdim=True))
                if first_conv.bias is not None and new_conv.bias is not None:
                    new_conv.bias.copy_(first_conv.bias)
        else:
            nn.init.kaiming_normal_(new_conv.weight, mode="fan_out", nonlinearity="relu")
            if new_conv.bias is not None:
                new_conv.bias.data.fill_(0.0)

        self.features[0][0] = new_conv

    def _preserve_final_time_resolution(self) -> None:
        """Change the final 2-D downsampling conv to frequency-only stride."""
        stride_convs = [
            module
            for module in self.features.modules()
            if isinstance(module, nn.Conv2d) and tuple(module.stride) == (2, 2)
        ]
        if not stride_convs:
            raise RuntimeError("EfficientNet-B0 has no stride-2 convolution to adapt")
        stride_convs[-1].stride = (1, 2)

    def forward_feature_map(self, x: torch.Tensor) -> torch.Tensor:
        return self.features(x)

    def classify_temporal(self, temporal: torch.Tensor) -> torch.Tensor:
        x = self.pool_temporal(temporal)
        x = self.dropout(x)
        return self.fc_audioset(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feature_map = self.forward_feature_map(x)
        return self.classify_temporal(self.temporal_features(feature_map))

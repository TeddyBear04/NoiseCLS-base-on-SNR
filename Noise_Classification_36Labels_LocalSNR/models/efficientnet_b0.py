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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)

        # PANNs-style temporal aggregation: collapse frequency, then combine
        # max and mean statistics across time.
        x = torch.mean(x, dim=3)
        x1, _ = torch.max(x, dim=2)
        x2 = torch.mean(x, dim=2)
        x = x1 + x2

        x = self.dropout(x)
        return self.fc_audioset(x)

"""Shared plumbing for torchvision image backbones reused on Mel-spectrograms.

Every torchvision classifier arrives expecting three-channel RGB images and
ends in a global-average-pool head. AudioFrontend hands out one-channel
[Batch, 1, Time, Mel] features instead, and PANNs-style audio tagging pools
the two axes differently, so each backbone needs the same two adjustments.
They live here rather than being copied into every backbone file.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from models.base_backbone import BaseBackbone


def adapt_first_conv(first_conv: nn.Conv2d, pretrained: bool) -> nn.Conv2d:
    """Rebuild a stem convolution to take one channel instead of three.

    Averaging the RGB weights keeps the pretrained filter's response to a grey
    image, which is what a single-channel spectrogram looks like to the stem.
    Without pretrained weights there is nothing to preserve, so the new kernel
    is initialised from scratch.
    """
    if not isinstance(first_conv, nn.Conv2d):
        raise TypeError(
            f"Expected the backbone stem to be a Conv2d, got {type(first_conv).__name__}. "
            "The torchvision layout probably changed."
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

    return new_conv


def panns_pool(feature_map: torch.Tensor) -> torch.Tensor:
    """Collapse a [Batch, Channels, Time, Mel] map down to [Batch, Channels].

    Frequency is averaged away first: a noise class is defined by which bands
    are active, not by where the activation sits inside a band. Time is then
    reduced by max and mean together, so a short transient and a sustained
    texture both leave a mark - plain average pooling washes the transient out.
    """
    feature_map = torch.mean(feature_map, dim=3)
    strongest, _ = torch.max(feature_map, dim=2)
    average = torch.mean(feature_map, dim=2)
    return strongest + average


class TorchvisionBackbone(BaseBackbone):
    """Base class for torchvision feature extractors used as audio backbones.

    Subclasses only have to build the feature extractor with a one-channel stem
    and declare how wide its output is; pooling, dropout and the classifier
    head are the same for all of them.
    """

    def __init__(
        self,
        model_name: str,
        features: nn.Module,
        feature_dim: int,
        classes_num: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.model_name = model_name
        self.features = features
        self.dropout = nn.Dropout(p=dropout)
        self.fc_audioset = nn.Linear(feature_dim, classes_num, bias=True)

        nn.init.xavier_uniform_(self.fc_audioset.weight)
        if self.fc_audioset.bias is not None:
            self.fc_audioset.bias.data.fill_(0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = panns_pool(x)
        x = self.dropout(x)
        return self.fc_audioset(x)

from __future__ import annotations

import logging
from typing import Dict, Type

from config import ModelConfig

from .base_backbone import BaseBackbone
from .cnn14_mobilev2 import Cnn14MobileV2, Cnn14MobileV2LocalSNR
from .cnn14_mobilev2_1p9m import Cnn14MobileV2_1P9M
from .efficientnet_b0 import EfficientNetB0
from .mobilenet_v1 import MobileNetV1
from .mobilenet_v2 import MobileNetV2
from .panns_cnn10 import PANNS_Cnn10
from .panns_cnn14 import PANNS_Cnn14
from .panns_cnn6 import PANNS_Cnn6
from .resnet22 import ResNet22

logger = logging.getLogger(__name__)

MODEL_REGISTRY: Dict[str, Type[BaseBackbone]] = {
    "Cnn14MobileV2": Cnn14MobileV2,
    "Cnn14MobileV2LocalSNR": Cnn14MobileV2LocalSNR,
    "Cnn14MobileV2_1P9M": Cnn14MobileV2_1P9M,
    "MobileNetV1": MobileNetV1,
    "MobileNetV2": MobileNetV2,
    "ResNet22": ResNet22,
    "EfficientNetB0": EfficientNetB0,
    "PANNS_Cnn6": PANNS_Cnn6,
    "PANNS_Cnn10": PANNS_Cnn10,
    "PANNS_Cnn14": PANNS_Cnn14,
}

PRETRAINED_BACKBONES = {"EfficientNetB0"}


def build_backbone(model_config: ModelConfig) -> BaseBackbone:
    name = model_config.backbone
    if name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown backbone {name!r}. Available: {sorted(MODEL_REGISTRY)}")
    kwargs = {"classes_num": model_config.classes_num}
    if name in PRETRAINED_BACKBONES:
        kwargs["pretrained"] = model_config.pretrained
    elif model_config.pretrained:
        logger.warning("Backbone %s has no pretrained-weight implementation; ignoring pretrained=true", name)
    return MODEL_REGISTRY[name](**kwargs)

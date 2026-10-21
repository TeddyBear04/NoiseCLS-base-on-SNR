from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, ClassVar, NamedTuple

import yaml
from torch import Tensor

from src.core import BaseConfig, BaseModel

from .backbone import BEATsBackbone
from .classifier import ClassifierHead
from .frontend import Fbank

_DUPLICATE_RELATIVE_BIAS = re.compile(r"encoder\.layers\.(?!0\.)\d+\.self_attn\.relative_attention_bias\.")


@dataclass(frozen=True)
class BEATConfig(BaseConfig):
    input_patch_size: int = 16
    embed_dim: int = 512
    encoder_embed_dim: int = 768
    encoder_layers: int = 12
    encoder_attention_heads: int = 12
    encoder_ffn_embed_dim: int = 3072
    deep_norm: bool = True
    conv_pos: int = 128
    conv_pos_groups: int = 16
    relative_position_embedding: bool = True
    num_buckets: int = 320
    max_distance: int = 800
    gru_rel_pos: bool = True
    dropout: float = 0.0
    attention_dropout: float = 0.0
    activation_dropout: float = 0.0
    dropout_input: float = 0.0
    predictor_dropout: float = 0.1
    num_classes: int = 10


class BEATOutput(NamedTuple):
    logits: Tensor
    features: Tensor


class BEATClassifier(BaseModel):
    expected_missing_prefixes: ClassVar[tuple[str, ...]] = ("head.",)

    def __init__(self, config: BEATConfig, frontend_config: dict[str, Any] | None = None) -> None:
        super().__init__()
        self.config = config
        self.frontend = Fbank(**(frontend_config or {}))
        self.backbone = BEATsBackbone(config)
        self.head = ClassifierHead(config.encoder_embed_dim, config.num_classes, config.predictor_dropout)

    @classmethod
    def from_yaml(cls, path: str | Path) -> BEATClassifier:
        with open(path, encoding="utf-8") as f:
            raw = yaml.safe_load(f)
        return cls(BEATConfig.from_dict(raw[BEATConfig.section]), raw.get("frontend"))

    def forward(self, x: Tensor, is_waveform: bool = True) -> BEATOutput:
        fbank = self.frontend(x) if is_waveform else x
        features = self.backbone(fbank)
        return BEATOutput(logits=self.head(features), features=features)

    def convert_checkpoint(self, checkpoint: dict[str, Any]) -> dict[str, Tensor]:
        state = {}
        for key, value in checkpoint["model"].items():
            if key.startswith("predictor") or _DUPLICATE_RELATIVE_BIAS.search(key):
                continue
            key = key.replace("patch_embedding.", "patch_embed.patch_embedding.")
            key = key.replace(".fc1.", ".ffn.fc1.").replace(".fc2.", ".ffn.fc2.")
            state[f"backbone.{key}"] = value
        return state

    def predict_proba(self, x: Tensor, is_waveform: bool = True) -> Tensor:
        return self(x, is_waveform).logits.softmax(dim=-1)

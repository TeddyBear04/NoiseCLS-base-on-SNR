from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any, ClassVar

import torch
from torch import Tensor, nn


@dataclass(frozen=True)
class LoadReport:
    missing: tuple[str, ...] = ()
    unexpected: tuple[str, ...] = ()

    @property
    def is_clean(self) -> bool:
        return not self.missing and not self.unexpected


class BaseModel(nn.Module, ABC):
    expected_missing_prefixes: ClassVar[tuple[str, ...]] = ()
    strict_pretrained: ClassVar[bool] = True
    weights_only: ClassVar[bool] = False

    @classmethod
    @abstractmethod
    def from_yaml(cls, path: str | Path) -> BaseModel: ...

    @abstractmethod
    def convert_checkpoint(self, checkpoint: dict[str, Any]) -> dict[str, Tensor]: ...

    def read_checkpoint(self, path: str | Path) -> dict[str, Any]:
        return torch.load(path, map_location="cpu", weights_only=self.weights_only)

    def load_pretrained(self, path: str | Path) -> LoadReport:
        checkpoint = self.read_checkpoint(path)
        result = self.load_state_dict(self.convert_checkpoint(checkpoint), strict=False)
        missing = tuple(key for key in result.missing_keys if not key.startswith(self.expected_missing_prefixes))
        report = LoadReport(missing=missing, unexpected=tuple(result.unexpected_keys))
        if self.strict_pretrained and not report.is_clean:
            raise ValueError(f"{path}: missing {list(report.missing[:5])}, unexpected {list(report.unexpected[:5])}")
        return report

    def freeze(self) -> BaseModel:
        self.eval()
        for parameter in self.parameters():
            parameter.requires_grad_(False)
        return self

    def num_parameters(self, trainable_only: bool = False) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad or not trainable_only)

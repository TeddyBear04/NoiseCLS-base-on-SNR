from __future__ import annotations

from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any, ClassVar, TypeVar

import yaml

ConfigT = TypeVar("ConfigT", bound="BaseConfig")


@dataclass(frozen=True)
class BaseConfig:
    section: ClassVar[str] = "model"

    @classmethod
    def from_dict(cls: type[ConfigT], values: dict[str, Any]) -> ConfigT:
        names = {field.name for field in fields(cls)}
        return cls(
            **{key: tuple(value) if isinstance(value, list) else value for key, value in values.items() if key in names}
        )

    @classmethod
    def from_yaml(cls: type[ConfigT], path: str | Path) -> ConfigT:
        with open(path, encoding="utf-8") as f:
            return cls.from_dict(yaml.safe_load(f)[cls.section])

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

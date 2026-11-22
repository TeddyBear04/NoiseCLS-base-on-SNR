"""Configuration loader for Dual-Expert CED."""

from __future__ import annotations

from pathlib import Path
from typing import Any
import yaml


def load_config(config_dir: str | Path) -> dict[str, Any]:
    """Load configuration dictionary from the config directory or YAML file."""
    path = Path(config_dir)
    if path.is_dir():
        yaml_file = path / "config.yaml"
    else:
        yaml_file = path

    if not yaml_file.is_file():
        raise FileNotFoundError(f"Configuration file not found: {yaml_file}")

    with open(yaml_file, encoding="utf-8") as f:
        config = yaml.safe_load(f)

    return config

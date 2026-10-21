from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch import nn


def save_checkpoint(path: str | Path, model: nn.Module, **metadata: Any) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    state = {key: value.detach().cpu() for key, value in model.state_dict().items()}
    torch.save({"model": state, **metadata}, path)
    return path


def load_checkpoint(path: str | Path, model: nn.Module, device: str | torch.device = "cpu") -> dict[str, Any]:
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model"])
    return {key: value for key, value in checkpoint.items() if key != "model"}

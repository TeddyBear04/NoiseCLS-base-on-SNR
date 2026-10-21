from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager

import torch
from torch import nn


class EMA:
    """Trung bình trượt hàm mũ của trọng số model, dùng khi đánh giá."""

    def __init__(self, model: nn.Module, decay: float = 0.999) -> None:
        if not 0.0 < decay < 1.0:
            raise ValueError(f"decay must be in (0, 1), got {decay}")
        self.decay = decay
        self.shadow = {name: parameter.detach().clone() for name, parameter in model.named_parameters()}

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        for name, parameter in model.named_parameters():
            if name in self.shadow:
                shadow = self.shadow[name]
                if shadow.device != parameter.device:
                    shadow = shadow.to(parameter.device)
                    self.shadow[name] = shadow
                shadow.mul_(self.decay).add_(parameter.detach(), alpha=1.0 - self.decay)

    @torch.no_grad()
    def copy_to(self, model: nn.Module) -> None:
        for name, parameter in model.named_parameters():
            if name in self.shadow:
                parameter.copy_(self.shadow[name])

    @contextmanager
    def swap(self, model: nn.Module) -> Iterator[None]:
        backup = {
            name: parameter.detach().clone() for name, parameter in model.named_parameters() if name in self.shadow
        }
        self.copy_to(model)
        try:
            yield
        finally:
            with torch.no_grad():
                for name, parameter in model.named_parameters():
                    if name in backup:
                        parameter.copy_(backup[name])

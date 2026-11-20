from __future__ import annotations

from abc import ABC, abstractmethod
from typing import ClassVar

from torch import Tensor, nn


class Loss(nn.Module, ABC):
    name: ClassVar[str]


class SupervisedLoss(Loss):
    @abstractmethod
    def forward(self, prediction: Tensor, target: Tensor) -> Tensor: ...


class DistillationLoss(Loss):
    @abstractmethod
    def forward(self, student: Tensor, teacher: Tensor) -> Tensor: ...


class SeparationLoss(Loss):
    @abstractmethod
    def forward(self, mixture: Tensor, target: Tensor, estimate: Tensor, mask: Tensor) -> Tensor: ...

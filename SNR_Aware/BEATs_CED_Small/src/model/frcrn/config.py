from __future__ import annotations

from dataclasses import dataclass

from src.core import BaseConfig

TARGETS = ("speech", "noise")


@dataclass(frozen=True)
class FRCRNConfig(BaseConfig):
    win_len: int = 640
    win_inc: int = 320
    fft_len: int = 640
    target: str = "speech"
    target_level_db: float = -25.0

    def __post_init__(self) -> None:
        if self.target not in TARGETS:
            raise ValueError(f"target must be one of {TARGETS}, got {self.target!r}")

    @property
    def num_bins(self) -> int:
        return self.fft_len // 2 + 1

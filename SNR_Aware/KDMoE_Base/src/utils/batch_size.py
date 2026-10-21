from __future__ import annotations

import gc
from collections.abc import Callable, Sequence

import torch


def largest_fitting_batch(
    step: Callable[[int], None],
    candidates: Sequence[int],
    device: torch.device,
    max_fraction: float = 0.9,
) -> int | None:
    if device.type != "cuda" or not candidates:
        return None
    if not 0.0 < max_fraction <= 1.0:
        raise ValueError(f"max_fraction must be in (0, 1], got {max_fraction}")
    budget = max_fraction * torch.cuda.get_device_properties(device).total_memory
    fitted: int | None = None
    for size in sorted(candidates):
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
        try:
            step(size)
            torch.cuda.synchronize(device)
        except torch.cuda.OutOfMemoryError:
            break
        if torch.cuda.max_memory_allocated(device) > budget:
            break
        fitted = size
    gc.collect()
    torch.cuda.empty_cache()
    return fitted

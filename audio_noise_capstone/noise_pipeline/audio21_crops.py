"""Alternative crop strategies for controlled 21-label experiments."""

from __future__ import annotations

import ast
import random
from typing import Any

import torch
from torch.nn import functional as F

from noise_pipeline.audio21 import Audio21Dataset
from noise_pipeline.data import load_pcm16


class EpochRandomCropAudio21Dataset(Audio21Dataset):
    """Use a new random offset every time a long training sample is loaded."""

    def _crop(self, waveform: torch.Tensor, index: int) -> torch.Tensor:
        if self.crop != "random":
            return super()._crop(waveform, index)
        length = waveform.shape[-1]
        if length < self.num_samples:
            return F.pad(waveform, (0, self.num_samples - length))
        if length == self.num_samples:
            return waveform
        offset = random.randrange(length - self.num_samples + 1)
        return waveform[..., offset : offset + self.num_samples]


class ThreeCropAudio21Dataset(Audio21Dataset):
    """Return deterministic start, center, and end crops for evaluation."""

    def __init__(self, *args, **kwargs) -> None:
        kwargs["crop"] = "center"
        super().__init__(*args, **kwargs)

    def _three_crops(self, waveform: torch.Tensor) -> torch.Tensor:
        length = waveform.shape[-1]
        if length <= self.num_samples:
            crop = F.pad(waveform, (0, max(0, self.num_samples - length)))
            return torch.stack([crop.squeeze(0)] * 3)
        maximum = length - self.num_samples
        offsets = (0, maximum // 2, maximum)
        return torch.stack(
            [
                waveform[..., offset : offset + self.num_samples].squeeze(0)
                for offset in offsets
            ]
        )

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.rows[index]
        sample_id = row["sample_id"]
        path = self.split_dir / "mixture" / f"{sample_id}.wav"
        waveform = load_pcm16(path, self.sample_rate)
        target = torch.zeros(self.num_classes, dtype=torch.float32)
        for original_index in ast.literal_eval(row["label_indices"]):
            model_index = self.original_to_model.get(int(original_index))
            if model_index is not None:
                target[model_index] = 1.0
        return {
            "waveform": self._three_crops(waveform),
            "target": target,
            "snr": float(row["target_snr_db"]),
            "sample_id": sample_id,
        }

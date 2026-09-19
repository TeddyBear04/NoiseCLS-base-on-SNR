"""Manifest loader shared with the BEATs audio_noise_capstone layout."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import soundfile as sound_file
import torch
import torchaudio
from torch.nn import functional
from torch.utils.data import Dataset


def load_audio(path: Path, sample_rate: int, samples: int) -> torch.Tensor:
    audio, source_rate = sound_file.read(path, dtype="float32", always_2d=True)
    waveform = torch.from_numpy(audio.copy()).mean(dim=1)
    if source_rate != sample_rate:
        waveform = torchaudio.functional.resample(waveform, source_rate, sample_rate)
    waveform = waveform[:samples]
    return functional.pad(waveform, (0, max(0, samples - waveform.numel())))


class MixNoiseDataset(Dataset[dict[str, Any]]):
    def __init__(self, root: str | Path, split: str, seconds: float, snr_min: float, snr_max: float) -> None:
        self.root = Path(root)
        self.sample_rate = 16_000
        self.samples = int(seconds * self.sample_rate)
        self.labels = (self.root / "labels.txt").read_text(encoding="utf-8").splitlines()
        self.label_to_index = {label: index for index, label in enumerate(self.labels)}
        with (self.root / "manifest.csv").open(encoding="utf-8-sig", newline="") as stream:
            rows = list(csv.DictReader(stream))
        self.rows = [
            row for row in rows
            if row["split"] == split and snr_min <= float(row["target_snr_db"]) <= snr_max
        ]
        if not self.rows:
            raise ValueError(f"No {split} samples in requested SNR band [{snr_min}, {snr_max}].")

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.rows[index]
        return {
            "mixture": load_audio(self.root / row["mixture_path"], self.sample_rate, self.samples),
            "noise": load_audio(self.root / row["noise_path"], self.sample_rate, self.samples),
            "target": self.label_to_index[row["label_names"]],
        }

"""Dataset interfaces for the 36-label Float32 noise-only dataset export."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import soundfile as sf
import torch
import torchaudio
from torch.utils.data import Dataset


def load_float_audio(
    path: str | Path,
    sample_rate: int = 16_000,
    num_samples: int | None = None,
) -> torch.Tensor:
    """Load PCM or IEEE-float audio as a mono float tensor shaped ``[samples]``."""
    waveform, source_rate = sf.read(
        str(path), dtype="float32", always_2d=True
    )
    tensor = torch.from_numpy(waveform.copy()).mean(dim=1)
    if source_rate != sample_rate:
        tensor = torchaudio.functional.resample(tensor, source_rate, sample_rate)
    if num_samples is not None:
        tensor = tensor[:num_samples]
        if tensor.shape[0] < num_samples:
            tensor = torch.nn.functional.pad(
                tensor, (0, num_samples - tensor.shape[0])
            )
    return tensor


def load_mix_manifest(root: str | Path) -> list[dict[str, str]]:
    root = Path(root)
    with (root / "manifest.csv").open(
        encoding="utf-8-sig", newline=""
    ) as file:
        return list(csv.DictReader(file))


class NoiseOnlyDataset(Dataset):
    """Load oracle-noise waveforms and their single-label targets."""

    def __init__(
        self,
        root: str | Path,
        split: str,
        seconds: float = 4.0,
        rows: list[dict[str, str]] | None = None,
    ) -> None:
        self.root = Path(root)
        self.split = split
        self.sample_rate = 16_000
        self.num_samples = int(seconds * self.sample_rate)
        self.labels = (self.root / "labels.txt").read_text(
            encoding="utf-8"
        ).splitlines()
        self.label_to_index = {
            label: index for index, label in enumerate(self.labels)
        }
        source_rows = load_mix_manifest(self.root) if rows is None else rows
        self.rows = [row for row in source_rows if row["split"] == split]
        if not self.rows:
            raise ValueError(f"No rows found for split={split!r}")
        unknown = sorted(
            {row["label_names"] for row in self.rows} - set(self.label_to_index)
        )
        if unknown:
            raise ValueError(f"Manifest contains unknown labels: {unknown}")

    @property
    def num_classes(self) -> int:
        return len(self.labels)

    def __len__(self) -> int:
        return len(self.rows)

    def _load(self, row: dict[str, str], column: str) -> torch.Tensor:
        return load_float_audio(
            self.root / row[column], self.sample_rate, self.num_samples
        )

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.rows[index]
        return {
            "waveform": self._load(row, "noise_path"),
            "target": self.label_to_index[row["label_names"]],
            "label_name": row["label_names"],
            "snr": int(float(row["target_snr_db"])),
            "sample_id": row["sample_id"],
            "noise_ytid": row["noise_ytid"],
        }


# Kept as an import-compatible alias for callers that used the previous name.
MixNoiseDataset = NoiseOnlyDataset

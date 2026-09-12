import ast
import csv
import wave
from pathlib import Path

import torch
import torchaudio
from torch.utils.data import Dataset


def load_pcm16(path: str | Path, sample_rate: int = 16_000, num_samples: int | None = None):
    """Load a PCM16 WAV as a mono float tensor shaped [1, samples]."""
    path = Path(path)
    with wave.open(str(path), "rb") as wav:
        channels = wav.getnchannels()
        source_rate = wav.getframerate()
        sample_width = wav.getsampwidth()
        frames = wav.readframes(wav.getnframes())
    if sample_width != 2:
        raise ValueError(f"Expected PCM16 WAV, got {sample_width * 8}-bit: {path}")
    waveform = torch.frombuffer(bytearray(frames), dtype=torch.int16)
    waveform = waveform.reshape(-1, channels).transpose(0, 1).float() / 32768.0
    waveform = waveform.mean(dim=0, keepdim=True)
    if source_rate != sample_rate:
        waveform = torchaudio.functional.resample(waveform, source_rate, sample_rate)
    if num_samples is not None:
        waveform = waveform[..., :num_samples]
        waveform = torch.nn.functional.pad(waveform, (0, num_samples - waveform.shape[-1]))
    return waveform


class NoiseDataset(Dataset):
    """Loads local WAVs by sample_id; manifest absolute paths are intentionally ignored."""

    def __init__(self, root: str | Path, split: str, seconds: float = 6.0, limit: int | None = None):
        self.root = Path(root)
        self.split_dir = self.root / split
        # These exported CSV files include a UTF-8 BOM. ``utf-8-sig`` removes it
        # instead of exposing keys such as ``\ufeffsample_id`` to DictReader.
        with (self.split_dir / "manifest.csv").open(encoding="utf-8-sig", newline="") as f:
            self.rows = list(csv.DictReader(f))
        if limit is not None:
            self.rows = self.rows[:limit]

        with (self.root / "selected_labels.csv").open(encoding="utf-8-sig", newline="") as f:
            selected = list(csv.DictReader(f))
        self.original_to_model = {
            int(row["original_index"]): int(row["model_index"]) for row in selected
        }
        self.num_classes = len(selected)
        self.sample_rate = 16_000
        self.num_samples = int(seconds * self.sample_rate)

    def __len__(self):
        return len(self.rows)

    def class_counts(self):
        counts = torch.zeros(self.num_classes, dtype=torch.float32)
        for row in self.rows:
            for original_index in ast.literal_eval(row["label_indices"]):
                model_index = self.original_to_model.get(int(original_index))
                if model_index is not None:
                    counts[model_index] += 1
        return counts

    def _load(self, folder: str, sample_id: str) -> torch.Tensor:
        path = self.split_dir / folder / f"{sample_id}.wav"
        # The dataset is RIFF PCM16. Reading it with the standard library keeps
        # the project independent of optional torchaudio I/O backends such as
        # FFmpeg, SoundFile or TorchCodec.
        return load_pcm16(path, self.sample_rate, self.num_samples)

    def __getitem__(self, index: int):
        row = self.rows[index]
        sample_id = row["sample_id"]
        target = torch.zeros(self.num_classes, dtype=torch.float32)
        for original_index in ast.literal_eval(row["label_indices"]):
            model_index = self.original_to_model.get(int(original_index))
            if model_index is not None:
                target[model_index] = 1.0
        return {
            "mixture": self._load("mixture", sample_id),
            "oracle_noise": self._load("oracle_noise", sample_id),
            "target": target,
            "sample_id": sample_id,
        }

"""Dataset and evaluation utilities for ``21_labels_dataset``."""

from __future__ import annotations

import ast
import csv
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_recall_fscore_support,
)
from torch.nn import functional as F
from torch.utils.data import Dataset

from noise_pipeline.data import load_pcm16


SPLIT_DIRS = {
    "train": "train_single",
    "validation": "validation_single",
    "test": "test_single",
}


def load_label_metadata(root: str | Path) -> list[dict[str, str]]:
    with (Path(root) / "selected_labels.csv").open(
        encoding="utf-8-sig", newline=""
    ) as file:
        return sorted(
            csv.DictReader(file), key=lambda row: int(row["model_index"])
        )


class Audio21Dataset(Dataset):
    """Mixture-only, multi-label view with local sample-id path resolution."""

    def __init__(
        self,
        root: str | Path,
        split: str,
        seconds: float = 6.0,
        crop: str = "center",
        limit: int | None = None,
        seed: int = 2026,
    ) -> None:
        if split not in SPLIT_DIRS:
            raise ValueError(f"Unknown split: {split!r}")
        if crop not in ("center", "random", "start"):
            raise ValueError(f"Unknown crop mode: {crop!r}")
        self.root = Path(root)
        self.split = split
        self.split_dir = self.root / SPLIT_DIRS[split]
        with (self.split_dir / "manifest.csv").open(
            encoding="utf-8-sig", newline=""
        ) as file:
            self.rows = list(csv.DictReader(file))
        if limit is not None:
            self.rows = self.rows[:limit]
        self.metadata = load_label_metadata(self.root)
        self.labels = [row["display_name"] for row in self.metadata]
        self.original_to_model = {
            int(row["original_index"]): int(row["model_index"])
            for row in self.metadata
        }
        self.sample_rate = 16_000
        self.num_samples = int(seconds * self.sample_rate)
        self.crop = crop
        self.seed = seed

    @property
    def num_classes(self) -> int:
        return len(self.labels)

    def __len__(self) -> int:
        return len(self.rows)

    def _crop(self, waveform: torch.Tensor, index: int) -> torch.Tensor:
        length = waveform.shape[-1]
        if length < self.num_samples:
            return F.pad(waveform, (0, self.num_samples - length))
        if length == self.num_samples:
            return waveform
        if self.crop == "start":
            offset = 0
        elif self.crop == "center":
            offset = (length - self.num_samples) // 2
        else:
            # Deterministic per epoch-independent sample crop. DataLoader shuffle
            # still changes batches while repeated experiments remain reproducible.
            offset = random.Random(self.seed + index).randrange(
                length - self.num_samples + 1
            )
        return waveform[..., offset : offset + self.num_samples]

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.rows[index]
        sample_id = row["sample_id"]
        path = self.split_dir / "mixture" / f"{sample_id}.wav"
        waveform = self._crop(load_pcm16(path, self.sample_rate), index)
        target = torch.zeros(self.num_classes, dtype=torch.float32)
        for original_index in ast.literal_eval(row["label_indices"]):
            model_index = self.original_to_model.get(int(original_index))
            if model_index is not None:
                target[model_index] = 1.0
        return {
            "waveform": waveform.squeeze(0),
            "target": target,
            "snr": float(row["target_snr_db"]),
            "sample_id": sample_id,
        }


def _safe_average_precision(targets: np.ndarray, probabilities: np.ndarray):
    valid = targets.sum(axis=0) > 0
    per_class = np.zeros(targets.shape[1], dtype=np.float64)
    if valid.any():
        per_class[valid] = average_precision_score(
            targets[:, valid], probabilities[:, valid], average=None
        )
    return per_class, float(per_class[valid].mean()) if valid.any() else 0.0


def multilabel_metrics(
    probabilities: np.ndarray | torch.Tensor,
    targets: np.ndarray | torch.Tensor,
    snrs: np.ndarray | torch.Tensor,
    labels: list[str],
    threshold: float | np.ndarray = 0.5,
) -> dict[str, Any]:
    if isinstance(probabilities, torch.Tensor):
        probabilities = probabilities.float().cpu().numpy()
    if isinstance(targets, torch.Tensor):
        targets = targets.cpu().numpy()
    if isinstance(snrs, torch.Tensor):
        snrs = snrs.float().cpu().numpy()
    targets = targets.astype(np.int64)
    predictions = probabilities >= threshold
    precision, recall, class_f1, support = precision_recall_fscore_support(
        targets, predictions, average=None, zero_division=0
    )
    class_ap, macro_ap = _safe_average_precision(targets, probabilities)
    per_class = {
        label: {
            "precision": float(precision[index]),
            "recall": float(recall[index]),
            "f1": float(class_f1[index]),
            "average_precision": float(class_ap[index]),
            "support": int(support[index]),
        }
        for index, label in enumerate(labels)
    }
    result: dict[str, Any] = {
        "micro_f1": float(f1_score(targets, predictions, average="micro", zero_division=0)),
        "macro_f1": float(f1_score(targets, predictions, average="macro", zero_division=0)),
        "micro_precision": float(precision_recall_fscore_support(
            targets, predictions, average="micro", zero_division=0
        )[0]),
        "micro_recall": float(precision_recall_fscore_support(
            targets, predictions, average="micro", zero_division=0
        )[1]),
        "macro_precision": float(precision.mean()),
        "macro_recall": float(recall.mean()),
        "micro_average_precision": float(
            average_precision_score(targets, probabilities, average="micro")
        ),
        "macro_average_precision": macro_ap,
        "exact_match": float((predictions == targets).all(axis=1).mean()),
        "predicted_labels_per_sample": float(predictions.sum(axis=1).mean()),
        "true_labels_per_sample": float(targets.sum(axis=1).mean()),
        "per_class": per_class,
        "per_snr": {},
    }
    boundaries = (-float("inf"), 0.0, 5.0, 10.0, 15.0, float("inf"))
    names = ("[-5,0)", "[0,5)", "[5,10)", "[10,15)", "[15,20]")
    for lower, upper, name in zip(boundaries[:-1], boundaries[1:], names):
        mask = (snrs >= lower) & (snrs < upper)
        if name == "[15,20]":
            mask = (snrs >= lower) & (snrs <= upper)
        if not mask.any():
            continue
        _, bin_map = _safe_average_precision(targets[mask], probabilities[mask])
        result["per_snr"][name] = {
            "samples": int(mask.sum()),
            "micro_f1": float(
                f1_score(targets[mask], predictions[mask], average="micro", zero_division=0)
            ),
            "macro_f1": float(
                f1_score(targets[mask], predictions[mask], average="macro", zero_division=0)
            ),
            "macro_average_precision": bin_map,
        }
    return result

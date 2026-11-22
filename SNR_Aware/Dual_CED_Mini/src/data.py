"""
Data loader and dataset definition for Dual-Expert BEATs.
Supports mix-dataset-v2 manifests, lazy FRCRN memory-mapped cache (mix + frcrn),
multiprocessing-safe pickling, and graceful synthetic fallback for smoke-testing.
"""

from __future__ import annotations

import ast
import csv
import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf
import torch
from torch import Tensor
from torch.utils.data import Dataset

LOGGER = logging.getLogger(__name__)

DEFAULT_36_CLASSES = [
    "Acoustic guitar", "Applause", "Bark", "Burping, eructation", "Bus", "Chime",
    "Clarinet", "Computer keyboard", "Cough", "Cowbell", "Drawer open or close",
    "Electric piano", "Fart", "Finger snapping", "Fireworks", "Flute",
    "Glockenspiel", "Gong", "Gunshot, gunfire", "Harmonica", "Hi-hat",
    "Keys jangling", "Knock", "Laughter", "Meow", "Microwave oven",
    "Saxophone", "Scissors", "Shatter", "Squeak", "Tambourine",
    "Tearing", "Telephone", "Trumpet", "Violin, fiddle", "Writing"
]


def load_class_names(path: Path | None = None) -> list[str]:
    """Load ordered class names from labels.txt or provide fallback default."""
    if path is not None and path.is_file():
        names = [line.strip() for line in path.read_text(encoding="utf-8-sig").splitlines() if line.strip()]
        if len(names) == 36:
            return names
    return list(DEFAULT_36_CLASSES)


def _parse_label_names(value: str) -> list[str]:
    try:
        parsed = json.loads(value)
    except Exception:
        try:
            parsed = ast.literal_eval(value)
        except Exception:
            parsed = None
    if isinstance(parsed, list):
        return [str(item).strip() for item in parsed if str(item).strip()]
    return [item.strip().strip('"\'') for item in value.split(",") if item.strip()]


class MixtureDataset(Dataset):
    """
    Dataset loader for 4-second audio mixtures across 36 environmental noise classes.
    Supports reading from lazy FRCRN memory-mapped cache (.npy) or raw audio fallback.
    Implements lightweight pickling (__getstate__ / __setstate__) for worker processes.
    """

    def __init__(
        self,
        root: str | Path,
        split: str = "train",
        manifest_file: str = "manifest.csv",
        labels_file: str = "labels.txt",
        frcrn_root: str | Path | None = None,
        use_frcrn: bool = True,
        sample_rate: int = 16000,
        clip_seconds: float = 4.0,
        num_classes: int = 36,
        max_samples: int | None = None,
        seed: int = 2026,
    ) -> None:
        self.root = Path(root)
        self.split = split
        self.manifest_file = manifest_file
        self.labels_file = labels_file
        self.frcrn_root = str(frcrn_root) if frcrn_root is not None else None
        self.use_frcrn = use_frcrn
        self.sample_rate = sample_rate
        self.clip_seconds = clip_seconds
        self.num_samples = int(clip_seconds * sample_rate)
        self.num_classes = num_classes
        self.max_samples = max_samples
        self.seed = seed
        self.class_names = load_class_names(self.root / labels_file if self.root.is_dir() else None)
        self.class_to_idx = {name.casefold(): i for i, name in enumerate(self.class_names)}

        self.rows: list[dict[str, Any]] = []
        self.is_synthetic = False

        # Lazy cache state: only store string path; open mmap per-process
        self.cache_path: str | None = None
        self._cached_array: np.ndarray | None = None

        # 1. Attempt to find manifest
        manifest_path = self._resolve_manifest_path(manifest_file)
        self.manifest_path = manifest_path
        if manifest_path is not None and manifest_path.is_file():
            with manifest_path.open(encoding="utf-8-sig", newline="") as f:
                reader = csv.DictReader(f)
                self.rows = list(reader)
            if max_samples and len(self.rows) > max_samples:
                self.rows = self.rows[:max_samples]
        else:
            # Fallback to deterministic synthetic samples for smoke-testing
            self.is_synthetic = True
            n_samples = max_samples if max_samples is not None else (32 if split == "train" else 16)
            rng = np.random.RandomState(seed + (0 if split == "train" else 100))
            snr_choices = [-5.0, 0.0, 5.0, 10.0, 15.0, 20.0]
            for i in range(n_samples):
                self.rows.append({
                    "sample_id": f"syn_{split}_{i:04d}",
                    "target_snr_db": str(float(rng.choice(snr_choices))),
                    "target_idx": str(int(rng.randint(0, num_classes))),
                    "mixture_path": "",
                    "label_names": self.class_names[i % num_classes],
                })

        # 2. Check for FRCRN memory-mapped cache (mix + frcrn input)
        if self.use_frcrn and not self.is_synthetic and self.frcrn_root:
            cache_names = {
                "train": "train.npy",
                "validation": "validation.npy",
                "eval": "validation.npy",
                "val": "validation.npy",
                "test": "test.npy",
            }
            c_name = cache_names.get(self.split, f"{self.split}.npy")
            candidate_cache = Path(self.frcrn_root) / c_name
            if candidate_cache.is_file():
                self.cache_path = str(candidate_cache)
                LOGGER.info("MixtureDataset [%s]: Registered FRCRN cache at %s", self.split, self.cache_path)

    def __getstate__(self) -> dict[str, Any]:
        """Drop open mmap file handles before pickling across worker processes."""
        state = self.__dict__.copy()
        state["_cached_array"] = None
        return state

    def __setstate__(self, state: dict[str, Any]) -> None:
        """Restore state in worker process with fresh lazy mmap."""
        self.__dict__.update(state)
        self._cached_array = None

    def _resolve_manifest_path(self, filename: str) -> Path | None:
        if not self.root.is_dir():
            return None
        partition = {"train": "train", "validation": "eval", "test": "test"}.get(self.split, self.split)
        candidate1 = self.root / partition / filename
        if candidate1.is_file():
            return candidate1
        candidate2 = self.root / filename
        if candidate2.is_file():
            return candidate2
        return None

    def __len__(self) -> int:
        return len(self.rows)

    def _get_mmap_array(self) -> np.ndarray | None:
        if self._cached_array is None and self.cache_path is not None:
            p = Path(self.cache_path)
            if p.is_file():
                try:
                    self._cached_array = np.load(str(p), mmap_mode="r")
                except Exception as exc:
                    LOGGER.warning("Could not open mmap at %s: %s", p, exc)
                    self._cached_array = None
        return self._cached_array

    def _load_raw_waveform(self, relative_path: str, index: int) -> Tensor:
        if self.is_synthetic or not relative_path:
            rng = np.random.RandomState(self.seed + index)
            t = np.linspace(0, 4.0, self.num_samples, endpoint=False)
            freq = float(rng.uniform(100.0, 2000.0))
            audio = 0.3 * np.sin(2 * np.pi * freq * t) + 0.1 * rng.randn(self.num_samples)
            return torch.from_numpy(audio.astype(np.float32))

        path = Path(relative_path)
        if not path.is_absolute():
            path = self.root / path
        if not path.is_file():
            return torch.randn(self.num_samples, dtype=torch.float32) * 0.1

        audio, sr = sf.read(str(path), dtype="float32", always_2d=True)
        wave = torch.from_numpy(audio.mean(axis=1).copy())
        if wave.numel() < self.num_samples:
            wave = torch.nn.functional.pad(wave, (0, self.num_samples - wave.numel()))
        return wave[: self.num_samples].contiguous()

    def _load_frcrn_waveform(self, relative_path: str, index: int) -> Tensor:
        arr = self._get_mmap_array()
        if arr is not None and index < len(arr):
            raw = np.array(arr[index], dtype=np.float32)
            if len(raw) >= self.num_samples:
                raw = raw[: self.num_samples]
            else:
                raw = np.pad(raw, (0, self.num_samples - len(raw)))
            return torch.from_numpy(raw)
        return self._load_raw_waveform(relative_path, index)

    def _load_waveform(self, relative_path: str, index: int) -> Tensor:
        # Default to FRCRN if cache exists, otherwise raw
        return self._load_frcrn_waveform(relative_path, index)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.rows[index]
        rel_path = row.get("mixture_path", "")
        wave_raw = self._load_raw_waveform(rel_path, index)
        wave_frcrn = self._load_frcrn_waveform(rel_path, index) if self.use_frcrn else wave_raw

        # Robust target index extraction
        target_idx = 0
        if "multi_hot_36" in row:
            try:
                mh = json.loads(row["multi_hot_36"])
                target_idx = int(np.argmax(mh))
            except Exception:
                target_idx = 0
        elif "target_idx" in row:
            target_idx = int(row["target_idx"])
        elif "label_names" in row:
            names = _parse_label_names(row["label_names"])
            if names and names[0].casefold() in self.class_to_idx:
                target_idx = self.class_to_idx[names[0].casefold()]

        snr_db = float(row.get("target_snr_db", row.get("achieved_snr_db", 0.0)))
        sample_id = row.get("sample_id", f"sample_{index}")

        return {
            "waveform": wave_raw,
            "waveform_raw": wave_raw,
            "waveform_frcrn": wave_frcrn,
            "target_idx": torch.tensor(target_idx, dtype=torch.long),
            "snr_db": torch.tensor(snr_db, dtype=torch.float32),
            "sample_id": sample_id,
            "index": index,
        }


def load_class_mids(dataset: MixtureDataset) -> list[str] | None:
    """Return AudioSet MIDs in the dataset class order when manifest metadata exists."""
    mid_by_name: dict[str, str] = {}
    if hasattr(dataset, "manifest_path") and dataset.manifest_path and Path(dataset.manifest_path).is_file():
        try:
            with open(dataset.manifest_path, encoding="utf-8-sig", newline="") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    name = row.get("label_names", "").strip()
                    mid = row.get("label_mids", "").strip()
                    if name and mid:
                        key = name.casefold()
                        if key not in mid_by_name:
                            mid_by_name[key] = mid
        except Exception:
            pass

    if not mid_by_name:
        for row in dataset.rows:
            name = row.get("label_names", "").strip()
            mid = row.get("label_mids", "").strip()
            if not name or not mid:
                continue
            key = name.casefold()
            if key not in mid_by_name:
                mid_by_name[key] = mid

    if not mid_by_name:
        return None
    missing = [name for name in dataset.class_names if name.casefold() not in mid_by_name]
    if missing:
        LOGGER.warning("Missing AudioSet MIDs for %d classes: %s", len(missing), missing)
        return None
    return [mid_by_name[name.casefold()] for name in dataset.class_names]


def build_datasets(
    config: dict[str, Any],
    max_samples: int | None = None,
    include_test: bool = True,
) -> dict[str, MixtureDataset]:
    """Construct dataset instances, optionally excluding the test split entirely."""
    dataset_cfg = config.get("dataset", {})
    root_path = dataset_cfg.get("path", "/dataset/mix-dataset-v2")
    frcrn_root = dataset_cfg.get("frcrn_root")
    if frcrn_root is None:
        for candidate in [Path("/datasets/frcrn"), Path("/marimo/datasets/frcrn"), Path("/datasets/mix-frcrn")]:
            if candidate.is_dir():
                frcrn_root = str(candidate)
                break

    manifest_file = dataset_cfg.get("manifest_file", "manifest.csv")
    labels_file = dataset_cfg.get("labels_file", "labels.txt")
    sample_rate = int(dataset_cfg.get("sample_rate", 16000))
    clip_seconds = float(dataset_cfg.get("clip_seconds", 4.0))
    num_classes = int(config.get("experiment", {}).get("num_classes", 36))
    seed = int(config.get("experiment", {}).get("seed", 2026))

    splits = ("train", "validation", "test") if include_test else ("train", "validation")
    return {
        split: MixtureDataset(
            root=root_path,
            split=split,
            manifest_file=manifest_file,
            labels_file=labels_file,
            frcrn_root=frcrn_root,
            use_frcrn=bool(dataset_cfg.get("use_frcrn", True)),
            sample_rate=sample_rate,
            clip_seconds=clip_seconds,
            num_classes=num_classes,
            max_samples=max_samples,
            seed=seed,
        )
        for split in splits
    }


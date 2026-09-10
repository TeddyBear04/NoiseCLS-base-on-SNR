"""Manifest-based loader for the speech-plus-noise datasets.

Two on-disk layouts are understood:

* ``labels.txt`` plus a ``multi_hot_<N>`` manifest column (the 36-label dataset).
* ``selected_labels.csv`` plus a ``label_indices`` manifest column (the older
  21-label dataset).
"""

from __future__ import annotations

import csv
import json
import logging
import math
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import soundfile as sf
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from config import AudioFeaturesConfig, SplitterConfig, TrainAugmentationConfig

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class LabelInfo:
    model_index: int
    original_index: int
    mid: str
    display_name: str


@dataclass(frozen=True)
class ManifestRecord:
    sample_id: str
    noise_source_id: str
    split_directory: str
    signal_path: Path
    clean_path: Path
    noise_path: Path
    target: Tuple[float, ...]
    target_snr_db: float
    duration_seconds: float


def _read_label_catalog_txt(path: Path) -> List[LabelInfo]:
    """One display name per line; line order defines both indices."""
    names = [line.strip() for line in path.read_text(encoding="utf-8-sig").splitlines()]
    names = [name for name in names if name]
    return [
        LabelInfo(model_index=index, original_index=index, mid="", display_name=name)
        for index, name in enumerate(names)
    ]


def _read_label_catalog_csv(path: Path) -> List[LabelInfo]:
    """model_index/original_index/mid/display_name, as used by 21_labels_dataset."""
    labels: List[LabelInfo] = []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            labels.append(
                LabelInfo(
                    model_index=int(row["model_index"]),
                    original_index=int(row["original_index"]),
                    mid=row["mid"],
                    display_name=row["display_name"],
                )
            )
    return labels


def read_label_catalog(dataset_root: Path, filename: str = "labels.txt") -> List[LabelInfo]:
    path = dataset_root / filename
    if not path.is_file():
        raise FileNotFoundError(f"Label catalog not found: {path}")

    reader = _read_label_catalog_csv if path.suffix.lower() == ".csv" else _read_label_catalog_txt
    labels = reader(path)
    if not labels:
        raise ValueError(f"Label catalog is empty: {path}")
    labels.sort(key=lambda item: item.model_index)
    expected = list(range(len(labels)))
    actual = [item.model_index for item in labels]
    if actual != expected:
        raise ValueError(f"model_index in {path} must be contiguous from 0; got {actual}")
    return labels


def _parse_json_list(column: str, raw_value: str) -> List[int]:
    try:
        values = json.loads(raw_value)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid {column} value: {raw_value!r}") from exc
    if not isinstance(values, list):
        raise ValueError(f"{column} must be a JSON list, got: {raw_value!r}")
    return [int(value) for value in values]


def _resolve_label_column(fieldnames: Sequence[str], manifest_path: Path) -> Tuple[str, str]:
    """Return (column, kind) for the column that carries a row's labels.

    kind ``indices``: a JSON list of original label indices.
    kind ``multi_hot``: a JSON list of 0/1 flags, one per original label.
    """
    if "label_indices" in fieldnames:
        return "label_indices", "indices"
    multi_hot = [name for name in fieldnames if name == "multi_hot" or name.startswith("multi_hot_")]
    if len(multi_hot) == 1:
        return multi_hot[0], "multi_hot"
    if len(multi_hot) > 1:
        raise ValueError(f"{manifest_path} has several multi-hot columns: {sorted(multi_hot)}")
    raise ValueError(
        f"{manifest_path} has no label column; expected 'label_indices' or 'multi_hot_<N>', "
        f"got: {sorted(fieldnames)}"
    )


def _row_original_indices(row: Dict[str, str], column: str, kind: str) -> List[int]:
    values = _parse_json_list(column, row[column])
    if kind == "indices":
        return values
    return [index for index, flag in enumerate(values) if flag]


def read_manifest(
    dataset_root: Path,
    split_directory: str,
    signal_type: str,
    labels: Sequence[LabelInfo],
    clean_directory: str = "clean",
    noise_directory: str = "noise",
) -> List[ManifestRecord]:
    manifest_path = dataset_root / split_directory / "manifest.csv"
    signal_dir = dataset_root / split_directory / signal_type
    clean_dir = dataset_root / split_directory / clean_directory
    noise_dir = dataset_root / split_directory / noise_directory
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")
    if not signal_dir.is_dir():
        raise FileNotFoundError(f"Audio directory not found: {signal_dir}")

    original_to_model = {item.original_index: item.model_index for item in labels}
    records: List[ManifestRecord] = []
    with manifest_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        label_column, label_kind = _resolve_label_column(reader.fieldnames or [], manifest_path)
        for row_number, row in enumerate(reader, start=2):
            sample_id = row["sample_id"]
            target = [0.0] * len(labels)
            for original_index in _row_original_indices(row, label_column, label_kind):
                if original_index not in original_to_model:
                    raise ValueError(
                        f"Unknown original label index {original_index} in {manifest_path}:{row_number}"
                    )
                target[original_to_model[original_index]] = 1.0
            records.append(
                ManifestRecord(
                    sample_id=sample_id,
                    noise_source_id=row.get("noise_ytid") or sample_id,
                    split_directory=split_directory,
                    signal_path=signal_dir / f"{sample_id}.wav",
                    clean_path=clean_dir / f"{sample_id}.wav",
                    noise_path=noise_dir / f"{sample_id}.wav",
                    target=tuple(target),
                    target_snr_db=float(row["target_snr_db"]),
                    duration_seconds=float(row["duration_seconds"]),
                )
            )
    if not records:
        raise ValueError(f"Manifest contains no samples: {manifest_path}")
    return records


def load_audio_file(path: Path, target_sample_rate: int) -> torch.Tensor:
    """Load a mono float waveform and resample only when metadata differs."""
    if not path.is_file():
        raise FileNotFoundError(f"WAV file not found: {path}")
    waveform, sample_rate = sf.read(str(path), dtype="float32", always_2d=True)
    waveform = waveform.mean(axis=1)
    if sample_rate != target_sample_rate:
        from math import gcd
        from scipy.signal import resample_poly

        divisor = gcd(int(sample_rate), int(target_sample_rate))
        waveform = resample_poly(
            waveform,
            up=target_sample_rate // divisor,
            down=sample_rate // divisor,
        ).astype(np.float32, copy=False)
    return torch.from_numpy(np.ascontiguousarray(waveform, dtype=np.float32))


def fixed_window(waveform: torch.Tensor, start: int, length: int) -> torch.Tensor:
    segment = waveform[start : start + length]
    if segment.numel() < length:
        segment = F.pad(segment, (0, length - segment.numel()))
    return segment


def sliding_window_starts(total_samples: int, window_samples: int, hop_samples: int) -> List[int]:
    if total_samples <= window_samples:
        return [0]
    starts = list(range(0, total_samples - window_samples + 1, hop_samples))
    last_start = total_samples - window_samples
    if starts[-1] != last_start:
        starts.append(last_start)
    return starts


def _peak_limit(waveform: torch.Tensor, maximum: float = 0.99) -> torch.Tensor:
    peak = waveform.abs().max()
    if peak > maximum:
        waveform = waveform * (maximum / peak)
    return waveform.to(torch.float32)


def random_pad_crop(
    waveform: torch.Tensor,
    length: int,
    padding_samples: int = 0,
) -> torch.Tensor:
    """Randomly crop a long source or pad-and-crop a short/fixed source.

    Pad-and-crop makes a real temporal shift possible for this dataset's
    four-second files, for which the old random crop always selected offset 0.
    """
    if waveform.numel() <= length and padding_samples > 0:
        waveform = F.pad(waveform, (padding_samples, padding_samples))
    if waveform.numel() < length:
        waveform = F.pad(waveform, (0, length - waveform.numel()))
    max_start = max(0, waveform.numel() - length)
    start = 0 if max_start == 0 else int(torch.randint(0, max_start + 1, (1,)).item())
    return fixed_window(waveform, start, length)


def random_time_shift(waveform: torch.Tensor, max_shift_samples: int) -> torch.Tensor:
    """Move a waveform in time with zero fill instead of circular wrapping."""
    if max_shift_samples <= 0 or waveform.numel() == 0:
        return waveform
    limit = min(max_shift_samples, max(0, waveform.numel() - 1))
    shift = int(torch.randint(-limit, limit + 1, (1,)).item())
    if shift == 0:
        return waveform
    shifted = torch.zeros_like(waveform)
    if shift > 0:
        shifted[shift:] = waveform[:-shift]
    else:
        shifted[:shift] = waveform[-shift:]
    return shifted


def speed_perturb(waveform: torch.Tensor, min_rate: float, max_rate: float) -> torch.Tensor:
    """Apply lightweight speed perturbation using linear resampling."""
    rate = float(torch.empty(1).uniform_(min_rate, max_rate).item())
    output_samples = max(1, int(round(waveform.numel() / rate)))
    return F.interpolate(
        waveform.view(1, 1, -1),
        size=output_samples,
        mode="linear",
        align_corners=False,
    ).view(-1)


def random_gain(waveform: torch.Tensor, min_db: float, max_db: float) -> torch.Tensor:
    gain_db = float(torch.empty(1).uniform_(min_db, max_db).item())
    return _peak_limit(waveform * (10.0 ** (gain_db / 20.0)))


def random_equalizer(waveform: torch.Tensor) -> torch.Tensor:
    """Apply a mild random smooth/high-frequency tilt without extra dependencies."""
    kernel_size = int(torch.randint(3, 16, (1,)).item())
    if kernel_size % 2 == 0:
        kernel_size += 1
    smooth = F.avg_pool1d(
        waveform.view(1, 1, -1),
        kernel_size=kernel_size,
        stride=1,
        padding=kernel_size // 2,
        count_include_pad=False,
    ).view(-1)
    strength = float(torch.empty(1).uniform_(-0.5, 0.8).item())
    return _peak_limit(waveform + strength * (smooth - waveform))


def random_reverb(waveform: torch.Tensor, sample_rate: int) -> torch.Tensor:
    """Add two low-level random echo taps as an inexpensive room response."""
    if waveform.numel() == 0:
        return waveform
    min_delay = max(1, int(round(0.025 * sample_rate)))
    max_delay = max(min_delay, int(round(0.120 * sample_rate)))
    delay = int(torch.randint(min_delay, max_delay + 1, (1,)).item())
    decay = float(torch.empty(1).uniform_(0.12, 0.35).item())
    reverberant = waveform.clone()
    if delay < waveform.numel():
        reverberant[delay:] += decay * waveform[:-delay]
    second_delay = delay * 2
    if second_delay < waveform.numel():
        reverberant[second_delay:] += (decay * decay) * waveform[:-second_delay]
    return _peak_limit(reverberant)


def same_class_mixup(
    first: torch.Tensor,
    second: torch.Tensor,
    alpha: float,
) -> tuple[torch.Tensor, float]:
    """Mix two same-label noises after RMS matching; the hard label stays valid."""
    eps = 1e-8
    first_power = first.square().mean().clamp_min(eps)
    second_power = second.square().mean().clamp_min(eps)
    second = second * torch.sqrt(first_power / second_power)
    coefficient = float(np.random.beta(alpha, alpha))
    mixed = coefficient * first + (1.0 - coefficient) * second
    return _peak_limit(mixed), coefficient


def mix_with_snr(clean: torch.Tensor, noise: torch.Tensor, snr_db: float) -> torch.Tensor:
    """Mix two equal-length stems at one fixed speech-to-noise ratio."""
    eps = 1e-8
    clean_power = clean.square().mean().clamp_min(eps)
    noise_power = noise.square().mean().clamp_min(eps)
    ratio = torch.tensor(10.0 ** (float(snr_db) / 10.0), dtype=noise.dtype)
    gain = torch.sqrt(clean_power / (noise_power * ratio))
    return _peak_limit(clean + noise * gain)


def mix_with_dynamic_snr(
    clean: torch.Tensor,
    noise: torch.Tensor,
    sample_rate: int,
    min_snr_db: float,
    max_snr_db: float,
    control_seconds: float,
) -> torch.Tensor:
    """Mix clean/noise using a linearly interpolated random SNR envelope."""
    eps = 1e-8
    clean_power = clean.square().mean().clamp_min(eps)
    noise_power = noise.square().mean().clamp_min(eps)
    control_samples = max(1, int(round(sample_rate * control_seconds)))
    control_count = max(2, math.ceil(clean.numel() / control_samples) + 1)
    snr_points = torch.empty(control_count).uniform_(min_snr_db, max_snr_db)
    snr_envelope = F.interpolate(
        snr_points.view(1, 1, -1), size=clean.numel(), mode="linear", align_corners=True
    ).view(-1)
    gain = torch.sqrt(clean_power / (noise_power * torch.pow(10.0, snr_envelope / 10.0)))
    mixture = clean + noise * gain
    return _peak_limit(mixture)


class NoiseManifestDataset(Dataset):
    """One random crop per train clip; deterministic sliding windows for val/test."""

    def __init__(
        self,
        dataset_config: SplitterConfig,
        audio_config: AudioFeaturesConfig,
        split: str,
        labels: Sequence[LabelInfo],
        cache_audio: bool = False,
        augmentation_config: TrainAugmentationConfig | None = None,
    ) -> None:
        if split not in {"train", "val", "test"}:
            raise ValueError(f"split must be train/val/test, got {split!r}")
        self.dataset_config = dataset_config
        self.audio_config = audio_config
        self.split = split
        self.training = split == "train"
        self.labels = list(labels)
        self.cache_audio = cache_audio
        self.augmentation = augmentation_config or TrainAugmentationConfig()
        self._audio_cache: Dict[Path, torch.Tensor] = {}
        self.clip_samples = int(round(audio_config.sample_rate * audio_config.clip_seconds))
        self.hop_samples = int(round(audio_config.sample_rate * audio_config.inference_hop_seconds))

        if dataset_config.signal_type != "noise":
            raise ValueError("NoiseOnly requires dataset_splitter.signal_type='noise'")
        if dataset_config.dynamic_snr_enabled:
            raise ValueError("NoiseOnly forbids dynamic-SNR clean/noise mixing")

        split_directory = {
            "train": dataset_config.train_directory,
            "val": dataset_config.validation_directory,
            "test": dataset_config.test_directory,
        }[split]
        self.records = read_manifest(
            dataset_root=Path(dataset_config.dataset_path),
            split_directory=split_directory,
            signal_type=dataset_config.signal_type,
            labels=self.labels,
            clean_directory=dataset_config.clean_directory,
            noise_directory=dataset_config.noise_directory,
        )

        self.index: List[Tuple[int, int]] = []
        for record_index, record in enumerate(self.records):
            if self.training:
                self.index.append((record_index, -1))
            else:
                total_samples = max(1, int(round(record.duration_seconds * audio_config.sample_rate)))
                for start in sliding_window_starts(total_samples, self.clip_samples, self.hop_samples):
                    self.index.append((record_index, start))

        targets = np.asarray([record.target for record in self.records], dtype=np.float32)
        self.positive_counts = targets.sum(axis=0)
        self.same_label_records: Dict[Tuple[float, ...], List[int]] = {}
        self.same_label_sources: Dict[Tuple[float, ...], Dict[str, List[int]]] = {}
        for record_index, record in enumerate(self.records):
            self.same_label_records.setdefault(record.target, []).append(record_index)
            self.same_label_sources.setdefault(record.target, {}).setdefault(
                record.noise_source_id, []
            ).append(record_index)
        logger.info(
            "%s dataset: %d clips, %d windows, signal=%s",
            split,
            len(self.records),
            len(self.index),
            dataset_config.signal_type,
        )
        if self.training and self.augmentation.enabled:
            logger.info("Noise-only waveform augmentation enabled")

    def __len__(self) -> int:
        return len(self.index)

    def _load(self, path: Path) -> torch.Tensor:
        if self.cache_audio and path in self._audio_cache:
            return self._audio_cache[path]
        waveform = load_audio_file(path, self.audio_config.sample_rate)
        if self.cache_audio:
            self._audio_cache[path] = waveform
        return waveform

    def _choose_train_start(self, total_samples: int) -> int:
        max_start = max(0, total_samples - self.clip_samples)
        if max_start == 0:
            return 0
        return int(torch.randint(0, max_start + 1, (1,)).item())

    def _different_record_index(self, current_index: int, candidates: Sequence[int]) -> int:
        if len(candidates) <= 1:
            return current_index
        offset = int(torch.randint(1, len(candidates), (1,)).item())
        position = candidates.index(current_index) if current_index in candidates else 0
        return int(candidates[(position + offset) % len(candidates)])

    def _augment_clean(self, waveform: torch.Tensor) -> torch.Tensor:
        config = self.augmentation
        if random.random() < config.clean_speed_probability:
            waveform = speed_perturb(
                waveform, config.clean_speed_min_rate, config.clean_speed_max_rate
            )
        padding = int(round(config.random_crop_padding_seconds * self.audio_config.sample_rate))
        waveform = random_pad_crop(waveform, self.clip_samples, padding)
        if random.random() < config.clean_gain_probability:
            waveform = random_gain(waveform, config.clean_gain_min_db, config.clean_gain_max_db)
        if random.random() < config.clean_reverb_probability:
            waveform = random_reverb(waveform, self.audio_config.sample_rate)
        return waveform

    def _augment_noise(self, waveform: torch.Tensor) -> torch.Tensor:
        config = self.augmentation
        if random.random() < config.noise_time_stretch_probability:
            waveform = speed_perturb(
                waveform,
                config.noise_time_stretch_min_rate,
                config.noise_time_stretch_max_rate,
            )
        padding = int(round(config.random_crop_padding_seconds * self.audio_config.sample_rate))
        waveform = random_pad_crop(waveform, self.clip_samples, padding)
        if random.random() < config.noise_time_shift_probability:
            max_shift = int(round(config.noise_time_shift_max_seconds * self.audio_config.sample_rate))
            waveform = random_time_shift(waveform, max_shift)
        if random.random() < config.noise_gain_probability:
            waveform = random_gain(waveform, config.noise_gain_min_db, config.noise_gain_max_db)
        if random.random() < config.noise_eq_probability:
            waveform = random_equalizer(waveform)
        if random.random() < config.noise_reverb_probability:
            waveform = random_reverb(waveform, self.audio_config.sample_rate)
        if random.random() < config.noise_polarity_probability:
            waveform = -waveform
        return waveform

    def _same_class_partner_index(self, record_index: int) -> int:
        """Choose the same label from a different original noise source."""
        record = self.records[record_index]
        source_groups = self.same_label_sources[record.target]
        source_ids = list(source_groups)
        if len(source_ids) <= 1:
            return self._different_record_index(
                record_index, self.same_label_records[record.target]
            )
        current_position = source_ids.index(record.noise_source_id)
        offset = int(torch.randint(1, len(source_ids), (1,)).item())
        partner_source = source_ids[(current_position + offset) % len(source_ids)]
        partners = source_groups[partner_source]
        return int(partners[int(torch.randint(0, len(partners), (1,)).item())])

    def _online_augmented_mixture(
        self,
        record_index: int,
        use_dynamic_snr: bool,
    ) -> tuple[torch.Tensor, bool, bool]:
        record = self.records[record_index]
        config = self.augmentation

        clean_index = record_index
        if random.random() < config.random_clean_probability:
            clean_index = self._different_record_index(
                record_index, range(len(self.records))
            )
        clean = self._augment_clean(self._load(self.records[clean_index].clean_path))
        noise = self._augment_noise(self._load(record.noise_path))

        mixed_same_class = False
        candidates = self.same_label_records[record.target]
        if len(candidates) > 1 and random.random() < config.same_class_mixup_probability:
            partner_index = self._same_class_partner_index(record_index)
            partner_noise = self._augment_noise(self._load(self.records[partner_index].noise_path))
            noise, _ = same_class_mixup(noise, partner_noise, config.same_class_mixup_alpha)
            mixed_same_class = True

        if use_dynamic_snr:
            waveform = mix_with_dynamic_snr(
                clean,
                noise,
                self.audio_config.sample_rate,
                self.dataset_config.dynamic_snr_min_db,
                self.dataset_config.dynamic_snr_max_db,
                self.dataset_config.dynamic_snr_control_seconds,
            )
        else:
            waveform = mix_with_snr(clean, noise, record.target_snr_db)
        return waveform, clean_index != record_index, mixed_same_class

    def _online_augmented_noise(self, record_index: int) -> tuple[torch.Tensor, bool]:
        """Augment noise without loading a clean or mixture waveform."""
        record = self.records[record_index]
        waveform = self._augment_noise(self._load(record.signal_path))
        mixed_same_class = False
        candidates = self.same_label_records[record.target]
        if len(candidates) > 1 and random.random() < self.augmentation.same_class_mixup_probability:
            partner_index = self._same_class_partner_index(record_index)
            partner = self._augment_noise(self._load(self.records[partner_index].signal_path))
            waveform, _ = same_class_mixup(
                waveform, partner, self.augmentation.same_class_mixup_alpha
            )
            mixed_same_class = True
        return waveform, mixed_same_class

    def __getitem__(self, index: int) -> Dict[str, object]:
        record_index, predefined_start = self.index[index]
        record = self.records[record_index]
        can_online_augment = (
            self.training
            and self.augmentation.enabled
        )
        use_dynamic_snr = False
        clean_replaced = False
        mixed_same_class = False

        if can_online_augment:
            waveform, mixed_same_class = self._online_augmented_noise(record_index)
            start = -self.audio_config.sample_rate
        else:
            source = self._load(record.signal_path)
            start = self._choose_train_start(source.numel()) if self.training else predefined_start
            waveform = fixed_window(source, start, self.clip_samples)

        return {
            "audio_name": record.sample_id,
            "waveform": waveform,
            "target": torch.tensor(record.target, dtype=torch.float32),
            "target_snr_db": torch.tensor(record.target_snr_db, dtype=torch.float32),
            "window_start_seconds": torch.tensor(start / self.audio_config.sample_rate, dtype=torch.float32),
            "dynamic_snr": torch.tensor(use_dynamic_snr, dtype=torch.bool),
            "waveform_augmented": torch.tensor(can_online_augment, dtype=torch.bool),
            "clean_replaced": torch.tensor(clean_replaced, dtype=torch.bool),
            "same_class_mixup": torch.tensor(mixed_same_class, dtype=torch.bool),
        }


def _seed_worker(worker_id: int) -> None:
    worker_seed = torch.initial_seed() % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


class NoiseDataLoaderManager:
    def __init__(
        self,
        dataset_config: SplitterConfig,
        audio_config: AudioFeaturesConfig,
        batch_size: int,
        num_workers: int = -1,
        cache_audio: bool = False,
        pin_memory: bool = True,
        seed: int = 2026,
        classes_num: int = 36,
        augmentation_config: TrainAugmentationConfig | None = None,
    ) -> None:
        self.dataset_config = dataset_config
        self.audio_config = audio_config
        self.batch_size = batch_size
        self.num_workers = (
            min(8, max(0, (os.cpu_count() or 2) // 2)) if num_workers == -1 else num_workers
        )
        self.cache_audio = cache_audio
        self.pin_memory = pin_memory
        self.seed = seed
        self.labels = read_label_catalog(
            Path(dataset_config.dataset_path), dataset_config.selected_labels_file
        )
        if len(self.labels) != classes_num:
            raise ValueError(
                f"Model expects {classes_num} classes, but "
                f"{dataset_config.selected_labels_file} defines {len(self.labels)}"
            )
        self.datasets = {
            split: NoiseManifestDataset(
                dataset_config=dataset_config,
                audio_config=audio_config,
                split=split,
                labels=self.labels,
                cache_audio=cache_audio,
                augmentation_config=augmentation_config,
            )
            for split in ("train", "val", "test")
        }

    @property
    def label_names(self) -> List[str]:
        return [item.display_name for item in self.labels]

    def positive_class_weights(self, max_weight: float = 20.0) -> torch.Tensor:
        dataset = self.datasets["train"]
        positives = torch.tensor(dataset.positive_counts, dtype=torch.float32)
        negatives = float(len(dataset.records)) - positives
        return (negatives / positives.clamp_min(1.0)).clamp(min=1.0, max=max_weight)

    def get_dataloader(self, split: str, shuffle: bool | None = None) -> DataLoader:
        if split not in self.datasets:
            raise ValueError(f"Unknown split: {split}")
        if shuffle is None:
            shuffle = split == "train"
        generator = torch.Generator()
        generator.manual_seed(self.seed)
        return DataLoader(
            self.datasets[split],
            batch_size=self.batch_size,
            shuffle=shuffle,
            drop_last=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory and torch.cuda.is_available(),
            persistent_workers=self.num_workers > 0,
            worker_init_fn=_seed_worker,
            generator=generator,
        )


# Compatibility alias for notebooks that imported the original class name.
FishVoiceDataLoader = NoiseDataLoaderManager

from __future__ import annotations

import csv
import json
import logging
import os
import zlib
from abc import ABC, abstractmethod
from collections.abc import Iterator, Mapping, Sequence
from functools import cache
from math import ceil, gcd
from pathlib import Path
from typing import Any, TypedDict

import numpy as np
import soundfile as sf
import torch
import yaml
from scipy.signal import resample_poly
from torch import Tensor
from torch.utils.data import DataLoader, Dataset, Sampler, Subset

ROOT = Path(__file__).resolve().parents[1]
URL_CONFIG = ROOT / "config" / "url.yaml"
LABELS_FALLBACK = ROOT / "config" / "audioset_40_labels.csv"
GROUP_MAP_FALLBACK = ROOT / "config" / "group_map.json"
AUDIOSET_KEY = "audioset-40l"
MIXTURE_KEY = "mixture-36l"
VLSP_KEY = "vlsp-100h"
VLSP_CONFIG = ROOT / "config" / "vlsp.yaml"
AUDIO_EXTENSIONS = (".wav", ".flac")
SAMPLE_RATE = 16000
ALL_SNR = (-5, 0, 5, 10, 15, 20)
TEACHER_CACHE_DIR = "cache"

logger = logging.getLogger(__name__)


class AudioItem(TypedDict, total=False):
    waveform: Tensor
    target: Tensor
    snr_db: Tensor
    sample_id: str
    noise: Tensor
    clean: Tensor
    label: Tensor
    group: Tensor
    noise_ytid: str
    teacher_logits36: Tensor


def dataset_env_name(key: str) -> str:
    return f"MLNC_{key.upper().replace('-', '_')}"


def resolve_dataset_root(key: str, url_config: Path = URL_CONFIG) -> Path:
    value = os.environ.get(dataset_env_name(key))
    if not value:
        with open(url_config, encoding="utf-8") as f:
            value = yaml.safe_load(f).get(key)
    if not value:
        raise KeyError(f"'{key}' is missing or empty in {url_config}")
    root = Path(value)
    if not root.is_dir():
        raise FileNotFoundError(f"Dataset directory does not exist: {root}")
    return root


def read_csv(path: Path) -> list[dict[str, str]]:
    with open(path, encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def read_class_mids(class_names: Sequence[str], labels_file: Path = LABELS_FALLBACK) -> list[str]:
    mids = {row["name"]: row["mid"] for row in read_csv(labels_file)}
    missing = [name for name in class_names if name not in mids]
    if missing:
        raise KeyError(f"classes not in {labels_file}: {missing}")
    return [mids[name] for name in class_names]


def snr_band(snr_db: float, edges: Sequence[float]) -> int:
    """Chỉ số khoảng SNR: số cạnh <= snr_db, vd cạnh [5, 15]: -5, 0 -> 0; 5, 10 -> 1; 15, 20 -> 2."""
    return sum(snr_db >= edge for edge in edges)


def fit_length(waveform: np.ndarray, num_samples: int) -> np.ndarray:
    if len(waveform) >= num_samples:
        return waveform[:num_samples]
    return np.pad(waveform, (0, num_samples - len(waveform)))


def load_waveform(
    path: Path,
    sample_rate: int,
    num_samples: int | None = None,
    rng: np.random.Generator | None = None,
) -> Tensor:
    waveform = read_audio(path, sample_rate, num_samples, rng)
    if num_samples is not None:
        waveform = fit_length(waveform, num_samples)
    return torch.from_numpy(np.ascontiguousarray(waveform, dtype=np.float32))


def read_audio(
    path: Path,
    sample_rate: int,
    num_samples: int | None = None,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    info = sf.info(path)
    start, frames = 0, -1
    if num_samples is not None:
        frames = round(num_samples * info.samplerate / sample_rate)
        slack = max(info.frames - frames, 0)
        start = int(rng.integers(0, slack + 1)) if rng is not None and slack > 0 else 0
    data, source_rate = sf.read(path, start=start, frames=frames, dtype="float32", always_2d=True)
    waveform = data.mean(axis=1)
    if source_rate != sample_rate:
        divisor = gcd(source_rate, sample_rate)
        waveform = resample_poly(waveform, sample_rate // divisor, source_rate // divisor)
    return waveform


def mix_at_snr(
    speech: np.ndarray, noise: np.ndarray, snr_db: float, speech_length: int | None = None
) -> tuple[np.ndarray, np.ndarray]:
    speech_power = float(np.mean(speech[:speech_length] ** 2)) + 1e-8
    noise_power = float(np.mean(noise**2)) + 1e-8
    noise = noise * np.sqrt(speech_power / (noise_power * 10 ** (snr_db / 10)))
    mixture = speech + noise
    peak = float(np.max(np.abs(mixture)))
    if peak > 1.0:
        mixture, noise = mixture / peak, noise / peak
    return mixture.astype(np.float32), noise.astype(np.float32)


class AudioDataset(Dataset, ABC):
    dataset_key: str

    def __init__(
        self,
        root: str | Path | None = None,
        split: str = "train",
        sample_rate: int = SAMPLE_RATE,
        duration: float | None = None,
    ) -> None:
        self.root = Path(root) if root else resolve_dataset_root(self.dataset_key)
        self.split = split
        self.sample_rate = sample_rate
        self.num_samples = None if duration is None else int(duration * sample_rate)

    def read(self, path: Path, rng: np.random.Generator | None = None) -> Tensor:
        return load_waveform(path, self.sample_rate, self.num_samples, rng)

    @abstractmethod
    def __len__(self) -> int: ...

    @abstractmethod
    def __getitem__(self, index: int) -> AudioItem: ...


class LabeledAudioDataset(AudioDataset):
    class_names: list[str]

    @property
    def num_classes(self) -> int:
        return len(self.class_names)


class AudioSet40Dataset(LabeledAudioDataset):
    dataset_key = AUDIOSET_KEY

    def __init__(
        self,
        root: str | Path | None = None,
        split: str = "train",
        sample_rate: int = SAMPLE_RATE,
        duration: float | None = 10.0,
        exclude_ytids: set[str] | None = None,
    ) -> None:
        super().__init__(root, split, sample_rate, duration)
        metadata_dir = self.root / "metadata"
        excluded = exclude_ytids or set()

        labels_file = metadata_dir / "40_noise_labels.csv"
        labels = read_csv(labels_file if labels_file.is_file() else LABELS_FALLBACK)
        self.class_names = [row["name"] for row in sorted(labels, key=lambda r: int(r["noise_index"]))]
        self.mid_to_index = {row["mid"]: int(row["noise_index"]) for row in labels}

        rows = [
            row
            for row in read_csv(metadata_dir / "manifest.csv")
            if row["split"] == split and row["status"] == "copied" and row["ytid"] not in excluded
        ]
        samples = [(self.root / "audio" / f"{row['ytid']}.wav", self._multi_hot(row["label_mids"])) for row in rows]
        self.samples = [(path, target) for path, target in samples if path.is_file()]
        if len(self.samples) < len(rows):
            logger.warning("%s: skipped %d rows without audio file", split, len(rows) - len(self.samples))

    def _multi_hot(self, label_mids: str) -> Tensor:
        target = torch.zeros(self.num_classes, dtype=torch.float32)
        for mid in label_mids.split(","):
            index = self.mid_to_index.get(mid.strip())
            if index is not None:
                target[index] = 1.0
        return target

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> AudioItem:
        path, target = self.samples[index]
        return {"waveform": self.read(path), "target": target}


class Mixture36Dataset(LabeledAudioDataset):
    dataset_key = MIXTURE_KEY

    def __init__(
        self,
        root: str | Path | None = None,
        split: str = "train",
        sample_rate: int = SAMPLE_RATE,
        duration: float | None = 4.0,
        load_noise: bool = False,
        load_clean: bool = False,
        snr_db: tuple[float, ...] | None = None,
    ) -> None:
        super().__init__(root, split, sample_rate, duration)
        self.load_noise = load_noise
        self.load_clean = load_clean

        with open(self.root / "labels.txt", encoding="utf-8") as f:
            self.class_names = [line.strip() for line in f if line.strip()]

        rows = read_csv(self.root / split / "manifest.csv")
        if snr_db is not None:
            allowed = set(snr_db)
            rows = [row for row in rows if float(row["target_snr_db"]) in allowed]
        self.rows = rows

    def __len__(self) -> int:
        return len(self.rows)

    def read_mixture(self, index: int) -> Tensor:
        return self.read(self.root / self.rows[index]["mixture_path"])

    def __getitem__(self, index: int) -> AudioItem:
        row = self.rows[index]
        item: AudioItem = {
            "waveform": self.read_mixture(index),
            "target": torch.tensor(json.loads(row["multi_hot_36"]), dtype=torch.float32),
            "snr_db": torch.tensor(float(row["target_snr_db"]), dtype=torch.float32),
            "sample_id": row["sample_id"],
        }
        if self.load_noise:
            item["noise"] = self.read(self.root / row["noise_path"])
        if self.load_clean:
            item["clean"] = self.read(self.root / row["clean_path"])
        return item


class KDMoEMixDataset(Mixture36Dataset):
    """Mixture-36l kèm nhãn, nhóm expert, noise_ytid và logit teacher đọc từ cache."""

    def __init__(
        self,
        root: str | Path | None = None,
        split: str = "train",
        sample_rate: int = SAMPLE_RATE,
        duration: float | None = 4.0,
        teacher_cache: str | Path | None = None,
        load_teacher: bool = True,
        require_teacher: bool = False,
        group_map: str | Path | None = None,
        snr_db: tuple[float, ...] | None = None,
        frcrn_cache: str | Path | None = None,
        snr_band_edges: Sequence[float] | None = None,
        exclude_noise_ytids: str | Path | None = None,
        teacher_hints: str | Path | None = None,
    ) -> None:
        """snr_band_edges: nhóm expert theo khoảng SNR thay cho nhóm lớp, vd [5, 15] -> [-5,5), [5,15), [15,20].

        exclude_noise_ytids: file danh sách noise_ytid (1 dòng/id, '#' là chú thích) cần bỏ, vd các clip
        nhiễu nằm trong AudioSet train mà backbone pretrain có nhãn AudioSet đã thấy.
        """
        super().__init__(root, split, sample_rate, duration, load_noise=True, snr_db=snr_db)
        if exclude_noise_ytids:
            lines = Path(exclude_noise_ytids).read_text(encoding="utf-8").splitlines()
            excluded = {line.strip() for line in lines if line.strip() and not line.startswith("#")}
            kept = [row for row in self.rows if row["noise_ytid"] not in excluded]
            logger.info("%s: dropped %d/%d rows whose noise clip is in %s", split, len(self.rows) - len(kept),
                        len(self.rows), Path(exclude_noise_ytids).name)
            self.rows = kept
        # n_hat = FRCRN(mixture) đã tính sẵn (fp32), thay cho mixture làm "waveform"
        self._frcrn_path = Path(frcrn_cache) if frcrn_cache else None
        self._frcrn: np.ndarray | None = None
        self._frcrn_row: dict[str, int] = {}
        if self._frcrn_path is not None:
            ids = json.loads(self._frcrn_path.with_suffix(".json").read_text(encoding="utf-8"))
            self._frcrn_row = {sample_id: i for i, sample_id in enumerate(ids)}
        # SNR ước lượng từ năng lượng FRCRN (frcrn_{split}.snr.npy, cùng thứ tự .json) -> item["snr_est"]
        snr_path = self._frcrn_path.with_suffix(".snr.npy") if self._frcrn_path is not None else None
        self._frcrn_snr = np.load(snr_path) if snr_path is not None and snr_path.is_file() else None
        group_map_path = Path(group_map) if group_map else GROUP_MAP_FALLBACK
        with open(group_map_path, encoding="utf-8") as f:
            mid_to_group = json.load(f)["mid_to_group"]
        self._labels: list[int] = []
        self._groups: list[int] = []
        for row in self.rows:
            mid = row["label_mids"].split(",")[0].strip()
            if mid not in mid_to_group:
                raise KeyError(f"mid {mid!r} of sample {row['sample_id']} is not in {group_map_path}")
            self._labels.append(int(np.argmax(json.loads(row["multi_hot_36"]))))
            if snr_band_edges is None:
                self._groups.append(int(mid_to_group[mid]))
            else:
                self._groups.append(snr_band(float(row["target_snr_db"]), snr_band_edges))
        # hint FitNets của teacher: .npy (rows, T, F, C) fp16 + .ids.json cùng thứ tự, đọc memmap trong từng worker
        self._hints_path = Path(teacher_hints) if teacher_hints else None
        self._hints: np.ndarray | None = None
        self._hint_row: dict[str, int] = {}
        if self._hints_path is not None:
            ids = json.loads(self._hints_path.with_suffix(".ids.json").read_text(encoding="utf-8"))
            self._hint_row = {sample_id: i for i, sample_id in enumerate(ids)}
            missing = [row["sample_id"] for row in self.rows if row["sample_id"] not in self._hint_row]
            if missing:
                raise KeyError(f"{split}: {len(missing)} rows have no teacher hint, e.g. {missing[0]}")
        self._load_teacher = load_teacher
        self._teacher_at: dict[str, np.ndarray] | None = None
        self._teacher_logits = self._load_teacher_cache(teacher_cache, split, require_teacher) if load_teacher else None

    def _load_teacher_cache(
        self, teacher_cache: str | Path | None, split: str, require_teacher: bool
    ) -> dict[str, np.ndarray] | None:
        path = Path(teacher_cache) if teacher_cache else self.root / TEACHER_CACHE_DIR / f"teacher_{split}.npz"
        if not path.is_file():
            if require_teacher:
                raise FileNotFoundError(f"{split}: teacher cache not found: {path}")
            logger.warning("%s: teacher cache %s not found, teacher_logits36 filled with zeros", split, path)
            return None
        with np.load(path) as data:
            if "at_maps" in data.files:  # bản đồ Attention Transfer của teacher, (layer, T, F) mỗi mix
                self._teacher_at = dict(zip(data["mix_id"].tolist(), data["at_maps"], strict=True))
            return dict(zip(data["mix_id"].tolist(), data["logits36"], strict=True))

    def read_mixture(self, index: int) -> Tensor:
        if self._frcrn_path is None:
            return super().read_mixture(index)
        if self._frcrn is None:  # mở memmap trong từng worker
            self._frcrn = np.load(self._frcrn_path, mmap_mode="r")
        return torch.from_numpy(np.array(self._frcrn[self._frcrn_row[self.rows[index]["sample_id"]]]))

    @property
    def labels(self) -> list[int]:
        return self._labels

    @property
    def groups(self) -> list[int]:
        return self._groups

    def __getitem__(self, index: int) -> AudioItem:
        item = super().__getitem__(index)
        item["label"] = torch.tensor(self._labels[index], dtype=torch.long)
        item["group"] = torch.tensor(self._groups[index], dtype=torch.long)
        item["noise_ytid"] = self.rows[index]["noise_ytid"]
        if self._frcrn_snr is not None:
            item["snr_est"] = torch.tensor(float(self._frcrn_snr[self._frcrn_row[item["sample_id"]]]))
        if self._load_teacher:
            logits = self._teacher_logits.get(item["sample_id"]) if self._teacher_logits else None
            if logits is None:
                item["teacher_logits36"] = torch.zeros(len(self.class_names))
            else:
                item["teacher_logits36"] = torch.from_numpy(np.asarray(logits, dtype=np.float32))
            if self._teacher_at is not None:
                item["teacher_at"] = torch.from_numpy(np.asarray(self._teacher_at[item["sample_id"]], dtype=np.float32))
        if self._hints_path is not None:
            if self._hints is None:
                self._hints = np.load(self._hints_path, mmap_mode="r")
            item["teacher_hint"] = torch.from_numpy(np.array(self._hints[self._hint_row[item["sample_id"]]]))
        return item


class DynamicMixDataset(AudioDataset):
    dataset_key = MIXTURE_KEY

    def __init__(
        self,
        root: str | Path | None = None,
        split: str = "train",
        noise_dataset: AudioSet40Dataset | None = None,
        snr_levels: Sequence[float] = ALL_SNR,
        sample_rate: int = SAMPLE_RATE,
        duration: float = 2.0,
        deterministic: bool = False,
        seed: int = 0,
    ) -> None:
        super().__init__(root, split, sample_rate, duration)
        self.speech_paths = self.list_speech(split)
        self.noise_dataset = noise_dataset or AudioSet40Dataset(split=split, sample_rate=sample_rate, duration=None)
        self.snr_levels = tuple(snr_levels)
        self.deterministic = deterministic
        self.seed = seed

    def list_speech(self, split: str) -> list[Path]:
        rows = read_csv(self.root / split / "manifest.csv")
        return sorted({self.root / row["clean_path"] for row in rows})

    def __len__(self) -> int:
        return len(self.speech_paths)

    def _rng(self, index: int) -> np.random.Generator:
        return np.random.default_rng(self.seed + index if self.deterministic else None)

    def __getitem__(self, index: int) -> AudioItem:
        rng = self._rng(index)
        noise_path, _ = self.noise_dataset.samples[int(rng.integers(0, len(self.noise_dataset)))]
        speech = read_audio(self.speech_paths[index], self.sample_rate, self.num_samples, rng)
        speech_length = len(speech)
        speech = fit_length(speech, self.num_samples)
        noise = self.read(noise_path, rng).numpy()
        snr_db = float(rng.choice(self.snr_levels))
        mixture, noise = mix_at_snr(speech, noise, snr_db, speech_length)
        return {
            "waveform": torch.from_numpy(mixture),
            "noise": torch.from_numpy(noise),
            "snr_db": torch.tensor(snr_db, dtype=torch.float32),
        }


def read_split_ratios(path: Path = VLSP_CONFIG) -> dict[str, float]:
    with open(path, encoding="utf-8") as f:
        ratios = {name: float(value) for name, value in yaml.safe_load(f)["split"].items()}
    if abs(sum(ratios.values()) - 1.0) > 1e-6:
        raise ValueError(f"split ratios in {path} must sum to 1, got {sum(ratios.values())}")
    return ratios


def split_of(relative_path: str, ratios: Mapping[str, float]) -> str:
    position = zlib.crc32(relative_path.encode("utf-8")) / 2**32
    upper = 0.0
    for split, ratio in ratios.items():
        upper += ratio
        if position < upper:
            return split
    return next(reversed(ratios))


@cache
def list_audio_files(root: Path) -> tuple[Path, ...]:
    files = tuple(sorted(path for path in root.rglob("*") if path.suffix.lower() in AUDIO_EXTENSIONS))
    if not files:
        raise FileNotFoundError(f"No {AUDIO_EXTENSIONS} files under {root}")
    return files


def vlsp_files(root: Path, split: str, split_ratios: Mapping[str, float] | None = None) -> list[Path]:
    ratios = dict(split_ratios) if split_ratios else read_split_ratios()
    if split not in ratios:
        raise ValueError(f"Unknown split '{split}', expected one of {sorted(ratios)}")
    return [path for path in list_audio_files(root) if split_of(path.relative_to(root).as_posix(), ratios) == split]


class VLSPDataset(AudioDataset):
    dataset_key = VLSP_KEY

    def __init__(
        self,
        root: str | Path | None = None,
        split: str = "train",
        sample_rate: int = SAMPLE_RATE,
        duration: float | None = None,
        split_ratios: Mapping[str, float] | None = None,
    ) -> None:
        super().__init__(root, split, sample_rate, duration)
        self.paths = vlsp_files(self.root, split, split_ratios)

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int) -> AudioItem:
        path = self.paths[index]
        return {"waveform": self.read(path), "sample_id": path.relative_to(self.root).as_posix()}


class VLSPMixDataset(DynamicMixDataset):
    dataset_key = VLSP_KEY

    def __init__(self, *args: Any, split_ratios: Mapping[str, float] | None = None, **kwargs: Any) -> None:
        self.split_ratios = split_ratios
        super().__init__(*args, **kwargs)

    def list_speech(self, split: str) -> list[Path]:
        return vlsp_files(self.root, split, self.split_ratios)


class PairedClipSampler(Sampler[list[int]]):
    """Batch = 32 đơn vị clip nhiễu; clip có >= 2 mix góp thành cặp, clip 1 mix đi đơn.

    Mỗi epoch: chia mix của từng clip thành các cặp rời nhau (còn lại đi đơn),
    xáo theo nhãn rồi xen kẽ các nhãn để batch cân bằng nhãn.
    """

    def __init__(self, dataset: KDMoEMixDataset, units_per_batch: int = 32, seed: int = 0) -> None:
        if units_per_batch < 1:
            raise ValueError(f"units_per_batch must be positive, got {units_per_batch}")
        self.units_per_batch = units_per_batch
        self.seed = seed
        self.epoch = 0
        if hasattr(dataset, "indices"):
            base = dataset.dataset
            rows = [base.rows[index] for index in dataset.indices]
            labels = [base.labels[index] for index in dataset.indices]
        else:
            rows = list(dataset.rows)
            labels = list(dataset.labels)
        self._clips: dict[str, list[int]] = {}
        self._clip_label: dict[str, int] = {}
        for position, row in enumerate(rows):
            self._clips.setdefault(row["noise_ytid"], []).append(position)
        for ytid, positions in self._clips.items():
            self._clip_label[ytid] = labels[positions[0]]
        self._num_units = sum(len(indices) // 2 + len(indices) % 2 for indices in self._clips.values())

    def _units(self, rng: np.random.Generator) -> list[tuple[int, ...]]:
        per_label: dict[int, list[tuple[int, ...]]] = {}
        for ytid in sorted(self._clips):
            indices = list(self._clips[ytid])
            rng.shuffle(indices)
            units = per_label.setdefault(self._clip_label[ytid], [])
            for start in range(0, len(indices) - 1, 2):
                units.append((indices[start], indices[start + 1]))
            if len(indices) % 2:
                units.append((indices[-1],))
        for units in per_label.values():
            rng.shuffle(units)
        ordered: list[tuple[int, ...]] = []
        streams = [iter(units) for units in per_label.values()]
        while streams:
            for stream in list(streams):
                unit = next(stream, None)
                if unit is None:
                    streams.remove(stream)
                else:
                    ordered.append(unit)
        return ordered

    def __iter__(self) -> Iterator[list[int]]:
        rng = np.random.default_rng(self.seed + self.epoch)
        units = self._units(rng)
        self.epoch += 1
        for start in range(0, len(units), self.units_per_batch):
            indices = [index for unit in units[start : start + self.units_per_batch] for index in unit]
            if indices:
                yield indices

    def __len__(self) -> int:
        return ceil(self._num_units / self.units_per_batch)


def build_dataset(name: str, split: str, **kwargs: Any) -> AudioDataset:
    builders: dict[str, type[AudioDataset]] = {
        AUDIOSET_KEY: AudioSet40Dataset,
        MIXTURE_KEY: Mixture36Dataset,
        VLSP_KEY: VLSPDataset,
    }
    if name not in builders:
        raise ValueError(f"Unknown dataset '{name}', expected one of {sorted(builders)}")
    return builders[name](split=split, **kwargs)


def build_dataloader(
    dataset: Dataset,
    batch_size: int | None,
    shuffle: bool = False,
    num_workers: int = 0,
    batch_sampler: Sampler[list[int]] | None = None,
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=1 if batch_sampler is not None else batch_size,
        shuffle=False if batch_sampler is not None else shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=shuffle and batch_sampler is None,
        persistent_workers=num_workers > 0,
        batch_sampler=batch_sampler,
    )


def summarize(name: str, splits: tuple[str, ...]) -> None:
    for split in splits:
        dataset = build_dataset(name, split)
        shapes = {k: tuple(v.shape) for k, v in dataset[0].items() if isinstance(v, Tensor)}
        classes = dataset.num_classes if isinstance(dataset, LabeledAudioDataset) else 0
        logger.info("%s/%s: %d samples, %d classes, %s", name, split, len(dataset), classes, shapes)


def read_noise_ytids(root: str | Path | None = None, splits: Sequence[str] = ("validation", "test")) -> set[str]:
    root = Path(root) if root else resolve_dataset_root(MIXTURE_KEY)
    return {row["noise_ytid"] for split in splits for row in read_csv(root / split / "manifest.csv")}


def subsample(dataset: Dataset, max_samples: int | None) -> Dataset:
    if max_samples is None or max_samples >= len(dataset):
        return dataset
    indices = np.linspace(0, len(dataset) - 1, max_samples).round().astype(int)
    return Subset(dataset, indices.tolist())

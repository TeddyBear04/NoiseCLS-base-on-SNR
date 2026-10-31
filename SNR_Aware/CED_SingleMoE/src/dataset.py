"""Manifest, bộ đệm n_hat (memmap) và logit teacher (nhiễu sạch) cho mix-dataset-v2.

mix-dataset-v2 chỉ có mixture (không có nhiễu sạch). Student nghe n_hat = FRCRN(mixture) đọc từ
bộ đệm .npy [N, 64000]; hàng thứ i của bộ đệm ứng với hàng thứ i của manifest trong split đó.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import TypedDict

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset

SAMPLE_RATE = 16000
NUM_CLASSES = 36
# Thư mục/giá trị cột split của v2 gọi tập validation là "eval"; hai tên dùng chung một split.
SPLIT_ALIASES = {"validation": ("validation", "eval"), "eval": ("validation", "eval")}


class AudioItem(TypedDict, total=False):
    waveform: Tensor
    target: Tensor
    target_idx: Tensor
    snr_db: Tensor
    sample_id: str
    teacher_logits: Tensor  # [K, C]: logit teacher của từng expert SNR
    snr_est: Tensor  # SNR ước lượng từ FRCRN (dB), cho routing.source = frcrn_snr


def read_csv(path: Path) -> list[dict[str, str]]:
    with open(path, encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def read_manifest(root: Path, split: str) -> list[dict[str, str]]:
    """Hàng manifest của `split`: manifest.csv gộp ở gốc (lọc theo cột split) hoặc <split>/manifest.csv."""
    names = SPLIT_ALIASES.get(split, (split,))
    merged = root / "manifest.csv"
    if merged.is_file():
        rows = read_csv(merged)
        if rows and "split" in rows[0]:
            return [row for row in rows if row["split"] in names]
    for name in names:
        path = root / name / "manifest.csv"
        if path.is_file():
            return read_csv(path)
    raise FileNotFoundError(f"không thấy manifest của split '{split}' dưới {root}")


class Mixture36Dataset(Dataset):
    """Nhãn, SNR và sample_id của một split; waveform do `NHatDataset` gắn vào."""

    def __init__(self, root: str | Path, split: str = "train", max_samples: int | None = None) -> None:
        self.root = Path(root)
        self.split = split
        labels = self.root / "labels.txt"
        if labels.is_file():
            self.class_names = [line.strip() for line in labels.read_text(encoding="utf-8").splitlines() if line.strip()]
        else:
            self.class_names = [f"class_{i}" for i in range(NUM_CLASSES)]
        self.rows = read_manifest(self.root, split)
        if max_samples is not None:
            self.rows = self.rows[:max_samples]

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> AudioItem:
        row = self.rows[index]
        target = torch.tensor(json.loads(row["multi_hot_36"]), dtype=torch.float32)
        snr = float(row.get("target_snr_db") or row.get("achieved_snr_db") or 0.0)
        return {
            "target": target,
            "target_idx": target.argmax(dim=-1),
            "snr_db": torch.tensor(snr, dtype=torch.float32),
            "sample_id": row["sample_id"],
        }


def cache_rows(path: Path, rows: list[dict[str, str]]) -> tuple[np.ndarray, list[int]]:
    """Memmap của bộ đệm n_hat và chỉ số hàng ứng với từng hàng manifest.

    Có <tên>.json (danh sách sample_id, như cache FRCRN của FRCRN_KD_MoE) thì ghép theo sample_id, nên cache
    được phép chứa nhiều clip hơn manifest (vd. val 6480 clip của 36_labels, v2 chỉ dùng 3064). Không có .json
    thì hàng i của cache phải là hàng i của manifest."""
    if not path.is_file():
        raise FileNotFoundError(f"thiếu bộ đệm: {path}")
    array = np.load(path, mmap_mode="r")
    ids_path = path.with_suffix(".json")
    if ids_path.is_file():
        index = {sample_id: i for i, sample_id in enumerate(json.loads(ids_path.read_text(encoding="utf-8")))}
        missing = [row["sample_id"] for row in rows if row["sample_id"] not in index]
        if missing:
            raise KeyError(f"{ids_path.name} thiếu {len(missing)} sample_id, vd {missing[:3]}")
        return array, [index[row["sample_id"]] for row in rows]
    if len(array) != len(rows):
        raise ValueError(f"{path.name}: {len(array)} hàng != {len(rows)} hàng manifest và không có {ids_path.name}")
    return array, list(range(len(rows)))


def load_teacher_logits(path: str | Path, split: str, rows: list[dict[str, str]]) -> np.ndarray:
    """Logit teacher 5 fold (nghe nhiễu sạch, out-of-fold) theo thứ tự manifest.

    `path` là .pt đóng gói {split: {mix_id, logits36}} (beats_teacher_logits_clean.pt); ghép theo sample_id."""
    packed = torch.load(path, map_location="cpu", weights_only=False)[split]
    index = {sample_id: i for i, sample_id in enumerate(packed["mix_id"])}
    missing = [row["sample_id"] for row in rows if row["sample_id"] not in index]
    if missing:
        raise KeyError(f"teacher thiếu {len(missing)} sample_id của split {split}, vd {missing[:3]}")
    return np.asarray(packed["logits36"], dtype=np.float32)[[index[row["sample_id"]] for row in rows]]


class NHatDataset(Dataset):
    """n_hat từ bộ đệm, kèm logit teacher của từng expert SNR nếu có.

    `teachers` là danh sách K mảng [N, C], một mảng mỗi expert (low, [mid,] high): một teacher 5 fold nhiễu
    sạch thì cả K là cùng một mảng; Fold0/EMA thì mỗi expert trỏ tới teacher của nó."""

    def __init__(
        self,
        base: Mixture36Dataset,
        nhat_path: str | Path,
        teachers: list[np.ndarray] | None = None,
        with_snr_est: bool = False,
    ) -> None:
        self.base = base
        nhat_path = Path(nhat_path)
        self.array, self.cache_index = cache_rows(nhat_path, base.rows)
        self.teachers = teachers
        # SNR ước lượng 10 log10(P(mixture - n_hat) / P(n_hat)) ở mức gốc (scripts/frcrn_cache.py của FRCRN_KD_MoE),
        # file <cache>.snr.npy cùng thứ tự hàng với bộ đệm n_hat.
        self.snr_est = None
        if with_snr_est:
            snr_path = nhat_path.with_suffix(".snr.npy")
            if not snr_path.is_file():
                raise FileNotFoundError(f"thiếu SNR ước lượng từ FRCRN: {snr_path}")
            self.snr_est = np.load(snr_path)
        for k, logits in enumerate(teachers or []):
            if len(logits) != len(base):
                raise ValueError(f"logit teacher expert {k}: {len(logits)} hàng != {len(base)} hàng manifest")

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, index: int) -> AudioItem:
        item = self.base[index]
        item["waveform"] = torch.from_numpy(self.array[self.cache_index[index]].copy()).float()
        if self.snr_est is not None:
            item["snr_est"] = torch.tensor(float(self.snr_est[self.cache_index[index]]), dtype=torch.float32)
        if self.teachers is not None:
            item["teacher_logits"] = torch.from_numpy(np.stack([np.asarray(t[index], dtype=np.float32) for t in self.teachers]))
        return item


def build_dataloader(
    dataset: Dataset, batch_size: int, shuffle: bool = False, num_workers: int = 0, pin_memory: bool = True
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory and torch.cuda.is_available(),
        drop_last=False,
    )

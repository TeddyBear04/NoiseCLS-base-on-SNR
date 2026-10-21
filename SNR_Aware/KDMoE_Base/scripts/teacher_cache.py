from __future__ import annotations

import logging
import sys
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.optim import AdamW
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.build_group_map import load_membership  # noqa: E402
from src.loss import si_snr  # noqa: E402
from src.metrics import ClassificationMetrics  # noqa: E402
from src.model.beat import BEATClassifier  # noqa: E402
from src.model.frcrn import FRCRNSeparator  # noqa: E402

logger = logging.getLogger(__name__)

NUM_CLASSES = 36
PROBE_LR = 1e-3
PROBE_EPOCHS = 100
PROBE_WEIGHT_DECAY = 1e-2
PROBE_BATCH_SIZE = 256
PROBE_SPLITS = 5
RMS_EPS = 1e-8


def rms(waveform: Tensor) -> Tensor:
    return waveform.square().mean(dim=-1).sqrt()


def median_rms(dataset: Any, max_clips: int | None = None) -> float:
    values = []
    for index in range(len(dataset)):
        waveform = dataset[index]["waveform"]
        values.append(float(rms(waveform.unsqueeze(0))))
        if max_clips is not None and len(values) >= max_clips:
            break
    if not values:
        raise ValueError("dataset is empty, cannot compute median RMS")
    return float(np.median(values))


def rms_normalize(waveform: Tensor, target_rms: float) -> Tensor:
    scale = target_rms / (rms(waveform) + RMS_EPS)
    return waveform * scale[:, None]


def beats_embedding(classifier: BEATClassifier, waveform: Tensor) -> Tensor:
    return classifier(waveform).features.mean(dim=1)


@torch.no_grad()
def frcrn_sisdr_by_snr(
    separator: FRCRNSeparator,
    loader: DataLoader,
    device: torch.device,
    speech_as_noise: bool = False,
) -> dict[str, float]:
    separator.eval()
    sums: defaultdict[str, float] = defaultdict(float)
    counts: defaultdict[str, int] = defaultdict(int)
    for batch in loader:
        mixture = batch["waveform"].to(device)
        noise = batch["noise"].to(device)
        estimate = separator(mixture)
        if speech_as_noise:
            estimate = mixture - estimate
        values = si_snr(noise, estimate)
        for snr, value in zip(batch["snr_db"].tolist(), values.tolist(), strict=True):
            key = f"sisdr_snr_{int(snr)}"
            sums[key] += value
            counts[key] += 1
    return {key: sums[key] / counts[key] for key in sorted(sums)}


@torch.no_grad()
def teacher_embeddings(
    separator: FRCRNSeparator | None,
    classifier: BEATClassifier,
    loader: DataLoader,
    device: torch.device,
    target_rms: float | None,
    speech_as_noise: bool = False,
    input_key: str = "waveform",
) -> tuple[np.ndarray, list[str]]:
    if separator is not None:
        separator.eval()
    classifier.eval()
    embeddings: list[np.ndarray] = []
    sample_ids: list[str] = []
    for batch in loader:
        source = batch[input_key].to(device)
        estimate = separator(source) if separator is not None else source
        if speech_as_noise:
            if separator is None:
                raise ValueError("speech_as_noise requires a separator")
            estimate = source - estimate
        if target_rms is not None:
            estimate = rms_normalize(estimate, target_rms)
        embedding = beats_embedding(classifier, estimate)
        embeddings.append(embedding.half().cpu().numpy())
        sample_ids.extend(batch["sample_id"])
    return np.concatenate(embeddings), sample_ids


@dataclass
class TeacherProbe:
    """Linear probe 36 lớp cross-fit: giữ trọng số 5 fold để dự đoán val/test."""

    folds: list[dict[str, Any]] = field(default_factory=list)

    def add_fold(self, weight: Tensor, bias: Tensor, mean: np.ndarray, std: np.ndarray) -> None:
        self.folds.append(
            {
                "weight": weight.detach().cpu().numpy(),
                "bias": bias.detach().cpu().numpy(),
                "mean": mean,
                "std": std,
            }
        )

    def logits(self, embeddings: np.ndarray) -> np.ndarray:
        if not self.folds:
            raise ValueError("no fold trained, call cross_fit_probe first")
        values = [
            ((embeddings - fold["mean"]) / (fold["std"] + 1e-5)) @ fold["weight"].T + fold["bias"]
            for fold in self.folds
        ]
        return np.mean(values, axis=0)


def _fit_linear(
    features: np.ndarray,
    labels: np.ndarray,
    epochs: int,
    lr: float,
    weight_decay: float,
    batch_size: int,
    seed: int,
) -> tuple[Tensor, Tensor]:
    torch.manual_seed(seed)
    linear = nn.Linear(features.shape[1], NUM_CLASSES)
    optimizer = AdamW(linear.parameters(), lr=lr, weight_decay=weight_decay)
    x = torch.from_numpy(features.astype(np.float32))
    y = torch.from_numpy(labels.astype(np.int64))
    for _ in range(epochs):
        order = torch.randperm(len(x))
        for start in range(0, len(x), batch_size):
            index = order[start : start + batch_size]
            loss = F.cross_entropy(linear(x[index]), y[index])
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
    return linear.weight.detach(), linear.bias.detach()


def cross_fit_probe(
    embeddings: np.ndarray,
    labels: np.ndarray,
    groups: np.ndarray,
    epochs: int = PROBE_EPOCHS,
    lr: float = PROBE_LR,
    weight_decay: float = PROBE_WEIGHT_DECAY,
    batch_size: int = PROBE_BATCH_SIZE,
    n_splits: int = PROBE_SPLITS,
    seed: int = 0,
) -> tuple[np.ndarray, TeacherProbe]:
    from sklearn.model_selection import GroupKFold, StratifiedGroupKFold

    splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    try:
        splits = list(splitter.split(embeddings, labels, groups))
    except ValueError as error:
        logger.warning("StratifiedGroupKFold failed (%s); falling back to GroupKFold", error)
        splits = list(GroupKFold(n_splits=n_splits).split(embeddings, labels, groups))
    out_of_fold = np.zeros((len(embeddings), NUM_CLASSES), dtype=np.float32)
    probe = TeacherProbe()
    for fold, (train_index, hold_index) in enumerate(splits):
        train_features = embeddings[train_index]
        mean = train_features.mean(axis=0)
        std = train_features.std(axis=0)
        standardized = (train_features - mean) / (std + 1e-5)
        weight, bias = _fit_linear(standardized, labels[train_index], epochs, lr, weight_decay, batch_size, seed + fold)
        hold = (embeddings[hold_index] - mean) / (std + 1e-5)
        out_of_fold[hold_index] = (hold @ weight.numpy().T + bias.numpy()).astype(np.float32)
        probe.add_fold(weight, bias, mean, std)
        logger.info(
            "probe fold %d/%d done: %d train, %d held out", fold + 1, n_splits, len(train_index), len(hold_index)
        )
    return out_of_fold, probe


def save_teacher_cache(
    path: str | Path,
    sample_ids: Sequence[str],
    logits36: np.ndarray,
    snr_db: np.ndarray,
) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        path,
        mix_id=np.asarray(sample_ids),
        logits36=logits36.astype(np.float16),
        snr_db=np.asarray(snr_db, dtype=np.float32),
    )
    return path


def load_teacher_cache(path: str | Path) -> dict[str, np.ndarray]:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"teacher cache not found: {path}")
    with np.load(path) as data:
        return {name: data[name] for name in data.files}


def accuracy_by_snr(
    logits: np.ndarray,
    labels: np.ndarray,
    groups: np.ndarray,
    snr_db: np.ndarray,
    membership: np.ndarray | None = None,
) -> dict[str, float]:
    if membership is None:
        membership = load_membership()
    predictions = logits.argmax(axis=1)
    probs = np.exp(logits.astype(np.float32) - logits.max(axis=1, keepdims=True))
    probs /= probs.sum(axis=1, keepdims=True)
    group_predictions = (probs @ membership).argmax(axis=1)
    overall = ClassificationMetrics(num_classes=NUM_CLASSES)
    overall.update(labels, probs)
    report: dict[str, float] = dict(overall.compute().to_dict())
    report["acc"] = float((predictions == labels).mean())
    report["router_acc"] = float((group_predictions == groups).mean())
    for snr in np.unique(snr_db):
        mask = snr_db == snr
        report[f"acc_snr_{int(snr)}"] = float((predictions[mask] == labels[mask]).mean())
        report[f"router_acc_snr_{int(snr)}"] = float((group_predictions[mask] == groups[mask]).mean())
    return report


def snr_kd_weights(snr_accuracy: Mapping[str, float]) -> dict[str, float]:
    best = max(snr_accuracy.values())
    if best <= 0.0:
        raise ValueError("snr accuracy must be positive")
    return {snr: value / best for snr, value in snr_accuracy.items()}

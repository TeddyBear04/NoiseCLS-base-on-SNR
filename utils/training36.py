"""Helpers shared by the separator, fusion-head, fine-tuning and test stages."""

from __future__ import annotations

import random
from collections import defaultdict

import numpy as np
import torch
from sklearn.metrics import f1_score, precision_recall_fscore_support

from config.paths import BEATS_CHECKPOINT
from models.beats_loader import load_beats_classes

SNRS = (-5, 0, 5, 10, 15, 20)
SEED = 2026


def seed_everything(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def mixed_precision_context(device: torch.device):
    """Use BF16 when available; fall back to numerically stable FP32."""
    return torch.autocast(
        device_type=device.type,
        dtype=torch.bfloat16,
        enabled=device.type == "cuda" and torch.cuda.is_bf16_supported(),
    )


def require_finite(value: torch.Tensor, name: str) -> None:
    """Fail immediately instead of saving a checkpoint with NaN weights."""
    if not torch.isfinite(value).all():
        raise FloatingPointError(f"Non-finite {name}; aborting training.")


def balanced_rows(
    rows: list[dict[str, str]], split: str, per_class: int, seed: int = SEED
) -> list[dict[str, str]]:
    groups: dict[tuple[str, int], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        if row["split"] == split:
            groups[(row["label_names"], int(float(row["target_snr_db"])))].append(row)
    labels = sorted({label for label, _ in groups})
    if per_class % len(SNRS):
        raise ValueError(f"per_class must be divisible by {len(SNRS)}")
    per_group = per_class // len(SNRS)
    rng = random.Random(seed + (0 if split == "train" else 10_000))
    selected = []
    for label in labels:
        for snr in SNRS:
            candidates = groups[(label, snr)].copy()
            rng.shuffle(candidates)
            selected.extend(candidates[:per_group])
    rng.shuffle(selected)
    return selected


def select_rows(rows: list[dict[str, str]], split: str, per_class: int | None):
    return rows if per_class is None else balanced_rows(rows, split, per_class)


def load_beats(device: torch.device):
    """Load the AudioSet fine-tuned BEATs encoder frozen and in eval mode."""
    if not BEATS_CHECKPOINT.exists():
        raise FileNotFoundError(BEATS_CHECKPOINT)
    BEATs, BEATsConfig = load_beats_classes()
    checkpoint = torch.load(BEATS_CHECKPOINT, map_location="cpu", weights_only=True)
    model = BEATs(BEATsConfig(checkpoint["cfg"]))
    model.load_state_dict(checkpoint["model"])
    model.predictor = None
    model.to(device).eval()
    for parameter in model.parameters():
        parameter.requires_grad = False
    return model, checkpoint


def metrics(logits: torch.Tensor, targets: torch.Tensor, snrs: torch.Tensor, labels):
    predictions = logits.argmax(dim=1).cpu().numpy()
    expected = targets.cpu().numpy()
    precision, recall, class_f1, support = precision_recall_fscore_support(
        expected, predictions, labels=np.arange(len(labels)), zero_division=0
    )
    result = {
        "accuracy": float((predictions == expected).mean()),
        "precision": float(precision.mean()),
        "recall": float(recall.mean()),
        "macro_f1": float(class_f1.mean()),
        "micro_f1": float(f1_score(expected, predictions, average="micro", zero_division=0)),
        "per_class": {
            label: {"f1": float(class_f1[index]), "support": int(support[index])}
            for index, label in enumerate(labels)
        },
        "per_snr": {},
    }
    snr_values = snrs.cpu().numpy()
    for snr in SNRS:
        mask = snr_values == snr
        if not mask.any():
            continue
        _, _, snr_f1, _ = precision_recall_fscore_support(
            expected[mask], predictions[mask],
            labels=np.arange(len(labels)), zero_division=0,
        )
        result["per_snr"][str(snr)] = {
            "samples": int(mask.sum()),
            "accuracy": float((predictions[mask] == expected[mask]).mean()),
            "macro_f1": float(snr_f1.mean()),
            "micro_f1": float(f1_score(expected[mask], predictions[mask], average="micro", zero_division=0)),
        }
    return result


def add_si_sdr_metrics(result: dict, si_sdrs: torch.Tensor, snrs: torch.Tensor) -> dict:
    """Attach mean separator SI-SDR overall and per SNR to a metrics dict."""
    result["si_sdr"] = float(si_sdrs.mean())
    for snr, values in result["per_snr"].items():
        result["per_snr"][snr]["si_sdr"] = float(si_sdrs[snrs == int(snr)].mean())
    return result


def label_to_mid_map(rows: list[dict[str, str]]) -> dict[str, str]:
    mapping: dict[str, str] = {}
    for row in rows:
        mapping.setdefault(row["label_names"], row["label_mids"])
    return mapping


def initialize_head_from_audioset(head: torch.nn.Linear, checkpoint, labels, label_to_mid) -> None:
    """Copy the AudioSet predictor rows of the 36 target classes into ``head``."""
    mid_to_source = {mid: int(index) for index, mid in checkpoint["label_dict"].items()}
    source_indices = [mid_to_source[label_to_mid[label]] for label in labels]
    with torch.no_grad():
        head.weight.copy_(checkpoint["model"]["predictor.weight"][source_indices])
        head.bias.copy_(checkpoint["model"]["predictor.bias"][source_indices])

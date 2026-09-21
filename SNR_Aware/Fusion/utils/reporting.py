"""Metrics and report tables for the fusion model."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    precision_recall_fscore_support,
    precision_score,
    recall_score,
    roc_auc_score,
)

# The four functions below (top_k_accuracy, per_class_metrics, metric_values,
# markdown_table) are copied verbatim from
# SNR_Aware/DPCRN_Noise_Target/tasks/run_36.py rather than imported: the two
# projects are separate trees with no shared package, and a cross-import
# would need sys.path surgery for four small functions.


def top_k_accuracy(targets: np.ndarray, probabilities: np.ndarray, k: int = 3) -> float:
    top = np.argsort(-probabilities, axis=1)[:, :k]
    return float((top == targets[:, None]).any(axis=1).mean())


def per_class_metrics(
    targets: np.ndarray, predictions: np.ndarray, probabilities: np.ndarray, labels: list[str]
) -> dict[str, dict[str, float | int]]:
    one_hot_targets = np.eye(len(labels), dtype=np.int32)[targets]
    precision, recall, f1, support = precision_recall_fscore_support(
        targets, predictions, labels=range(len(labels)), zero_division=0
    )
    rows: dict[str, dict[str, float | int]] = {}
    for index, label in enumerate(labels):
        try:
            ap = float(average_precision_score(one_hot_targets[:, index], probabilities[:, index]))
            auc = float(roc_auc_score(one_hot_targets[:, index], probabilities[:, index]))
        except ValueError:
            ap, auc = float("nan"), float("nan")
        rows[label] = {
            "support": int(support[index]), "precision": float(precision[index]),
            "recall": float(recall[index]), "f1": float(f1[index]), "ap": ap, "auc": auc,
        }
    return rows


def metric_values(
    targets: np.ndarray, predictions: np.ndarray, probabilities: np.ndarray
) -> dict[str, float]:
    one_hot_targets = np.eye(probabilities.shape[1], dtype=np.float32)[targets]
    try:
        macro_auc = float(
            roc_auc_score(one_hot_targets, probabilities, average="macro", multi_class="ovr")
        )
        mean_ap = float(average_precision_score(one_hot_targets, probabilities, average="macro"))
    except ValueError:
        # A debug slice can lack one or more of the 36 classes.
        macro_auc, mean_ap = float("nan"), float("nan")
    return {
        "accuracy": float(accuracy_score(targets, predictions)),
        "top3_accuracy": top_k_accuracy(targets, probabilities),
        "precision": float(precision_score(targets, predictions, average="macro", zero_division=0)),
        "recall": float(recall_score(targets, predictions, average="macro", zero_division=0)),
        "macro_f1": float(f1_score(targets, predictions, average="macro", zero_division=0)),
        "micro_f1": float(f1_score(targets, predictions, average="micro", zero_division=0)),
        "map": mean_ap,
        "balanced_accuracy": float(balanced_accuracy_score(targets, predictions)),
        "macro_auc": macro_auc,
    }


def markdown_table(headers: list[str], rows: list[list[str]]) -> str:
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join("---" for _ in headers) + "|"]
    lines.extend("| " + " | ".join(row) + " |" for row in rows)
    return "\n".join(lines)


SNR_LEVELS = (-5.0, 0.0, 5.0, 10.0, 15.0, 20.0)


def label_by_snr_f1(
    targets: np.ndarray, predictions: np.ndarray, snrs: np.ndarray, labels: list[str]
) -> dict[str, dict[str, float]]:
    """F1 per label at each observed SNR - the two report axes crossed.

    Reporting them separately cannot say whether a weak class fails everywhere
    or only once speech starts to mask it. Iterating the SNR levels actually
    present (rather than the fixed SNR_LEVELS grid) keeps this table's columns
    in agreement with evaluate_fusion's per_snr table: a grid level with zero
    samples is absent from both instead of vanishing from one and lingering
    as NaN in the other, and an off-grid clip is still accounted for here.
    """
    levels = sorted(set(float(snr) for snr in snrs.tolist()))
    table: dict[str, dict[str, float]] = {}
    for index, label in enumerate(labels):
        row: dict[str, float] = {}
        for snr in levels:
            mask = snrs == snr
            row[f"{snr:g}"] = float(f1_score(
                targets[mask] == index, predictions[mask] == index, zero_division=0
            ))
        table[label] = row
    return table


def write_label_by_snr_csv(path: Path, table: dict[str, dict[str, float]]) -> None:
    columns = sorted({column for row in table.values() for column in row}, key=float)
    with Path(path).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["label", *columns])
        for label, row in table.items():
            writer.writerow([label, *(row[column] for column in columns)])


def write_label_by_snr_heatmap(path: Path, table: dict[str, dict[str, float]]) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    columns = sorted({column for row in table.values() for column in row}, key=float)
    labels = list(table)
    matrix = np.array([[table[label][column] for column in columns] for label in labels])
    figure, axes = plt.subplots(figsize=(7, 12))
    image = axes.imshow(matrix, aspect="auto", vmin=0.0, vmax=1.0, cmap="viridis")
    axes.set_xticks(range(len(columns)), columns)
    axes.set_yticks(range(len(labels)), labels, fontsize=7)
    axes.set_xlabel("SNR (dB)")
    figure.colorbar(image, ax=axes, label="F1")
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)


TRANSIENT_LABELS = frozenset({
    "Knock", "Keys jangling", "Finger snapping", "Tearing",
    "Scissors", "Hi-hat", "Computer keyboard", "Gunshot, gunfire",
})
TONAL_LABELS = frozenset({
    "Gong", "Flute", "Electric piano", "Clarinet",
    "Harmonica", "Violin, fiddle", "Acoustic guitar", "Telephone",
})


def band_si_sdr(
    estimate: "torch.Tensor", target: "torch.Tensor",
    low_hz: float, high_hz: float,
    sample_rate: int = 16000, n_fft: int = 512, hop_length: int = 160,
) -> "torch.Tensor":
    """SI-SDR after zeroing every bin outside [low_hz, high_hz).

    The EDA found 6-8 kHz is where noise survives at high SNR, so a
    full-band SI-SDR hides exactly the band that matters there.
    """
    import torch

    window = torch.hann_window(n_fft, device=estimate.device)
    frequencies = torch.fft.rfftfreq(n_fft, 1 / sample_rate).to(estimate.device)
    keep = ((frequencies >= low_hz) & (frequencies < high_hz)).view(1, -1, 1)

    def filtered(signal: "torch.Tensor") -> "torch.Tensor":
        spectrum = torch.stft(
            signal, n_fft, hop_length, window=window, return_complex=True
        )
        return torch.istft(
            spectrum * keep, n_fft, hop_length, window=window, length=signal.shape[-1]
        )

    estimate, target = filtered(estimate), filtered(target)
    estimate = estimate - estimate.mean(dim=-1, keepdim=True)
    target = target - target.mean(dim=-1, keepdim=True)
    scale = (estimate * target).sum(dim=-1, keepdim=True) / (
        target.pow(2).sum(dim=-1, keepdim=True) + 1e-8
    )
    projection = scale * target
    residual = estimate - projection
    return 10 * torch.log10(
        (projection.pow(2).sum(dim=-1) + 1e-8) / (residual.pow(2).sum(dim=-1) + 1e-8)
    )


def group_f1(
    targets: np.ndarray, predictions: np.ndarray, labels: list[str], group: set[str]
) -> float:
    """Mean per-label F1 across one named group of labels."""
    indices = [index for index, label in enumerate(labels) if label in group]
    if not indices:
        raise ValueError("group matched none of the labels")
    scores = f1_score(targets, predictions, labels=indices, average=None, zero_division=0)
    return float(np.mean(scores))

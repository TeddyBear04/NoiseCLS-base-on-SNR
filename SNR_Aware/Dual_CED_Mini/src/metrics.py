"""
Classification metrics and SNR-grouped evaluation for Dual-Expert CED.
Evaluates Top-1, Top-3, Balanced Accuracy, Precision, Recall, Macro F1, Micro F1, mAP, and ROC-AUC.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
from sklearn.metrics import average_precision_score, precision_recall_fscore_support, roc_auc_score
from torch import Tensor


def classification_metrics(logits: Tensor, targets: Tensor, num_classes: int = 36) -> dict[str, float]:
    """Compute top-1/top-3 accuracy, macro/micro classification metrics, and mAP."""
    scores = logits.detach().float().cpu().numpy()
    labels = targets.detach().long().cpu().numpy()
    predicted = scores.argmax(axis=1)

    precision, recall, macro_f1, _ = precision_recall_fscore_support(
        labels, predicted, labels=np.arange(num_classes), average="macro", zero_division=0
    )
    _, _, micro_f1, _ = precision_recall_fscore_support(labels, predicted, average="micro", zero_division=0)

    # One-hot encoding for multi-class mAP and ROC-AUC
    one_hot = np.eye(num_classes, dtype=np.float32)[labels]
    probabilities = torch.from_numpy(scores).softmax(dim=-1).numpy()

    ap_values = [
        average_precision_score(one_hot[:, i], probabilities[:, i])
        for i in range(num_classes)
        if one_hot[:, i].sum() > 0
    ]
    auc_values = [
        roc_auc_score(one_hot[:, i], probabilities[:, i])
        for i in range(num_classes)
        if one_hot[:, i].min() != one_hot[:, i].max()
    ]
    topk = scores.argsort(axis=1)[:, -min(3, num_classes):]

    return {
        "acc": float(np.mean(predicted == labels)),
        "top3": float(np.mean(np.any(topk == labels[:, None], axis=1))),
        "balanced_accuracy": float(recall),
        "precision": float(precision),
        "recall": float(recall),
        "macro_f1": float(macro_f1),
        "micro_f1": float(micro_f1),
        "mAP": float(np.mean(ap_values)) if ap_values else 0.0,
        "macro_auc": float(np.mean(auc_values)) if auc_values else 0.0,
    }


def snr_metrics(
    logits: Tensor, targets: Tensor, snrs: Tensor, num_classes: int = 36
) -> dict[str, dict[str, Any]]:
    """Compute metrics across 6 discrete SNR tiers (-5, 0, 5, 10, 15, 20 dB) and ALL."""
    result: dict[str, dict[str, Any]] = {}
    for level in (-5, 0, 5, 10, 15, 20):
        mask = torch.isclose(snrs, torch.tensor(float(level), device=snrs.device), atol=0.5)
        if mask.any():
            result[str(level)] = {
                "n": int(mask.sum()),
                **classification_metrics(logits[mask], targets[mask], num_classes),
            }
        else:
            result[str(level)] = {
                "n": 0,
                "acc": 0.0,
                "top3": 0.0,
                "balanced_accuracy": 0.0,
                "precision": 0.0,
                "recall": 0.0,
                "macro_f1": 0.0,
                "micro_f1": 0.0,
                "mAP": 0.0,
                "macro_auc": 0.0,
            }

    # Global split metrics
    result["ALL"] = {
        "n": int(len(targets)),
        **classification_metrics(logits, targets, num_classes),
    }
    return result

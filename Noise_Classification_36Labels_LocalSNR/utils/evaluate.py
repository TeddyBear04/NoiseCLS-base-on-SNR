from __future__ import annotations

import logging
from collections import OrderedDict
from typing import Any, Dict, List, Sequence

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    f1_score,
    hamming_loss,
    multilabel_confusion_matrix,
    precision_recall_fscore_support,
    roc_auc_score,
)
from tqdm import tqdm

logger = logging.getLogger(__name__)


def _per_class_average_precision(target: np.ndarray, probability: np.ndarray) -> np.ndarray:
    values = []
    for class_index in range(target.shape[1]):
        y_true = target[:, class_index]
        values.append(
            float("nan")
            if np.unique(y_true).size < 2
            else average_precision_score(y_true, probability[:, class_index])
        )
    return np.asarray(values, dtype=np.float64)


def _per_class_auc(target: np.ndarray, probability: np.ndarray) -> np.ndarray:
    values = []
    for class_index in range(target.shape[1]):
        y_true = target[:, class_index]
        values.append(
            float("nan")
            if np.unique(y_true).size < 2
            else roc_auc_score(y_true, probability[:, class_index])
        )
    return np.asarray(values, dtype=np.float64)


def compute_multilabel_metrics(
    target: np.ndarray,
    probability: np.ndarray,
    threshold: float,
    label_names: Sequence[str],
    include_report: bool = True,
    single_label: bool = False,
) -> Dict[str, Any]:
    """Score clip predictions, either by thresholding or by argmax.

    Every clip in this dataset carries exactly one label, so ``single_label=True``
    decides with argmax. Thresholding at 0.5 makes a model that ranks correctly but
    scores conservatively look far worse than it is, and it lets a predict-nothing
    model score 35/36 on Hamming accuracy.
    """
    target = target.astype(np.int32, copy=False)
    if single_label:
        prediction = np.zeros_like(target, dtype=np.int32)
        prediction[np.arange(len(probability)), probability.argmax(axis=1)] = 1
    else:
        prediction = (probability >= threshold).astype(np.int32)
    top1_accuracy = (
        float(np.mean(probability.argmax(axis=1) == target.argmax(axis=1)))
        if len(target)
        else float("nan")
    )
    average_precision = _per_class_average_precision(target, probability)
    auc = _per_class_auc(target, probability)
    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
        target, prediction, average="macro", zero_division=0
    )
    precision_micro, recall_micro, f1_micro, _ = precision_recall_fscore_support(
        target, prediction, average="micro", zero_division=0
    )
    precision_weighted, recall_weighted, f1_weighted, _ = precision_recall_fscore_support(
        target, prediction, average="weighted", zero_division=0
    )
    # Element-wise correctness per label. Its mean over labels equals
    # hamming_accuracy, so it is kept only for the per-label report.
    per_label_accuracy = (target == prediction).mean(axis=0).astype(np.float64)
    report = ""
    if include_report:
        report = classification_report(
            target,
            prediction,
            target_names=list(label_names),
            digits=4,
            zero_division=0,
        )
    return {
        "average_precision": average_precision,
        "mAP": float(np.nanmean(average_precision)),
        "auc": auc,
        "macro_auc": float(np.nanmean(auc)),
        "subset_accuracy": float(accuracy_score(target, prediction)),
        "top1_accuracy": top1_accuracy,
        "accuracy": float(accuracy_score(target, prediction)),
        "hamming_accuracy": float(1.0 - hamming_loss(target, prediction)),
        "per_label_accuracy": per_label_accuracy,
        "precision_macro": float(precision_macro),
        "recall_macro": float(recall_macro),
        "f1_macro": float(f1_macro),
        "precision_micro": float(precision_micro),
        "recall_micro": float(recall_micro),
        "f1_micro": float(f1_micro),
        "precision_weighted": float(precision_weighted),
        "recall_weighted": float(recall_weighted),
        "f1_weighted": float(f1_weighted),
        # Compatibility keys used by the copied logger.
        "prec_macro": float(precision_macro),
        "rec_macro": float(recall_macro),
        "prec_weighted": float(precision_weighted),
        "rec_weighted": float(recall_weighted),
        "confu_matrix": multilabel_confusion_matrix(target, prediction),
        "message": "\n" + report if report else "",
        "target": target,
        "probability": probability,
        "prediction": prediction,
    }


def compute_local_snr_metrics(
    target_db: np.ndarray,
    prediction_db: np.ndarray,
    mask: np.ndarray,
) -> Dict[str, float | int]:
    """Compute regression metrics only where clean speech is active."""
    valid = mask.astype(bool, copy=False)
    target = target_db[valid].astype(np.float64, copy=False)
    prediction = prediction_db[valid].astype(np.float64, copy=False)
    if target.size == 0:
        return {
            "local_snr_mae_db": float("nan"),
            "local_snr_rmse_db": float("nan"),
            "local_snr_pearson": float("nan"),
            "local_snr_valid_segments": 0,
        }
    error = prediction - target
    pearson = float("nan")
    if target.size > 1 and np.std(target) > 0.0 and np.std(prediction) > 0.0:
        pearson = float(np.corrcoef(target, prediction)[0, 1])
    return {
        "local_snr_mae_db": float(np.mean(np.abs(error))),
        "local_snr_rmse_db": float(np.sqrt(np.mean(np.square(error)))),
        "local_snr_pearson": pearson,
        "local_snr_valid_segments": int(target.size),
    }


def aggregate_windows(
    sample_ids: Sequence[str],
    probabilities: np.ndarray,
    targets: np.ndarray,
    snr_values: np.ndarray,
    reduction: str = "mean",
) -> tuple[List[str], np.ndarray, np.ndarray, np.ndarray]:
    groups: "OrderedDict[str, List[int]]" = OrderedDict()
    for index, sample_id in enumerate(sample_ids):
        groups.setdefault(sample_id, []).append(index)

    clip_probabilities = []
    clip_targets = []
    clip_snrs = []
    for indexes in groups.values():
        window_values = probabilities[indexes]
        if reduction == "max":
            clip_probabilities.append(window_values.max(axis=0))
        elif reduction == "mean":
            clip_probabilities.append(window_values.mean(axis=0))
        else:
            raise ValueError(f"Unsupported window reduction: {reduction}")
        clip_targets.append(targets[indexes[0]])
        clip_snrs.append(snr_values[indexes[0]])
    return (
        list(groups.keys()),
        np.asarray(clip_probabilities, dtype=np.float32),
        np.asarray(clip_targets, dtype=np.float32),
        np.asarray(clip_snrs, dtype=np.float32),
    )


# (name, min_db, max_db), both bounds inclusive. Mirrors the discrete target SNRs
# of the 36-label dataset, so the six bands cover every clip exactly once.
DEFAULT_SNR_BANDS: tuple[tuple[str, float, float], ...] = (
    ("-5dB", -5.0, -5.0),
    ("0dB", 0.0, 0.0),
    ("5dB", 5.0, 5.0),
    ("10dB", 10.0, 10.0),
    ("15dB", 15.0, 15.0),
    ("20dB", 20.0, 20.0),
)


def compute_snr_band_metrics(
    clip_target: np.ndarray,
    clip_probability: np.ndarray,
    clip_snr: np.ndarray,
    threshold: float,
    label_names: Sequence[str],
    snr_bands: Sequence[tuple[str, float, float]] = DEFAULT_SNR_BANDS,
    single_label: bool = False,
) -> Dict[str, Dict[str, Any]]:
    """Break the clip-level metrics down by target SNR band."""
    band_metrics: Dict[str, Dict[str, Any]] = {}
    covered = np.zeros(len(clip_snr), dtype=bool)
    for name, min_db, max_db in snr_bands:
        mask = (clip_snr >= min_db) & (clip_snr <= max_db)
        covered |= mask
        if not mask.any():
            logger.warning("SNR band %s matched no clips; skipping it", name)
            continue
        metrics = compute_multilabel_metrics(
            clip_target[mask],
            clip_probability[mask],
            threshold,
            label_names,
            include_report=False,
            single_label=single_label,
        )
        band_metrics[name] = {
            "samples": int(mask.sum()),
            "snr_min_db": float(min_db),
            "snr_max_db": float(max_db),
            "mAP": metrics["mAP"],
            "macro_auc": metrics["macro_auc"],
            "macro_f1": metrics["f1_macro"],
            "micro_f1": metrics["f1_micro"],
            "precision_macro": metrics["precision_macro"],
            "recall_macro": metrics["recall_macro"],
            "hamming_accuracy": metrics["hamming_accuracy"],
            "subset_accuracy": metrics["subset_accuracy"],
        }

    uncovered = int((~covered).sum())
    if uncovered:
        logger.warning(
            "%d of %d clips (%.1f%%) fall outside every SNR band and are excluded "
            "from the per-SNR report; check snr_bands in train_config.json",
            uncovered,
            len(clip_snr),
            100.0 * uncovered / max(len(clip_snr), 1),
        )
    return band_metrics


class BaseEvaluator:
    def __init__(self, model: nn.Module) -> None:
        self.model = model
        self.device = next(model.parameters()).device

    def evaluate(self, data_loader: Any) -> Dict[str, Any]:
        raise NotImplementedError


class AudioEvaluator(BaseEvaluator):
    def __init__(
        self,
        model: nn.Module,
        label_names: Sequence[str],
        threshold: float = 0.5,
        loss_fn: nn.Module | None = None,
        window_reduction: str = "mean",
        snr_bands: Sequence[tuple[str, float, float]] | None = None,
        single_label: bool = False,
    ) -> None:
        super().__init__(model)
        self.label_names = list(label_names)
        self.threshold = threshold
        self.single_label = single_label
        self.loss_fn = loss_fn
        self.window_reduction = window_reduction
        self.snr_bands = tuple(snr_bands) if snr_bands else DEFAULT_SNR_BANDS

    def evaluate(self, data_loader: Any) -> Dict[str, Any]:
        sample_ids: List[str] = []
        probabilities = []
        targets = []
        snr_values = []
        local_snr_targets = []
        local_snr_predictions = []
        local_snr_masks = []
        loss_sum = 0.0
        window_count = 0
        self.model.eval()
        filter_loss_sum = 0.0

        with torch.no_grad():
            for batch in tqdm(data_loader, desc="Evaluating", unit="batch", dynamic_ncols=True):
                waveform = batch["waveform"].to(self.device, non_blocking=True)
                target = batch["target"].to(self.device, non_blocking=True)
                output = self.model(waveform)
                logits = output["clipwise_output"]
                if self.loss_fn is not None:
                    batch_size = waveform.size(0)
                    loss_targets = {
                        "target": target,
                        "local_snr_db": batch["local_snr_db"].to(self.device, non_blocking=True),
                        "local_snr_mask": batch["local_snr_mask"].to(
                            self.device, non_blocking=True
                        ),
                    }
                    if "noise_waveform" in batch:
                        loss_targets["noise_waveform"] = batch["noise_waveform"].to(
                            self.device, non_blocking=True
                        )
                    components = self.loss_fn.components(output, loss_targets)
                    loss_sum += float(components["total"].item()) * batch_size
                    if "filter" in components:
                        filter_loss_sum += float(components["filter"].item()) * batch_size
                    window_count += batch_size
                sample_ids.extend(list(batch["audio_name"]))
                probabilities.append(torch.sigmoid(logits).cpu().numpy())
                targets.append(target.cpu().numpy())
                snr_values.append(batch["target_snr_db"].cpu().numpy())
                local_snr_targets.append(batch["local_snr_db"].cpu().numpy())
                local_snr_masks.append(batch["local_snr_mask"].cpu().numpy())
                local_snr_predictions.append(output["local_snr_db"].cpu().numpy())

        window_probability = np.concatenate(probabilities, axis=0)
        window_target = np.concatenate(targets, axis=0)
        window_snr = np.concatenate(snr_values, axis=0)
        clip_ids, clip_probability, clip_target, clip_snr = aggregate_windows(
            sample_ids,
            window_probability,
            window_target,
            window_snr,
            reduction=self.window_reduction,
        )
        statistics = compute_multilabel_metrics(
            clip_target,
            clip_probability,
            self.threshold,
            self.label_names,
            single_label=self.single_label,
        )
        statistics["loss"] = loss_sum / max(window_count, 1)
        # The first number to watch when the filter is on: if it does not fall, the mask
        # has learned nothing and everything downstream of it is meaningless.
        statistics["filter_loss"] = filter_loss_sum / max(window_count, 1)
        statistics["sample_ids"] = clip_ids
        statistics["target_snr_db"] = clip_snr
        statistics["num_clips"] = len(clip_ids)
        statistics["num_windows"] = len(sample_ids)
        statistics.update(
            compute_local_snr_metrics(
                np.concatenate(local_snr_targets, axis=0),
                np.concatenate(local_snr_predictions, axis=0),
                np.concatenate(local_snr_masks, axis=0),
            )
        )

        statistics["snr_metrics"] = compute_snr_band_metrics(
            clip_target,
            clip_probability,
            clip_snr,
            self.threshold,
            self.label_names,
            self.snr_bands,
            single_label=self.single_label,
        )
        return statistics

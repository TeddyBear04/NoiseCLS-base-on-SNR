"""Evaluate a fine-tuned BEATs checkpoint and calibrate multilabel thresholds."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from finetune_beats_21 import load_model
from noise_pipeline.audio21 import (
    Audio21Dataset,
    load_label_metadata,
    multilabel_metrics,
)


@torch.inference_mode()
def collect_predictions(model, head, loader, device):
    model.eval()
    head.eval()
    probabilities, targets, snrs = [], [], []
    for batch_index, batch in enumerate(loader, start=1):
        waveforms = batch["waveform"].to(device, non_blocking=True)
        with torch.autocast(
            device_type=device.type,
            dtype=torch.bfloat16,
            enabled=device.type == "cuda",
        ):
            sequence, _ = model.extract_features(waveforms)
            logits = head(sequence.mean(dim=1))
        probabilities.append(torch.sigmoid(logits.float()).cpu())
        targets.append(batch["target"])
        snrs.append(batch["snr"])
        if batch_index == 1 or batch_index % 25 == 0 or batch_index == len(loader):
            print(f"validation batch={batch_index}/{len(loader)}", flush=True)
    return (
        torch.cat(probabilities).numpy(),
        torch.cat(targets).numpy(),
        torch.cat(snrs).numpy(),
    )


def calibrate_thresholds(probabilities, targets, lower=0.05, upper=0.95):
    """Choose robust 0.01-grid thresholds that maximize validation F1."""
    grid = np.round(np.arange(lower, upper + 0.001, 0.01), 2)
    thresholds = np.empty(targets.shape[1], dtype=np.float64)
    class_best_f1 = np.empty(targets.shape[1], dtype=np.float64)
    for class_index in range(targets.shape[1]):
        truth = targets[:, class_index].astype(bool)
        predicted = probabilities[:, class_index, None] >= grid[None, :]
        true_positive = (predicted & truth[:, None]).sum(axis=0)
        false_positive = (predicted & ~truth[:, None]).sum(axis=0)
        false_negative = (~predicted & truth[:, None]).sum(axis=0)
        denominator = 2 * true_positive + false_positive + false_negative
        scores = np.divide(
            2 * true_positive,
            denominator,
            out=np.zeros_like(denominator, dtype=np.float64),
            where=denominator > 0,
        )
        best_index = int(np.argmax(scores))
        thresholds[class_index] = grid[best_index]
        class_best_f1[class_index] = scores[best_index]
    return thresholds, class_best_f1


def calibrate_global_threshold(probabilities, targets, lower=0.05, upper=0.95):
    grid = np.round(np.arange(lower, upper + 0.001, 0.01), 2)
    best_threshold = 0.5
    best_macro_f1 = -1.0
    for threshold in grid:
        metrics = multilabel_metrics(
            probabilities,
            targets,
            np.zeros(len(targets)),
            [str(index) for index in range(targets.shape[1])],
            float(threshold),
        )
        if metrics["macro_f1"] > best_macro_f1:
            best_macro_f1 = metrics["macro_f1"]
            best_threshold = float(threshold)
    return best_threshold, best_macro_f1


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("21_labels_dataset"))
    parser.add_argument(
        "--checkpoint", type=Path, default=Path("checkpoints/beats_21_last4.pt")
    )
    parser.add_argument("--seconds", type=float, default=6.0)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--limit", type=int)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("benchmark_results/beats_21_last4_evaluation.json"),
    )
    parser.add_argument(
        "--threshold-output",
        type=Path,
        default=Path("checkpoints/beats_21_thresholds.json"),
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cuda.matmul.allow_tf32 = True
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    metadata = load_label_metadata(args.data_root)
    labels = [row["display_name"] for row in metadata]
    model, head, _ = load_model(device, metadata, checkpoint["trainable_blocks"])
    missing, unexpected = model.load_state_dict(
        checkpoint["encoder_delta"], strict=False
    )
    expected_missing = [key for key in missing if not key.startswith("predictor.")]
    if unexpected:
        raise RuntimeError(f"Unexpected encoder keys: {unexpected}")
    # A partial delta intentionally leaves all frozen base parameters missing.
    print(
        f"loaded encoder_delta={len(checkpoint['encoder_delta'])} "
        f"base_parameters_retained={len(expected_missing)}",
        flush=True,
    )
    head.load_state_dict(checkpoint["head"])

    dataset = Audio21Dataset(
        args.data_root, "validation", args.seconds, "center", args.limit
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=device.type == "cuda",
        persistent_workers=args.workers > 0,
    )
    probabilities, targets, snrs = collect_predictions(
        model, head, loader, device
    )
    fixed = multilabel_metrics(probabilities, targets, snrs, labels, 0.5)
    global_threshold, _ = calibrate_global_threshold(probabilities, targets)
    global_metrics = multilabel_metrics(
        probabilities, targets, snrs, labels, global_threshold
    )
    thresholds, _ = calibrate_thresholds(probabilities, targets)
    calibrated = multilabel_metrics(probabilities, targets, snrs, labels, thresholds)

    threshold_map = {
        label: float(thresholds[index]) for index, label in enumerate(labels)
    }
    result = {
        "stage": "beats_21_validation_threshold_calibration",
        "checkpoint": str(args.checkpoint),
        "split": "validation",
        "samples": len(dataset),
        "fixed_0_5": fixed,
        "global_threshold": global_threshold,
        "global_calibrated": global_metrics,
        "per_class_thresholds": threshold_map,
        "per_class_calibrated": calibrated,
        "calibration_note": (
            "Thresholds were selected on validation and must be frozen before "
            "a one-time test evaluation. mAP is threshold-independent."
        ),
        "test_used": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    threshold_artifact = {
        "checkpoint": str(args.checkpoint),
        "selection_split": "validation",
        "global_threshold": global_threshold,
        "per_class_thresholds": threshold_map,
        "labels": labels,
    }
    args.threshold_output.parent.mkdir(parents=True, exist_ok=True)
    args.threshold_output.write_text(
        json.dumps(threshold_artifact, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(
        f"fixed macro_f1={fixed['macro_f1']:.4f} "
        f"global({global_threshold:.2f}) macro_f1={global_metrics['macro_f1']:.4f} "
        f"per_class macro_f1={calibrated['macro_f1']:.4f} "
        f"mAP={calibrated['macro_average_precision']:.4f}",
        flush=True,
    )
    print(f"results={args.output} thresholds={args.threshold_output}", flush=True)


if __name__ == "__main__":
    main()

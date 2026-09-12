"""One-time BEATs test evaluation using thresholds frozen on validation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from evaluate_beats_21 import collect_predictions
from finetune_beats_21 import load_model
from noise_pipeline.audio21 import (
    Audio21Dataset,
    load_label_metadata,
    multilabel_metrics,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("21_labels_dataset"))
    parser.add_argument(
        "--checkpoint", type=Path, default=Path("checkpoints/beats_21_last4.pt")
    )
    parser.add_argument(
        "--thresholds",
        type=Path,
        default=Path("checkpoints/beats_21_thresholds.json"),
    )
    parser.add_argument("--seconds", type=float, default=6.0)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--limit", type=int)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("benchmark_results/beats_21_last4_test.json"),
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cuda.matmul.allow_tf32 = True
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    threshold_artifact = json.loads(args.thresholds.read_text(encoding="utf-8"))
    metadata = load_label_metadata(args.data_root)
    labels = [row["display_name"] for row in metadata]
    if threshold_artifact["labels"] != labels:
        raise ValueError("Threshold labels do not match dataset labels")

    model, head, _ = load_model(device, metadata, checkpoint["trainable_blocks"])
    _, unexpected = model.load_state_dict(checkpoint["encoder_delta"], strict=False)
    if unexpected:
        raise RuntimeError(f"Unexpected encoder keys: {unexpected}")
    head.load_state_dict(checkpoint["head"])
    dataset = Audio21Dataset(
        args.data_root, "test", args.seconds, "center", args.limit
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=device.type == "cuda",
        persistent_workers=args.workers > 0,
    )
    probabilities, targets, snrs = collect_predictions(model, head, loader, device)
    threshold_map = threshold_artifact["per_class_thresholds"]
    class_thresholds = np.asarray(
        [threshold_map[label] for label in labels], dtype=np.float64
    )
    global_threshold = float(threshold_artifact["global_threshold"])
    result = {
        "stage": "beats_21_one_time_test_evaluation",
        "checkpoint": str(args.checkpoint),
        "threshold_artifact": str(args.thresholds),
        "samples": len(dataset),
        "fixed_0_5": multilabel_metrics(
            probabilities, targets, snrs, labels, 0.5
        ),
        "global_threshold": global_threshold,
        "global_calibrated": multilabel_metrics(
            probabilities, targets, snrs, labels, global_threshold
        ),
        "per_class_thresholds": threshold_map,
        "per_class_calibrated": multilabel_metrics(
            probabilities, targets, snrs, labels, class_thresholds
        ),
        "calibration_note": (
            "All thresholds were frozen from validation before this test run; "
            "test labels were not used for model or threshold selection."
        ),
        "test_used": True,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    calibrated = result["per_class_calibrated"]
    print(
        f"test micro_f1={calibrated['micro_f1']:.4f} "
        f"macro_f1={calibrated['macro_f1']:.4f} "
        f"mAP={calibrated['macro_average_precision']:.4f} "
        f"exact_match={calibrated['exact_match']:.4f}",
        flush=True,
    )
    print(f"results={args.output}", flush=True)


if __name__ == "__main__":
    main()

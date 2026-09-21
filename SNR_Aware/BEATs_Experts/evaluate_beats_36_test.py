"""Evaluate the selected 36-label BEATs checkpoint on the held-out test split."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
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
from torch.utils.data import DataLoader

from finetune_beats import (
    MixtureDataset,
    load_model,
    mixed_precision_context,
    require_finite,
)
from noise_pipeline.mix_data import load_mix_manifest
from train_beats_head import filter_snr_rows
from config.paths import BEATS_CHECKPOINT
from utils.reporting36 import (
    full_metrics,
    per_class_metrics,
    print_report,
    save_evaluation_artifacts,
)


@torch.inference_mode()
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("mix-dataset"))
    parser.add_argument("--checkpoint", type=Path, default=Path("checkpoints/beats_last4.pt"))
    parser.add_argument("--head-checkpoint", type=Path, default=Path("checkpoints/beats_head.pt"))
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--split", choices=("validation", "test"), default="test")
    parser.add_argument("--output", type=Path, default=Path("checkpoint/test_metrics_36.json"))
    parser.add_argument("--output-dir", type=Path, default=Path("checkpoint"))
    parser.add_argument("--snr-min-db", type=float)
    parser.add_argument("--snr-max-db", type=float)
    parser.add_argument("--pretrained-checkpoint", type=Path, default=BEATS_CHECKPOINT)
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    model, head, labels = load_model(
        device, args.head_checkpoint, checkpoint["trainable_blocks"], args.pretrained_checkpoint
    )
    model.load_state_dict(checkpoint["encoder_delta"], strict=False)
    head.load_state_dict(checkpoint["head"])
    model.eval(); head.eval()
    rows = filter_snr_rows(
        load_mix_manifest(args.data_root), args.snr_min_db, args.snr_max_db
    )
    dataset = MixtureDataset(args.data_root, args.split, rows)
    loader = DataLoader(dataset, batch_size=args.batch_size, num_workers=args.workers,
                        pin_memory=device.type == "cuda", persistent_workers=args.workers > 0)
    predicted, expected, snrs, all_logits = [], [], [], []
    for batch in loader:
        with mixed_precision_context(device):
            sequence, _ = model.extract_features(batch["mixture"].to(device, non_blocking=True))
            logits = head(sequence.mean(dim=1))
        require_finite(logits, "evaluation logits")
        predicted.append(logits.argmax(dim=1).cpu())
        expected.append(batch["target"])
        snrs.append(batch["snr"])
        all_logits.append(logits.float().cpu())
    predicted = torch.cat(predicted).numpy(); expected = torch.cat(expected).numpy(); snrs = torch.cat(snrs).numpy()
    probabilities = torch.cat(all_logits).softmax(dim=1).numpy()
    metrics = full_metrics(expected, predicted, probabilities)
    per_snr = {}
    for snr in sorted(set(snrs.tolist())):
        mask = snrs == snr
        if not mask.any():
            continue
        per_snr[f"{snr:g}"] = {
            "samples": int(mask.sum()),
            **full_metrics(expected[mask], predicted[mask], probabilities[mask]),
            "si_sdr_db": None,  # BEATs does not separate noise, so there is no SI-SDR.
        }
    result = {
        "dataset": "mix-dataset", "task": "single-label_36", "split": args.split,
        "samples": int(len(dataset)), "checkpoint": str(args.checkpoint),
        "accuracy": metrics["accuracy"], "precision": metrics["precision"],
        "recall": metrics["recall"], "macro_f1": metrics["macro_f1"], "micro_f1": metrics["micro_f1"],
        "test_accuracy": metrics["accuracy"], "test_top3_accuracy": metrics["top3_accuracy"],
        "test_precision": metrics["precision"], "test_recall": metrics["recall"],
        "test_macro_f1": metrics["macro_f1"], "test_micro_f1": metrics["micro_f1"],
        "test_map": metrics["map"], "test_balanced_accuracy": metrics["balanced_accuracy"],
        "test_macro_auc": metrics["macro_auc"], "test_si_sdr_db": None,
        "per_snr": per_snr,
        "per_class": per_class_metrics(expected, predicted, probabilities, labels),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    save_evaluation_artifacts(args.output_dir, result, labels, expected, predicted)
    print_report(result)


if __name__ == "__main__":
    main()

"""Evaluate the selected 36-label BEATs checkpoint on the held-out test split."""

import argparse
import json
from pathlib import Path

import torch
from sklearn.metrics import f1_score, precision_recall_fscore_support
from torch.utils.data import DataLoader

from finetune_beats import (
    MixtureDataset,
    load_model,
    mixed_precision_context,
    require_finite,
)
from noise_pipeline.mix_data import load_mix_manifest
from utils.reporting36 import save_evaluation_artifacts


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
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    model, head, labels = load_model(device, args.head_checkpoint, checkpoint["trainable_blocks"])
    model.load_state_dict(checkpoint["encoder_delta"], strict=False)
    head.load_state_dict(checkpoint["head"])
    model.eval(); head.eval()
    dataset = MixtureDataset(args.data_root, args.split, load_mix_manifest(args.data_root))
    loader = DataLoader(dataset, batch_size=args.batch_size, num_workers=args.workers,
                        pin_memory=device.type == "cuda", persistent_workers=args.workers > 0)
    predicted, expected, snrs = [], [], []
    for batch in loader:
        with mixed_precision_context(device):
            sequence, _ = model.extract_features(batch["mixture"].to(device, non_blocking=True))
            logits = head(sequence.mean(dim=1))
        require_finite(logits, "evaluation logits")
        predicted.append(logits.argmax(dim=1).cpu())
        expected.append(batch["target"])
        snrs.append(batch["snr"])
    predicted = torch.cat(predicted).numpy(); expected = torch.cat(expected).numpy(); snrs = torch.cat(snrs).numpy()
    precision, recall, f1, support = precision_recall_fscore_support(expected, predicted, labels=range(len(labels)), zero_division=0)
    per_snr = {}
    for snr in (-5, 0, 5, 10, 15, 20):
        mask = snrs == snr
        if not mask.any():
            continue
        per_snr[str(snr)] = {
            "samples": int(mask.sum()),
            "accuracy": float((predicted[mask] == expected[mask]).mean()),
            "macro_f1": float(f1_score(expected[mask], predicted[mask], labels=range(len(labels)), average="macro", zero_division=0)),
            "micro_f1": float(f1_score(expected[mask], predicted[mask], average="micro", zero_division=0)),
        }
    result = {"dataset": "mix-dataset", "task": "single-label_36", "split": args.split, "samples": int(len(dataset)), "checkpoint": str(args.checkpoint), "accuracy": float((predicted == expected).mean()), "precision": float(precision.mean()), "recall": float(recall.mean()), "macro_f1": float(f1.mean()), "micro_f1": float(f1_score(expected, predicted, average="micro", zero_division=0)), "per_snr": per_snr, "per_class": {label: {"f1": float(f1[i]), "support": int(support[i])} for i, label in enumerate(labels)}}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    save_evaluation_artifacts(args.output_dir, result, labels, expected, predicted)
    print("accuracy={accuracy:.4f} precision={precision:.4f} recall={recall:.4f} macro_f1={macro_f1:.4f} micro_f1={micro_f1:.4f}".format(**result))


if __name__ == "__main__":
    main()

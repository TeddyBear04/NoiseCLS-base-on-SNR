"""Evaluate the selected 36-label BEATs checkpoint on the held-out test split."""

import argparse
import json
from pathlib import Path

import torch
from sklearn.metrics import precision_recall_fscore_support
from torch.utils.data import DataLoader

from finetune_beats import MixtureDataset, load_model
from noise_pipeline.mix_data import load_mix_manifest


@torch.inference_mode()
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("mix-dataset"))
    parser.add_argument("--checkpoint", type=Path, default=Path("checkpoints/beats_last4.pt"))
    parser.add_argument("--head-checkpoint", type=Path, default=Path("checkpoints/beats_head.pt"))
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--split", choices=("validation", "test"), default="test")
    parser.add_argument("--output", type=Path, default=Path("benchmark_results/beats_36_last4_test_metrics.json"))
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
    predicted, expected = [], []
    for batch in loader:
        with torch.autocast(device.type, dtype=torch.float16, enabled=device.type == "cuda"):
            sequence, _ = model.extract_features(batch["mixture"].to(device, non_blocking=True))
            logits = head(sequence.mean(dim=1))
        predicted.append(logits.argmax(dim=1).cpu())
        expected.append(batch["target"])
    predicted = torch.cat(predicted).numpy(); expected = torch.cat(expected).numpy()
    precision, recall, f1, support = precision_recall_fscore_support(expected, predicted, labels=range(len(labels)), zero_division=0)
    result = {"dataset": "mix-dataset", "task": "single-label_36", "split": args.split, "samples": int(len(dataset)), "checkpoint": str(args.checkpoint), "accuracy": float((predicted == expected).mean()), "macro_precision": float(precision.mean()), "macro_recall": float(recall.mean()), "macro_f1": float(f1.mean()), "per_class": {label: {"precision": float(precision[i]), "recall": float(recall[i]), "f1": float(f1[i]), "support": int(support[i])} for i, label in enumerate(labels)}}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print("accuracy={accuracy:.4f} macro_precision={macro_precision:.4f} macro_recall={macro_recall:.4f} macro_f1={macro_f1:.4f}".format(**result))


if __name__ == "__main__":
    main()

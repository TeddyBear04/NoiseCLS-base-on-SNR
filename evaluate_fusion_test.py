"""Evaluate the fine-tuned mixture/noise fusion checkpoint on a held-out split."""

import argparse
import json
from pathlib import Path

import torch

from finetune_fusion import evaluate, load_model, make_loader
from noise_pipeline.mix_data import MixNoiseDataset, load_mix_manifest
from utils.reporting36 import save_evaluation_artifacts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("../../36_labels"))
    parser.add_argument("--checkpoint", type=Path, default=Path("checkpoint/audio_best_36_fusion.pt"))
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--split", choices=("validation", "test"), default="test")
    parser.add_argument("--output", type=Path, default=Path("checkpoint/test_metrics_36_fusion.json"))
    parser.add_argument("--output-dir", type=Path, default=Path("checkpoint"))
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    encoder, separator, classifier, labels = load_model(
        device, checkpoint, checkpoint["trainable_blocks"],
        dropout=checkpoint["dropout"], noise_dropout=checkpoint.get("noise_dropout", 0.2),
        conditioned_dropout=checkpoint.get("conditioned_dropout", 0.2),
    )
    dataset = MixNoiseDataset(args.data_root, args.split, rows=load_mix_manifest(args.data_root))
    loader = make_loader(dataset, args.batch_size, args.workers, False, device)
    sep_loss_weight = checkpoint["args"]["sep_loss_weight"]
    metrics, expected, predicted = evaluate(
        encoder, separator, classifier, loader, device, labels, sep_loss_weight,
        return_predictions=True,
    )
    result = {
        "dataset": "36_labels",
        "input_kind": "mixture+separated_noise",
        "task": "single-label_36",
        "split": args.split,
        "samples": len(dataset),
        "checkpoint": str(args.checkpoint),
        **metrics,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    report = save_evaluation_artifacts(
        args.output_dir, result, labels, expected, predicted, split=args.split
    )
    print(
        "accuracy={accuracy:.4f} precision={precision:.4f} recall={recall:.4f} "
        "macro_f1={macro_f1:.4f} micro_f1={micro_f1:.4f} mAP={mAP:.4f} "
        "si_sdr={si_sdr:.2f}".format(**result)
    )
    print(report, flush=True)


if __name__ == "__main__":
    main()

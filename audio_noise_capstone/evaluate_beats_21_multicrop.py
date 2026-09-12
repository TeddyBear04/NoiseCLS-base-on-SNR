"""Compare center, mean three-crop, and max three-crop BEATs validation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from finetune_beats_21 import classify, load_model
from noise_pipeline.audio21 import load_label_metadata, multilabel_metrics
from noise_pipeline.audio21_crops import ThreeCropAudio21Dataset


@torch.inference_mode()
def collect(model, head, loader, device):
    model.eval()
    head.eval()
    crop_probabilities, targets, snrs = [], [], []
    for batch_index, batch in enumerate(loader, start=1):
        waveforms = batch["waveform"]
        batch_size, crop_count, samples = waveforms.shape
        waveforms = waveforms.reshape(batch_size * crop_count, samples).to(
            device, non_blocking=True
        )
        with torch.autocast(
            device_type=device.type,
            dtype=torch.bfloat16,
            enabled=device.type == "cuda",
        ):
            sequence, _ = model.extract_features(waveforms)
            logits = classify(head, sequence)
        probabilities = torch.sigmoid(logits.float()).reshape(
            batch_size, crop_count, -1
        )
        crop_probabilities.append(probabilities.cpu())
        targets.append(batch["target"])
        snrs.append(batch["snr"])
        if batch_index == 1 or batch_index % 25 == 0 or batch_index == len(loader):
            print(f"validation batch={batch_index}/{len(loader)}", flush=True)
    return torch.cat(crop_probabilities), torch.cat(targets), torch.cat(snrs)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("21_labels_dataset"))
    parser.add_argument(
        "--checkpoint", type=Path, default=Path("checkpoints/beats_21_last4.pt")
    )
    parser.add_argument("--seconds", type=float, default=6.0)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("benchmark_results/beats_21_last4_multicrop_validation.json"),
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cuda.matmul.allow_tf32 = True
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    metadata = load_label_metadata(args.data_root)
    labels = [row["display_name"] for row in metadata]
    pooling = checkpoint.get("args", {}).get("pooling", "mean")
    model, head, _ = load_model(
        device, metadata, checkpoint["trainable_blocks"], pooling
    )
    _, unexpected = model.load_state_dict(checkpoint["encoder_delta"], strict=False)
    if unexpected:
        raise RuntimeError(f"Unexpected encoder keys: {unexpected}")
    head.load_state_dict(checkpoint["head"])
    dataset = ThreeCropAudio21Dataset(
        args.data_root,
        "validation",
        seconds=args.seconds,
        limit=args.limit,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=device.type == "cuda",
        persistent_workers=args.workers > 0,
    )
    crop_probabilities, targets, snrs = collect(model, head, loader, device)
    views = {
        "start": crop_probabilities[:, 0],
        "center": crop_probabilities[:, 1],
        "end": crop_probabilities[:, 2],
        "three_crop_mean": crop_probabilities.mean(dim=1),
        "three_crop_max": crop_probabilities.max(dim=1).values,
    }
    result = {
        "stage": "beats_21_three_crop_ablation",
        "checkpoint": str(args.checkpoint),
        "pooling": pooling,
        "split": "validation",
        "samples": len(dataset),
        "seconds_per_crop": args.seconds,
        "crop_positions": ["start", "center", "end"],
        "threshold": args.threshold,
        "threshold_calibrated": False,
        "test_used": False,
        "views": {
            name: multilabel_metrics(probability, targets, snrs, labels, args.threshold)
            for name, probability in views.items()
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    for name, metrics in result["views"].items():
        print(
            f"view={name} micro_f1={metrics['micro_f1']:.4f} "
            f"macro_f1={metrics['macro_f1']:.4f} "
            f"mAP={metrics['macro_average_precision']:.4f}",
            flush=True,
        )
    print(f"results={args.output}", flush=True)


if __name__ == "__main__":
    main()

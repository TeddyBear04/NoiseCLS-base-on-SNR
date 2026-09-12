"""Report BEATs validation false-positive/false-negative context by class."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from finetune_beats_21 import classify, load_model
from noise_pipeline.audio21 import Audio21Dataset, load_label_metadata


@torch.inference_mode()
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=Path("checkpoints/beats_21_last4.pt"))
    parser.add_argument("--output", type=Path, default=Path("benchmark_results/beats_21_error_analysis.json"))
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    metadata = load_label_metadata("21_labels_dataset")
    labels = [row["display_name"] for row in metadata]
    model, head, _ = load_model(device, metadata, checkpoint["trainable_blocks"])
    model.load_state_dict(checkpoint["encoder_delta"], strict=False)
    head.load_state_dict(checkpoint["head"])
    model.eval()
    head.eval()
    dataset = Audio21Dataset("21_labels_dataset", "validation", crop="center")
    loader = DataLoader(dataset, batch_size=args.batch_size, num_workers=args.workers,
                        pin_memory=device.type == "cuda", persistent_workers=args.workers > 0)
    probabilities, targets = [], []
    for batch in loader:
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            sequence, _ = model.extract_features(batch["waveform"].to(device, non_blocking=True))
            logits = classify(head, sequence)
        probabilities.append(torch.sigmoid(logits.float()).cpu())
        targets.append(batch["target"])
    probability = torch.cat(probabilities).numpy()
    target = torch.cat(targets).numpy().astype(bool)
    prediction = probability >= 0.5
    results = {}
    for index, label in enumerate(labels):
        entry = {"support": int(target[:, index].sum()), "false_positive": int((prediction[:, index] & ~target[:, index]).sum()), "false_negative": int((~prediction[:, index] & target[:, index]).sum())}
        for name, mask in (("fp_label_context", prediction[:, index] & ~target[:, index]), ("fn_label_context", ~prediction[:, index] & target[:, index])):
            counts = target[mask].sum(axis=0)
            counts[index] = 0
            top = np.argsort(counts)[-5:][::-1]
            entry[name] = [{"label": labels[item], "count": int(counts[item])} for item in top if counts[item] > 0]
        results[label] = entry
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({"checkpoint": str(args.checkpoint), "threshold": 0.5, "classes": results}, ensure_ascii=False, indent=2), encoding="utf-8")
    for label in ("Speech", "Inside, small room", "Car", "Musical instrument"):
        print(label, results[label], flush=True)


if __name__ == "__main__":
    main()

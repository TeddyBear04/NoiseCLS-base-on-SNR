import argparse
from pathlib import Path

import torch
import torchaudio
from torch.utils.data import DataLoader

from noise_pipeline.data import NoiseDataset
from noise_pipeline.model import NoiseNet


def calculate_metrics(probabilities, targets, threshold):
    predictions = probabilities >= threshold
    targets = targets.bool()
    tp = (predictions & targets).sum(dim=0).float()
    fp = (predictions & ~targets).sum(dim=0).float()
    fn = (~predictions & targets).sum(dim=0).float()
    eps = 1e-8
    precision = tp / (tp + fp + eps)
    recall = tp / (tp + fn + eps)
    class_f1 = 2 * precision * recall / (precision + recall + eps)
    micro_precision = tp.sum() / (tp.sum() + fp.sum() + eps)
    micro_recall = tp.sum() / (tp.sum() + fn.sum() + eps)
    micro_f1 = 2 * micro_precision * micro_recall / (
        micro_precision + micro_recall + eps
    )
    return {
        "threshold": threshold,
        "micro_precision": micro_precision.item(),
        "micro_recall": micro_recall.item(),
        "micro_f1": micro_f1.item(),
        "macro_f1": class_f1.mean().item(),
        "exact": (predictions == targets).all(dim=1).float().mean().item(),
    }


def collect_predictions(model, loader, transform, device):
    probabilities, targets = [], []
    model.eval()
    with torch.inference_mode():
        for batch in loader:
            mixture = batch["mixture"].to(device)
            features = torch.log(transform(mixture).squeeze(1).clamp_min(1e-6))
            logits, _, _ = model(features)
            probabilities.append(logits.sigmoid().cpu())
            targets.append(batch["target"])
    return torch.cat(probabilities), torch.cat(targets)


def main():
    parser = argparse.ArgumentParser(
        description="Tune a global threshold on validation or evaluate a fixed threshold"
    )
    parser.add_argument("--checkpoint", default="checkpoints/best.pt")
    parser.add_argument("--data-root", default="21_labels_dataset")
    parser.add_argument("--split", default="validation_single")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--threshold", type=float)
    parser.add_argument("--start", type=float, default=0.50)
    parser.add_argument("--stop", type=float, default=0.90)
    parser.add_argument("--step", type=float, default=0.01)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    saved_args = checkpoint.get("args", {})
    seconds = float(saved_args.get("seconds", 6.0))
    dataset = NoiseDataset(args.data_root, args.split, seconds)
    loader = DataLoader(
        dataset, args.batch_size, num_workers=args.workers, pin_memory=device.type == "cuda"
    )
    model = NoiseNet(num_classes=dataset.num_classes).to(device)
    model.load_state_dict(checkpoint["model"])
    transform = torchaudio.transforms.MelSpectrogram(
        sample_rate=16_000, n_fft=512, win_length=400, hop_length=160,
        n_mels=64, f_min=0, f_max=8_000, power=2.0,
    ).to(device)

    probabilities, targets = collect_predictions(model, loader, transform, device)
    if args.threshold is not None:
        results = [calculate_metrics(probabilities, targets, args.threshold)]
    else:
        count = round((args.stop - args.start) / args.step) + 1
        thresholds = [args.start + index * args.step for index in range(count)]
        results = [calculate_metrics(probabilities, targets, value) for value in thresholds]

    best = max(results, key=lambda item: item["micro_f1"])
    print(f"device={device} split={args.split} samples={len(dataset)}")
    print(
        f"threshold={best['threshold']:.2f} micro_p={best['micro_precision']:.4f} "
        f"micro_r={best['micro_recall']:.4f} micro_f1={best['micro_f1']:.4f} "
        f"macro_f1={best['macro_f1']:.4f} exact={best['exact']:.4f}"
    )


if __name__ == "__main__":
    main()

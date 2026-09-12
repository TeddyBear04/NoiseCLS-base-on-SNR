import argparse
import csv
from pathlib import Path

import torch
import torchaudio

from noise_pipeline.data import load_pcm16
from noise_pipeline.model import NoiseNet


def main():
    parser = argparse.ArgumentParser(description="Predict noise labels for one PCM16 WAV file")
    parser.add_argument("wav")
    parser.add_argument("--checkpoint", default="checkpoints/best.pt")
    parser.add_argument("--data-root", default="21_labels_dataset")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    saved_args = checkpoint.get("args", {})
    seconds = float(saved_args.get("seconds", 6.0))
    with (Path(args.data_root) / "selected_labels.csv").open(
        encoding="utf-8-sig", newline=""
    ) as file:
        labels = sorted(csv.DictReader(file), key=lambda row: int(row["model_index"]))

    model = NoiseNet(num_classes=len(labels)).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    waveform = load_pcm16(args.wav, 16_000, int(seconds * 16_000)).unsqueeze(0).to(device)
    transform = torchaudio.transforms.MelSpectrogram(
        sample_rate=16_000, n_fft=512, win_length=400, hop_length=160,
        n_mels=64, f_min=0, f_max=8_000, power=2.0,
    ).to(device)
    features = torch.log(transform(waveform).squeeze(1).clamp_min(1e-6))
    with torch.inference_mode():
        logits, _, _ = model(features)
        probabilities = logits.sigmoid()[0].cpu()

    ranked = probabilities.argsort(descending=True)
    selected = [index for index in ranked if probabilities[index] >= args.threshold]
    if not selected:
        selected = ranked[: args.top_k].tolist()
        print(f"No label reached threshold {args.threshold:.2f}; showing top {args.top_k}.")
    else:
        selected = selected[: args.top_k]
    for index in selected:
        print(f"{probabilities[index].item():.4f}\t{labels[index]['display_name']}")


if __name__ == "__main__":
    main()

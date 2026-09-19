"""Train, evaluate, and run inference for the DPCRN noise-target expert."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import soundfile as sound_file
import torch
from sklearn.metrics import accuracy_score, f1_score
from torch import Tensor, nn
from torch.utils.data import DataLoader

from dataset import MixNoiseDataset
from models import DPCRNNoiseClassifier


def load_config(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def build_model(config: dict[str, Any], classes_num: int, device: torch.device) -> DPCRNNoiseClassifier:
    model_config = config["model"]
    model = DPCRNNoiseClassifier(
        classes_num=classes_num,
        n_fft=model_config["n_fft"],
        hop_length=model_config["hop_length"],
        encoder_channels=tuple(model_config["encoder_channels"]),
        dprnn_blocks=model_config["dprnn_blocks"],
        embedding_dim=model_config["embedding_dim"],
    )
    return model.to(device)


def separation_loss(estimated: Tensor, target: Tensor) -> Tensor:
    scale = target.abs().mean(dim=-1, keepdim=True).clamp_min(1e-4)
    waveform_l1 = ((estimated - target).abs() / scale).mean()
    estimated_spectrum = torch.stft(estimated, 512, 160, return_complex=True).abs()
    target_spectrum = torch.stft(target, 512, 160, return_complex=True).abs()
    spectral_l1 = (torch.log1p(estimated_spectrum) - torch.log1p(target_spectrum)).abs().mean()
    return waveform_l1 + spectral_l1


def make_loader(config: dict[str, Any], split: str, shuffle: bool) -> tuple[MixNoiseDataset, DataLoader[dict[str, Any]]]:
    dataset_config = config["dataset"]
    dataset = MixNoiseDataset(
        dataset_config["path"],
        split,
        dataset_config["clip_seconds"],
        dataset_config["snr_min_db"],
        dataset_config["snr_max_db"],
    )
    training = config["training"]
    return dataset, DataLoader(
        dataset,
        batch_size=training["batch_size"],
        shuffle=shuffle,
        num_workers=training["workers"],
        pin_memory=torch.cuda.is_available(),
    )


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader[dict[str, Any]], device: torch.device) -> dict[str, float]:
    model.eval()
    targets: list[int] = []
    predictions: list[int] = []
    for batch in loader:
        output = model(batch["mixture"].to(device))
        targets.extend(batch["target"].tolist())
        predictions.extend(output["logits"].argmax(dim=1).cpu().tolist())
    return {
        "accuracy": float(accuracy_score(targets, predictions)),
        "macro_f1": float(f1_score(targets, predictions, average="macro", zero_division=0)),
    }


def train(config: dict[str, Any], device: torch.device) -> None:
    train_data, train_loader = make_loader(config, "train", shuffle=True)
    _, validation_loader = make_loader(config, "validation", shuffle=False)
    model = build_model(config, len(train_data.labels), device)
    training = config["training"]
    optimizer = torch.optim.AdamW(model.parameters(), lr=training["learning_rate"], weight_decay=training["weight_decay"])
    cross_entropy = nn.CrossEntropyLoss()
    checkpoint = Path(training["checkpoint"])
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    best_f1 = float("-inf")

    for epoch in range(1, training["epochs"] + 1):
        model.train()
        for batch in train_loader:
            mixture = batch["mixture"].to(device)
            noise = batch["noise"].to(device)
            target = batch["target"].to(device)
            output = model(mixture)
            loss = cross_entropy(output["logits"], target) + training["lambda_separation"] * separation_loss(output["noise"], noise)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
        metrics = evaluate(model, validation_loader, device)
        print(f"epoch={epoch} validation_macro_f1={metrics['macro_f1']:.4f} validation_accuracy={metrics['accuracy']:.4f}")
        if metrics["macro_f1"] > best_f1:
            best_f1 = metrics["macro_f1"]
            torch.save({"model": model.state_dict(), "labels": train_data.labels, "config": config}, checkpoint)


def test(config: dict[str, Any], checkpoint_path: str | Path, device: torch.device) -> None:
    payload = torch.load(checkpoint_path, map_location=device)
    _, loader = make_loader(config, "test", shuffle=False)
    model = build_model(config, len(payload["labels"]), device)
    model.load_state_dict(payload["model"])
    print(json.dumps(evaluate(model, loader, device), indent=2))


def predict(config: dict[str, Any], checkpoint_path: str | Path, audio_path: str | Path, device: torch.device) -> None:
    payload = torch.load(checkpoint_path, map_location=device)
    model = build_model(config, len(payload["labels"]), device)
    model.load_state_dict(payload["model"])
    audio, rate = sound_file.read(audio_path, dtype="float32", always_2d=True)
    waveform = torch.from_numpy(audio.copy()).mean(dim=1)
    expected = int(config["dataset"]["clip_seconds"] * config["dataset"]["sample_rate"])
    waveform = waveform[:expected]
    waveform = torch.nn.functional.pad(waveform, (0, max(0, expected - waveform.numel())))
    with torch.no_grad():
        probabilities = model(waveform.unsqueeze(0).to(device))["logits"].softmax(dim=1)[0]
    index = int(probabilities.argmax())
    print(json.dumps({"label": payload["labels"][index], "confidence": float(probabilities[index])}, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("train36", "test36", "predict"))
    parser.add_argument("audio", nargs="?")
    parser.add_argument("--config", default="config/train_config.json")
    parser.add_argument("--checkpoint")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    config = load_config(args.config)
    device = torch.device(args.device)
    checkpoint = args.checkpoint or config["training"]["checkpoint"]
    if args.command == "train36":
        train(config, device)
    elif args.command == "test36":
        test(config, checkpoint, device)
    else:
        if args.audio is None:
            parser.error("predict requires an audio path")
        predict(config, checkpoint, args.audio, device)


if __name__ == "__main__":
    main()

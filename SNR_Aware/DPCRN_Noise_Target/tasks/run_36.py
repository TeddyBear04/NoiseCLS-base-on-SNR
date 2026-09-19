"""Train, evaluate, and run inference for the DPCRN noise-target expert."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sound_file
import torch
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from torch import Tensor, nn
from torch.utils.data import DataLoader

from dataset import MixNoiseDataset
from models import DPCRNNoiseClassifier


def load_config(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def validate_config(config: dict[str, Any]) -> None:
    for section in ("dataset", "model", "training", "evaluation", "runtime"):
        if section not in config:
            raise ValueError(f"Missing required config section: {section}")
    model = config["model"]
    if model["dprnn_feature_dim"] != model["encoder_channels"][-1]:
        raise ValueError("model.dprnn_feature_dim must match the final encoder channel count.")
    if config["training"]["optimizer"] != "AdamW":
        raise ValueError("Only AdamW is implemented by this runner.")
    if model["num_classes"] != config["experiment"]["num_classes"]:
        raise ValueError("model.num_classes must match experiment.num_classes.")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_model(config: dict[str, Any], classes_num: int, device: torch.device) -> DPCRNNoiseClassifier:
    model_config = config["model"]
    model = DPCRNNoiseClassifier(
        classes_num=classes_num,
        n_fft=model_config["n_fft"],
        hop_length=model_config["hop_length"],
        encoder_channels=tuple(model_config["encoder_channels"]),
        dprnn_blocks=model_config["dprnn_blocks"],
        embedding_dim=model_config["embedding_dim"],
        classifier_dropout=model_config["classifier_dropout"],
    )
    return model.to(device)


def separation_loss(estimated: Tensor, target: Tensor, n_fft: int, hop_length: int) -> Tensor:
    scale = target.abs().mean(dim=-1, keepdim=True).clamp_min(1e-4)
    waveform_l1 = ((estimated - target).abs() / scale).mean()
    estimated_spectrum = torch.stft(estimated, n_fft, hop_length, return_complex=True).abs()
    target_spectrum = torch.stft(target, n_fft, hop_length, return_complex=True).abs()
    spectral_l1 = (torch.log1p(estimated_spectrum) - torch.log1p(target_spectrum)).abs().mean()
    return waveform_l1 + spectral_l1


def make_loader(config: dict[str, Any], split: str, shuffle: bool) -> tuple[MixNoiseDataset, DataLoader[dict[str, Any]]]:
    dataset_config = config["dataset"]
    dataset = MixNoiseDataset(
        dataset_config["path"],
        split,
        dataset_config["clip_seconds"],
        dataset_config["sample_rate"],
        dataset_config["snr_min_db"],
        dataset_config["snr_max_db"],
        dataset_config["manifest_file"],
        dataset_config["labels_file"],
    )
    training = config["training"]
    return dataset, DataLoader(
        dataset,
        batch_size=training["batch_size"],
        shuffle=shuffle,
        num_workers=training["workers"],
        pin_memory=config["runtime"]["pin_memory"] and torch.cuda.is_available(),
        persistent_workers=training["workers"] > 0,
    )


def metric_values(
    targets: np.ndarray, predictions: np.ndarray, probabilities: np.ndarray
) -> dict[str, float]:
    one_hot_targets = np.eye(probabilities.shape[1], dtype=np.float32)[targets]
    try:
        macro_auc = float(
            roc_auc_score(one_hot_targets, probabilities, average="macro", multi_class="ovr")
        )
        mean_ap = float(average_precision_score(one_hot_targets, probabilities, average="macro"))
    except ValueError:
        # A debug slice can lack one or more of the 36 classes.
        macro_auc, mean_ap = float("nan"), float("nan")
    return {
        "accuracy": float(accuracy_score(targets, predictions)),
        "precision": float(precision_score(targets, predictions, average="macro", zero_division=0)),
        "recall": float(recall_score(targets, predictions, average="macro", zero_division=0)),
        "macro_f1": float(f1_score(targets, predictions, average="macro", zero_division=0)),
        "micro_f1": float(f1_score(targets, predictions, average="micro", zero_division=0)),
        "map": mean_ap,
        "balanced_accuracy": float(balanced_accuracy_score(targets, predictions)),
        "macro_auc": macro_auc,
    }


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader[dict[str, Any]],
    device: torch.device,
    per_snr_enabled: bool,
) -> dict[str, Any]:
    model.eval()
    targets: list[int] = []
    predictions: list[int] = []
    probabilities: list[np.ndarray] = []
    snrs: list[float] = []
    for batch in loader:
        output = model(batch["mixture"].to(device))
        posterior = output["logits"].softmax(dim=1).cpu().numpy()
        targets.extend(batch["target"].tolist())
        predictions.extend(output["logits"].argmax(dim=1).cpu().tolist())
        probabilities.extend(posterior)
        snrs.extend(batch["snr"].tolist())

    target_array = np.asarray(targets)
    prediction_array = np.asarray(predictions)
    probability_array = np.asarray(probabilities)
    overall = metric_values(target_array, prediction_array, probability_array)
    per_snr: dict[str, dict[str, float | int]] = {}
    snr_array = np.asarray(snrs)
    if per_snr_enabled:
        for snr in sorted(set(snrs)):
            mask = snr_array == snr
            per_snr[f"{snr:g}"] = {
                "samples": int(mask.sum()),
                **metric_values(target_array[mask], prediction_array[mask], probability_array[mask]),
            }
    return {
        **{f"test_{key}": value for key, value in overall.items()},
        "samples": int(len(target_array)),
        "per_snr": per_snr,
    }


def print_test_table(metrics: dict[str, Any]) -> None:
    headers = (
        "Test Accuracy", "Test Precision", "Test Recall", "Test Macro-F1",
        "Test Micro-F1", "Test mAP", "Test Balanced Acc", "Test Macro-AUC",
    )
    values = (
        metrics["test_accuracy"], metrics["test_precision"], metrics["test_recall"], metrics["test_macro_f1"],
        metrics["test_micro_f1"], metrics["test_map"], metrics["test_balanced_accuracy"], metrics["test_macro_auc"],
    )
    print("| " + " | ".join(headers) + " |")
    print("|" + "|".join("---" for _ in headers) + "|")
    print("| " + " | ".join(f"{value:.4f}" for value in values) + " |")
    if metrics["per_snr"]:
        print("\n| SNR (dB) | Samples | Accuracy | Macro-F1 | mAP | Balanced Acc | Macro-AUC |")
        print("| --- | --- | --- | --- | --- | --- | --- |")
        for snr, values in metrics["per_snr"].items():
            print(
                f"| {snr} | {values['samples']} | {values['accuracy']:.4f} | "
                f"{values['macro_f1']:.4f} | {values['map']:.4f} | "
                f"{values['balanced_accuracy']:.4f} | {values['macro_auc']:.4f} |"
            )


def train(config: dict[str, Any], device: torch.device) -> None:
    dataset_config = config["dataset"]
    train_data, train_loader = make_loader(config, dataset_config["train_split"], shuffle=True)
    _, validation_loader = make_loader(config, dataset_config["validation_split"], shuffle=False)
    model = build_model(config, len(train_data.labels), device)
    training = config["training"]
    optimizer = torch.optim.AdamW(model.parameters(), lr=training["learning_rate"], weight_decay=training["weight_decay"])
    cross_entropy = nn.CrossEntropyLoss()
    checkpoint = Path(training["checkpoint"])
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    best_f1 = float("-inf")
    stale_epochs = 0
    history: list[dict[str, Any]] = []

    for epoch in range(1, training["epochs"] + 1):
        model.train()
        for batch in train_loader:
            mixture = batch["mixture"].to(device)
            noise = batch["noise"].to(device)
            target = batch["target"].to(device)
            output = model(mixture)
            classification = cross_entropy(output["logits"], target)
            separation = separation_loss(
                output["noise"], noise, config["model"]["n_fft"], config["model"]["hop_length"]
            )
            loss = training["lambda_classification"] * classification + training["lambda_separation"] * separation
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), training["gradient_clip_norm"])
            optimizer.step()
        metrics = evaluate(
            model, validation_loader, device, config["evaluation"]["per_snr"]
        )
        history.append({"epoch": epoch, "validation": metrics})
        print(f"epoch={epoch} validation_macro_f1={metrics['test_macro_f1']:.4f} validation_accuracy={metrics['test_accuracy']:.4f}")
        selection = metrics[f"test_{training['selection_metric']}"]
        if selection > best_f1:
            best_f1 = selection
            stale_epochs = 0
            torch.save({"model": model.state_dict(), "labels": train_data.labels, "config": config, "best_epoch": epoch, "validation_metrics": metrics}, checkpoint)
        else:
            stale_epochs += 1
            if stale_epochs >= training["patience"]:
                print(f"early_stop={epoch}")
                break
    results_path = Path(training["results"])
    results_path.parent.mkdir(parents=True, exist_ok=True)
    results_path.write_text(
        json.dumps({"best_validation_macro_f1": best_f1, "history": history}, indent=2),
        encoding="utf-8",
    )


def test(config: dict[str, Any], checkpoint_path: str | Path, device: torch.device) -> None:
    payload = torch.load(checkpoint_path, map_location=device)
    _, loader = make_loader(config, config["evaluation"]["split"], shuffle=False)
    model = build_model(config, len(payload["labels"]), device)
    model.load_state_dict(payload["model"])
    metrics = evaluate(model, loader, device, config["evaluation"]["per_snr"])
    output_path = Path(config["evaluation"]["output"])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print_test_table(metrics)
    print(f"Saved metrics to {output_path}")


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
    parser.add_argument("--device")
    args = parser.parse_args()
    config = load_config(args.config)
    validate_config(config)
    set_seed(config["experiment"]["seed"])
    requested_device = args.device or config["runtime"]["device"]
    if requested_device == "cuda" and not torch.cuda.is_available():
        requested_device = "cpu"
    device = torch.device(requested_device)
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

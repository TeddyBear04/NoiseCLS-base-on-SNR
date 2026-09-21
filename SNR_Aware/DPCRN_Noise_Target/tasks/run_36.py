"""Train, evaluate, and run inference for the DPCRN noise-target expert."""

from __future__ import annotations

import argparse
import csv
import json
import math
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
    precision_recall_fscore_support,
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
        bidirectional_time=model_config["bidirectional_time"],
    )
    return model.to(device)


def build_scheduler(
    optimizer: torch.optim.Optimizer, config: dict[str, Any], steps_per_epoch: int
) -> torch.optim.lr_scheduler.LambdaLR:
    """Linear warmup, then cosine decay to zero over the remaining steps."""
    training = config["training"]
    warmup = max(1, training["warmup_epochs"] * steps_per_epoch)
    total = max(warmup + 1, training["epochs"] * steps_per_epoch)

    def factor(step: int) -> float:
        if step < warmup:
            return (step + 1) / warmup
        progress = (step - warmup) / (total - warmup)
        return 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, factor)


def separation_loss(estimated: Tensor, target: Tensor, n_fft: int, hop_length: int) -> Tensor:
    """Negative SNR plus a log spectral MSE term (DPCRN paper, eq. 6)."""
    noise_power = (target - estimated).pow(2).sum(dim=-1)
    signal_power = target.pow(2).sum(dim=-1)
    negative_snr = -10 * torch.log10(signal_power.clamp_min(1e-8) / noise_power.clamp_min(1e-8))

    window = torch.hann_window(n_fft, device=estimated.device)
    estimated_spectrum = torch.stft(
        estimated, n_fft, hop_length, window=window, return_complex=True
    )
    target_spectrum = torch.stft(
        target, n_fft, hop_length, window=window, return_complex=True
    )
    spectral = (
        (estimated_spectrum.real - target_spectrum.real).pow(2).mean()
        + (estimated_spectrum.imag - target_spectrum.imag).pow(2).mean()
        + (estimated_spectrum.abs() - target_spectrum.abs()).pow(2).mean()
    )
    return negative_snr.mean() + torch.log(spectral.clamp_min(1e-8))


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


def si_sdr(estimate: Tensor, target: Tensor, eps: float = 1e-8) -> Tensor:
    """Scale-invariant SDR in dB, one value per clip."""
    estimate = estimate - estimate.mean(dim=-1, keepdim=True)
    target = target - target.mean(dim=-1, keepdim=True)
    scale = (estimate * target).sum(dim=-1, keepdim=True) / (target.pow(2).sum(dim=-1, keepdim=True) + eps)
    projection = scale * target
    residual = estimate - projection
    return 10 * torch.log10((projection.pow(2).sum(dim=-1) + eps) / (residual.pow(2).sum(dim=-1) + eps))


def top_k_accuracy(targets: np.ndarray, probabilities: np.ndarray, k: int = 3) -> float:
    top = np.argsort(-probabilities, axis=1)[:, :k]
    return float((top == targets[:, None]).any(axis=1).mean())


def per_class_metrics(
    targets: np.ndarray, predictions: np.ndarray, probabilities: np.ndarray, labels: list[str]
) -> dict[str, dict[str, float | int]]:
    one_hot_targets = np.eye(len(labels), dtype=np.int32)[targets]
    precision, recall, f1, support = precision_recall_fscore_support(
        targets, predictions, labels=range(len(labels)), zero_division=0
    )
    rows: dict[str, dict[str, float | int]] = {}
    for index, label in enumerate(labels):
        try:
            ap = float(average_precision_score(one_hot_targets[:, index], probabilities[:, index]))
            auc = float(roc_auc_score(one_hot_targets[:, index], probabilities[:, index]))
        except ValueError:
            ap, auc = float("nan"), float("nan")
        rows[label] = {
            "support": int(support[index]), "precision": float(precision[index]),
            "recall": float(recall[index]), "f1": float(f1[index]), "ap": ap, "auc": auc,
        }
    return rows


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
        "top3_accuracy": top_k_accuracy(targets, probabilities),
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
    labels: list[str] | None = None,
) -> dict[str, Any]:
    """Score a loader. Pass `labels` for the test report: it adds SI-SDR and per-label rows."""
    model.eval()
    targets: list[int] = []
    predictions: list[int] = []
    probabilities: list[np.ndarray] = []
    snrs: list[float] = []
    separation_scores: list[float] = []
    for batch in loader:
        output = model(batch["mixture"].to(device))
        posterior = output["logits"].softmax(dim=1).cpu().numpy()
        targets.extend(batch["target"].tolist())
        predictions.extend(output["logits"].argmax(dim=1).cpu().tolist())
        probabilities.extend(posterior)
        snrs.extend(batch["snr"].tolist())
        if labels is not None:
            separation_scores.extend(si_sdr(output["noise"], batch["noise"].to(device)).cpu().tolist())

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
            if labels is not None:
                per_snr[f"{snr:g}"]["si_sdr_db"] = float(np.mean(np.asarray(separation_scores)[mask]))
    result = {
        **{f"test_{key}": value for key, value in overall.items()},
        "samples": int(len(target_array)),
        "per_snr": per_snr,
    }
    if labels is not None:
        result["test_si_sdr_db"] = float(np.mean(separation_scores))
        result["per_class"] = per_class_metrics(target_array, prediction_array, probability_array, labels)
    return result


# (header, key) pairs. The per-SNR table reads `key`; its "All" row reads `test_<key>`.
SNR_COLUMNS = (
    ("Top-1 Accuracy", "accuracy"), ("Top-3 Accuracy", "top3_accuracy"),
    ("Balanced Acc", "balanced_accuracy"), ("Precision Macro", "precision"),
    ("Recall Macro", "recall"), ("Macro-F1", "macro_f1"), ("Micro-F1", "micro_f1"),
    ("mAP", "map"), ("Macro-AUC", "macro_auc"), ("SI-SDR (dB)", "si_sdr_db"),
)
TEST_COLUMNS = (
    ("Test Accuracy", "test_accuracy"), ("Test Precision", "test_precision"),
    ("Test Recall", "test_recall"), ("Test Macro-F1", "test_macro_f1"),
    ("Test Micro-F1", "test_micro_f1"), ("Test mAP", "test_map"),
    ("Test Balanced Acc", "test_balanced_accuracy"), ("Test Macro-AUC", "test_macro_auc"),
)
CLASS_COLUMNS = (
    ("Support", "support"), ("Precision", "precision"), ("Recall", "recall"),
    ("F1", "f1"), ("AP", "ap"), ("AUC", "auc"),
)


def format_cell(value: Any) -> str:
    if value is None:
        return "-"
    if isinstance(value, int):
        return str(value)
    return f"{value:.4f}"


def markdown_table(headers: list[str], rows: list[list[str]]) -> str:
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join("---" for _ in headers) + "|"]
    lines.extend("| " + " | ".join(row) + " |" for row in rows)
    return "\n".join(lines)


def print_test_table(metrics: dict[str, Any]) -> None:
    print(markdown_table([h for h, _ in TEST_COLUMNS], [[format_cell(metrics[k]) for _, k in TEST_COLUMNS]]))
    if metrics["per_snr"]:
        snr_rows = [
            [snr, *(format_cell(values.get(key)) for _, key in SNR_COLUMNS)]
            for snr, values in metrics["per_snr"].items()
        ]
        snr_rows.append(["All", *(format_cell(metrics.get(f"test_{key}")) for _, key in SNR_COLUMNS)])
        print("\n" + markdown_table(["SNR (dB)", *(h for h, _ in SNR_COLUMNS)], snr_rows))
    class_rows = [
        [label, *(format_cell(values[key]) for _, key in CLASS_COLUMNS)]
        for label, values in metrics["per_class"].items()
    ]
    print("\n" + markdown_table(["Label", *(h for h, _ in CLASS_COLUMNS)], class_rows))


def save_report_csvs(output_dir: Path, metrics: dict[str, Any]) -> None:
    """Write the per-SNR and per-label tables next to the JSON metrics."""
    snr_fields = ["snr_db", "samples", *(key for _, key in SNR_COLUMNS)]
    with (output_dir / "snr_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=snr_fields)
        writer.writeheader()
        for snr, values in metrics["per_snr"].items():
            writer.writerow({"snr_db": snr, **{key: values.get(key) for key in snr_fields[1:]}})
        writer.writerow({
            "snr_db": "All", "samples": metrics["samples"],
            **{key: metrics.get(f"test_{key}") for _, key in SNR_COLUMNS},
        })
    class_fields = ["label", *(key for _, key in CLASS_COLUMNS)]
    with (output_dir / "per_class_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=class_fields)
        writer.writeheader()
        for label, values in metrics["per_class"].items():
            writer.writerow({"label": label, **values})


def train(config: dict[str, Any], device: torch.device) -> None:
    dataset_config = config["dataset"]
    train_data, train_loader = make_loader(config, dataset_config["train_split"], shuffle=True)
    _, validation_loader = make_loader(config, dataset_config["validation_split"], shuffle=False)
    model = build_model(config, len(train_data.labels), device)
    training = config["training"]
    optimizer = torch.optim.AdamW(model.parameters(), lr=training["learning_rate"], weight_decay=training["weight_decay"])
    scheduler = build_scheduler(optimizer, config, steps_per_epoch=len(train_loader))
    cross_entropy = nn.CrossEntropyLoss()
    checkpoint = Path(training["checkpoint"])
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    best_f1 = float("-inf")
    stale_epochs = 0
    history: list[dict[str, Any]] = []

    for epoch in range(1, training["epochs"] + 1):
        model.train()
        classification_total = 0.0
        separation_total = 0.0
        batch_count = 0
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
            scheduler.step()
            classification_total += float(classification.detach())
            separation_total += float(separation.detach())
            batch_count += 1
        train_losses = {
            "classification": classification_total / max(1, batch_count),
            "separation": separation_total / max(1, batch_count),
        }
        metrics = evaluate(
            model, validation_loader, device, config["evaluation"]["per_snr"]
        )
        history.append({"epoch": epoch, "train_loss": train_losses, "validation": metrics})
        print(
            f"epoch={epoch} train_classification={train_losses['classification']:.4f} "
            f"train_separation={train_losses['separation']:.4f} "
            f"validation_macro_f1={metrics['test_macro_f1']:.4f} validation_accuracy={metrics['test_accuracy']:.4f}"
        )
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
    metrics = evaluate(model, loader, device, config["evaluation"]["per_snr"], payload["labels"])
    output_path = Path(config["evaluation"]["output"])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    save_report_csvs(output_path.parent, metrics)
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

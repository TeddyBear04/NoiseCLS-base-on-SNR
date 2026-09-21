"""Extract branch embeddings, train the fusion head, and report results."""

from __future__ import annotations

import argparse
import csv
import json
import random
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from dataset.embedding_cache import (
    BRANCH_DIMS,
    BRANCH_ORDER,
    DPCRN_ROOT,
    EmbeddingCacheDataset,
    _load_module,
    load_beats_branch,
    load_dpcrn_branch,
    write_cache,
)
from models.fusion_head import FusionHead
from utils.reporting import (
    label_by_snr_f1,
    markdown_table,
    metric_values,
    per_class_metrics,
    write_label_by_snr_csv,
    write_label_by_snr_heatmap,
)

ABLATION_ROWS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("beats_low", ("beats_low",)),
    ("beats_mid", ("beats_mid",)),
    ("dpcrn_high", ("dpcrn_high",)),
    ("fusion", ("beats_low", "beats_mid", "dpcrn_high")),
)


def load_config(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def validate_config(config: dict[str, Any]) -> None:
    for section in ("dataset", "branches", "cache", "model", "training", "evaluation", "runtime"):
        if section not in config:
            raise ValueError(f"Missing required config section: {section}")
    for name in BRANCH_ORDER:
        if name not in config["branches"]:
            raise ValueError(f"branches.{name} is required")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def slice_features(features: torch.Tensor, selected: tuple[str, ...]) -> torch.Tensor:
    """Take only the chosen branches' columns out of the 1792-wide cache row."""
    pieces, start = [], 0
    for name in BRANCH_ORDER:
        width = BRANCH_DIMS[name]
        if name in selected:
            pieces.append(features[:, start:start + width])
        start += width
    return torch.cat(pieces, dim=1)


def build_fusion_model(
    branch_dims: dict[str, int], config: dict[str, Any], device: torch.device
) -> FusionHead:
    model_config = config["model"]
    model = FusionHead(
        branch_dims,
        hidden=tuple(model_config["hidden"]),
        classes_num=config["experiment"]["num_classes"],
        dropout=model_config["dropout"],
    )
    return model.to(device)


# ---------------------------------------------------------------------------
# extract
# ---------------------------------------------------------------------------


def extract(config: dict[str, Any], device: torch.device) -> None:
    dataset_cfg = config["dataset"]
    cache_cfg = config["cache"]
    branches_cfg = config["branches"]

    mix_dataset = _load_module(DPCRN_ROOT / "dataset" / "mix_dataset.py", "fusion_mix_dataset")
    MixNoiseDataset = mix_dataset.MixNoiseDataset

    pretrained = Path(branches_cfg["pretrained_beats"])
    beats_low, _ = load_beats_branch(pretrained, Path(branches_cfg["beats_low"]), device)
    beats_mid, _ = load_beats_branch(pretrained, Path(branches_cfg["beats_mid"]), device)
    dpcrn_high, _ = load_dpcrn_branch(Path(branches_cfg["dpcrn_high"]), device)
    branches = {"beats_low": beats_low, "beats_mid": beats_mid, "dpcrn_high": dpcrn_high}

    out_dir = Path(cache_cfg["dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    for split in (dataset_cfg["train_split"], dataset_cfg["validation_split"], dataset_cfg["test_split"]):
        dataset = MixNoiseDataset(
            dataset_cfg["path"],
            split,
            dataset_cfg["clip_seconds"],
            dataset_cfg["sample_rate"],
            -5.0,
            20.0,
            dataset_cfg["manifest_file"],
            dataset_cfg["labels_file"],
        )
        loader = DataLoader(
            dataset,
            batch_size=cache_cfg["extract_batch_size"],
            shuffle=False,
            num_workers=cache_cfg["workers"],
        )
        write_cache(branches, loader, split, out_dir, device)

    labels_src = Path(dataset_cfg["path"]) / dataset_cfg["labels_file"]
    shutil.copy(labels_src, out_dir / "labels.txt")
    print(f"copied labels to {out_dir / 'labels.txt'}")


# ---------------------------------------------------------------------------
# train / evaluate
# ---------------------------------------------------------------------------


@torch.no_grad()
def evaluate_fusion(
    model: FusionHead, loader: DataLoader[dict[str, Any]], device: torch.device, labels: list[str]
) -> dict[str, Any]:
    """Score a loader with `model`, slicing each batch to the branches it was built with."""
    model.eval()
    selected = tuple(model.branch_dims.keys())
    targets: list[int] = []
    predictions: list[int] = []
    probabilities: list[np.ndarray] = []
    snrs: list[float] = []
    snr_predictions: list[float] = []

    for batch in loader:
        features = slice_features(batch["features"].to(device), selected)
        output = model(features)
        probability = output["logits"].softmax(dim=1).cpu().numpy()
        targets.extend(batch["target"].tolist())
        predictions.extend(output["logits"].argmax(dim=1).cpu().tolist())
        probabilities.extend(probability)
        snrs.extend(batch["snr"].tolist())
        snr_predictions.extend(output["snr"].cpu().tolist())

    target_array = np.asarray(targets)
    prediction_array = np.asarray(predictions)
    probability_array = np.asarray(probabilities)
    snr_array = np.asarray(snrs)
    snr_prediction_array = np.asarray(snr_predictions)

    overall = metric_values(target_array, prediction_array, probability_array)

    per_snr: dict[str, dict[str, Any]] = {}
    for level in sorted(set(snr_array.tolist())):
        mask = snr_array == level
        per_snr[f"{level:g}"] = {
            "samples": int(mask.sum()),
            **metric_values(target_array[mask], prediction_array[mask], probability_array[mask]),
            "snr_mae": float(np.mean(np.abs(snr_prediction_array[mask] - snr_array[mask]))),
        }

    return {
        "overall": overall,
        "samples": int(len(target_array)),
        "per_snr": per_snr,
        "snr_mae": float(np.mean(np.abs(snr_prediction_array - snr_array))),
        "per_class": per_class_metrics(target_array, prediction_array, probability_array, labels),
        "label_by_snr_f1": label_by_snr_f1(target_array, prediction_array, snr_array, labels),
    }


# (header, key) pairs mirroring DPCRN_Noise_Target/tasks/run_36.py's report
# columns, swapping its SI-SDR column for the fusion head's SNR-regression MAE.
SNR_COLUMNS = (
    ("Top-1 Accuracy", "accuracy"), ("Top-3 Accuracy", "top3_accuracy"),
    ("Balanced Acc", "balanced_accuracy"), ("Precision Macro", "precision"),
    ("Recall Macro", "recall"), ("Macro-F1", "macro_f1"), ("Micro-F1", "micro_f1"),
    ("mAP", "map"), ("Macro-AUC", "macro_auc"), ("SNR MAE", "snr_mae"),
)
CLASS_COLUMNS = (
    ("Support", "support"), ("Precision", "precision"), ("Recall", "recall"),
    ("F1", "f1"), ("AP", "ap"), ("AUC", "auc"),
)


def save_report_csvs(output_dir: Path, metrics: dict[str, Any]) -> None:
    """Write the per-SNR and per-label tables next to the JSON metrics.

    Follows DPCRN_Noise_Target/tasks/run_36.py:save_report_csvs's column layout.
    """
    snr_fields = ["snr_db", "samples", *(key for _, key in SNR_COLUMNS)]
    with (output_dir / "snr_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=snr_fields)
        writer.writeheader()
        for snr, values in metrics["per_snr"].items():
            writer.writerow({"snr_db": snr, **{key: values.get(key) for key in snr_fields[1:]}})
        writer.writerow({
            "snr_db": "All", "samples": metrics["samples"],
            **{key: metrics["overall"].get(key) for _, key in SNR_COLUMNS if key != "snr_mae"},
            "snr_mae": metrics["snr_mae"],
        })
    class_fields = ["label", *(key for _, key in CLASS_COLUMNS)]
    with (output_dir / "per_class_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=class_fields)
        writer.writeheader()
        for label, values in metrics["per_class"].items():
            writer.writerow({"label": label, **values})


def train_fusion(
    config: dict[str, Any], cache_dir: str | Path, branch_dims: dict[str, int], device: torch.device
) -> dict[str, Any]:
    cache_dir = Path(cache_dir)
    dataset_cfg = config["dataset"]
    training_cfg = config["training"]

    train_dataset = EmbeddingCacheDataset(cache_dir, dataset_cfg["train_split"])
    validation_dataset = EmbeddingCacheDataset(cache_dir, dataset_cfg["validation_split"])
    train_loader = DataLoader(train_dataset, batch_size=training_cfg["batch_size"], shuffle=True)
    validation_loader = DataLoader(
        validation_dataset, batch_size=training_cfg["batch_size"], shuffle=False
    )

    selected = tuple(branch_dims.keys())
    model = build_fusion_model(branch_dims, config, device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=training_cfg["learning_rate"], weight_decay=training_cfg["weight_decay"]
    )
    cross_entropy = nn.CrossEntropyLoss()
    smooth_l1 = nn.SmoothL1Loss()
    lambda_snr = training_cfg["lambda_snr"]

    best_metric = float("-inf")
    best_state: dict[str, torch.Tensor] | None = None
    best_epoch = 0
    best_validation_metrics: dict[str, Any] = {}
    stale_epochs = 0
    history: list[dict[str, Any]] = []

    for epoch in range(1, training_cfg["epochs"] + 1):
        model.train()
        for batch in train_loader:
            features = slice_features(batch["features"].to(device), selected)
            target = batch["target"].to(device)
            snr_true = batch["snr"].to(device)
            output = model(features)
            loss = cross_entropy(output["logits"], target) + lambda_snr * smooth_l1(
                output["snr"], snr_true
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

        validation_metrics = evaluate_fusion(model, validation_loader, device, train_dataset.labels)
        selection = validation_metrics["overall"][training_cfg["selection_metric"]]
        history.append({
            "epoch": epoch,
            "validation_macro_f1": validation_metrics["overall"]["macro_f1"],
            "validation_accuracy": validation_metrics["overall"]["accuracy"],
        })
        print(f"epoch={epoch} validation_macro_f1={selection:.4f}")

        if selection > best_metric:
            best_metric = selection
            best_state = {key: value.detach().clone() for key, value in model.state_dict().items()}
            best_epoch = epoch
            best_validation_metrics = validation_metrics
            stale_epochs = 0
        else:
            stale_epochs += 1
            if stale_epochs >= training_cfg["patience"]:
                print(f"early_stop={epoch}")
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    return {
        "model": model,
        "history": history,
        "best_epoch": best_epoch,
        "best_metric": best_metric,
        "validation_metrics": best_validation_metrics,
        "labels": train_dataset.labels,
        "branch_dims": dict(branch_dims),
    }


# ---------------------------------------------------------------------------
# commands
# ---------------------------------------------------------------------------


def train36(config: dict[str, Any], device: torch.device) -> None:
    cache_dir = Path(config["cache"]["dir"])
    result = train_fusion(config, cache_dir, dict(BRANCH_DIMS), device)

    checkpoint_path = Path(config["training"]["checkpoint"])
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": result["model"].state_dict(),
            "branch_dims": result["branch_dims"],
            "labels": result["labels"],
            "config": config,
            "best_epoch": result["best_epoch"],
            "validation_metrics": result["validation_metrics"],
        },
        checkpoint_path,
    )

    results_path = Path(config["training"]["results"])
    results_path.parent.mkdir(parents=True, exist_ok=True)
    results_path.write_text(
        json.dumps(
            {"best_epoch": result["best_epoch"], "best_metric": result["best_metric"], "history": result["history"]},
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"best_epoch={result['best_epoch']} best_macro_f1={result['best_metric']:.4f}")
    print(f"Saved checkpoint to {checkpoint_path}")


def test36(config: dict[str, Any], checkpoint_path: str | Path, device: torch.device) -> None:
    payload = torch.load(checkpoint_path, map_location=device, weights_only=False)
    cache_dir = Path(config["cache"]["dir"])
    test_dataset = EmbeddingCacheDataset(cache_dir, config["dataset"]["test_split"])
    loader = DataLoader(test_dataset, batch_size=config["training"]["batch_size"], shuffle=False)

    model = build_fusion_model(payload["branch_dims"], config, device)
    model.load_state_dict(payload["model"])
    metrics = evaluate_fusion(model, loader, device, payload["labels"])

    output_path = Path(config["evaluation"]["output"])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    output_dir = Path(config["evaluation"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    write_label_by_snr_csv(output_dir / "label_by_snr_f1.csv", metrics["label_by_snr_f1"])
    write_label_by_snr_heatmap(output_dir / "label_by_snr_f1.png", metrics["label_by_snr_f1"])
    save_report_csvs(output_dir, metrics)

    rows = [[key, f"{value:.4f}" if isinstance(value, float) else str(value)] for key, value in metrics["overall"].items()]
    print(markdown_table(["Metric", "Value"], rows))
    print(f"Saved metrics to {output_path}")


def ablation(config: dict[str, Any], device: torch.device) -> None:
    cache_dir = Path(config["cache"]["dir"])
    fieldnames = ["name", "branches", "accuracy", "macro_f1", "top3_accuracy", "balanced_accuracy"]
    rows: list[dict[str, Any]] = []

    for name, selected in ABLATION_ROWS:
        # Reseed before every row: train_fusion never reseeds, so without this
        # each row starts from whatever RNG state the previous row left behind
        # (weight init and batch shuffling both drift). With a fresh seed here,
        # the "fusion" row reproduces train36's checkpoint exactly instead of
        # publishing a second, unexplained accuracy for the same configuration.
        set_seed(config["experiment"]["seed"])
        branch_dims = {branch: BRANCH_DIMS[branch] for branch in BRANCH_ORDER if branch in selected}
        result = train_fusion(config, cache_dir, branch_dims, device)

        test_dataset = EmbeddingCacheDataset(cache_dir, config["dataset"]["test_split"])
        loader = DataLoader(test_dataset, batch_size=config["training"]["batch_size"], shuffle=False)
        metrics = evaluate_fusion(result["model"], loader, device, result["labels"])

        row = {
            "name": name,
            "branches": "+".join(selected),
            "accuracy": metrics["overall"]["accuracy"],
            "macro_f1": metrics["overall"]["macro_f1"],
            "top3_accuracy": metrics["overall"]["top3_accuracy"],
            "balanced_accuracy": metrics["overall"]["balanced_accuracy"],
        }
        rows.append(row)
        print(f"ablation row={name} accuracy={row['accuracy']:.4f} macro_f1={row['macro_f1']:.4f}")

    baseline = config["evaluation"]["baseline"]
    rows.append({
        "name": "reference_baseline",
        "branches": "-",
        "accuracy": baseline["accuracy"],
        "macro_f1": baseline["macro_f1"],
        "top3_accuracy": float("nan"),  # not reported by the baseline's source metrics file
        "balanced_accuracy": baseline["balanced_accuracy"],
    })

    output_path = Path(config["evaluation"]["ablation_output"])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved ablation to {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("extract", "train36", "test36", "ablation"))
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
    if args.command == "extract":
        extract(config, device)
    elif args.command == "train36":
        train36(config, device)
    elif args.command == "test36":
        test36(config, checkpoint, device)
    else:
        ablation(config, device)


if __name__ == "__main__":
    main()

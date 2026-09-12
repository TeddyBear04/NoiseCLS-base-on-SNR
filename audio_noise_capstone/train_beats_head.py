"""Train a 36-class head on frozen BEATs embeddings from ``mix-dataset``."""

from __future__ import annotations

import argparse
import csv
import gc
import json
import random
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import f1_score, precision_recall_fscore_support
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from config.paths import BEATS_CHECKPOINT
from models.beats_loader import load_beats_classes
from noise_pipeline.mix_data import MixNoiseDataset, load_mix_manifest

DEFAULT_CACHE_DIR = Path("benchmark_results/full_embedding_cache")
SNRS = (-5, 0, 5, 10, 15, 20)


def balanced_rows(
    rows: list[dict[str, str]], split: str, per_class: int, seed: int
) -> list[dict[str, str]]:
    groups: dict[tuple[str, int], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        if row["split"] == split:
            groups[(row["label_names"], int(float(row["target_snr_db"])))].append(
                row
            )
    labels = sorted({label for label, _ in groups})
    if per_class % len(SNRS):
        raise ValueError(f"per_class must be divisible by {len(SNRS)}")
    per_group = per_class // len(SNRS)
    rng = random.Random(seed + (0 if split == "train" else 10_000))
    selected = []
    for label in labels:
        for snr in SNRS:
            candidates = groups[(label, snr)].copy()
            rng.shuffle(candidates)
            selected.extend(candidates[:per_group])
    rng.shuffle(selected)
    return selected


def load_beats(device: torch.device):
    if not BEATS_CHECKPOINT.exists():
        raise FileNotFoundError(BEATS_CHECKPOINT)
    BEATs, BEATsConfig = load_beats_classes()

    checkpoint = torch.load(
        BEATS_CHECKPOINT, map_location="cpu", weights_only=True
    )
    model = BEATs(BEATsConfig(checkpoint["cfg"]))
    model.load_state_dict(checkpoint["model"])
    model.predictor = None
    model.to(device).eval()
    for parameter in model.parameters():
        parameter.requires_grad = False
    return model, checkpoint


@torch.inference_mode()
def extract_split(
    model,
    dataset: MixNoiseDataset,
    device: torch.device,
    batch_size: int,
    workers: int,
):
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=workers,
        pin_memory=device.type == "cuda",
        persistent_workers=workers > 0,
    )
    embeddings, targets, snrs = [], [], []
    started = time.perf_counter()
    for batch_index, batch in enumerate(loader, start=1):
        sequence, _ = model.extract_features(
            batch["mixture"].to(device, non_blocking=True)
        )
        embeddings.append(sequence.mean(dim=1).float().cpu())
        targets.append(batch["target"].long())
        snrs.append(batch["snr"].long())
        if batch_index == 1 or batch_index % 100 == 0 or batch_index == len(loader):
            print(
                f"extract split={dataset.split} batch={batch_index}/{len(loader)}",
                flush=True,
            )
    return {
        "x": torch.cat(embeddings),
        "y": torch.cat(targets),
        "snr": torch.cat(snrs),
        "seconds": time.perf_counter() - started,
    }


def cache_name(split: str, per_class: int | None) -> str:
    suffix = "full" if per_class is None else f"pc{per_class}"
    return f"beats_{split}_{suffix}_seed2026.pt"


def load_or_extract(
    split: str,
    root: Path,
    rows: list[dict[str, str]],
    per_class: int | None,
    model,
    device: torch.device,
    cache_dir: Path,
    batch_size: int,
    workers: int,
    force: bool,
):
    cache_path = cache_dir / cache_name(split, per_class)
    if cache_path.exists() and not force:
        print(f"cache split={split} path={cache_path}", flush=True)
        return torch.load(cache_path, map_location="cpu", weights_only=True)
    selected = (
        rows
        if per_class is None
        else balanced_rows(rows, split, per_class, seed=2026)
    )
    dataset = MixNoiseDataset(root, split, rows=selected)
    data = extract_split(model, dataset, device, batch_size, workers)
    cache_dir.mkdir(parents=True, exist_ok=True)
    torch.save(data, cache_path)
    print(
        f"cached split={split} samples={len(dataset)} seconds={data['seconds']:.1f} "
        f"path={cache_path}",
        flush=True,
    )
    return data


def metrics(logits: torch.Tensor, targets: torch.Tensor, snrs: torch.Tensor, labels):
    predictions = logits.argmax(dim=1).cpu().numpy()
    expected = targets.cpu().numpy()
    precision, recall, class_f1, support = precision_recall_fscore_support(
        expected,
        predictions,
        labels=np.arange(len(labels)),
        zero_division=0,
    )
    result = {
        "accuracy": float((predictions == expected).mean()),
        "precision": float(precision.mean()),
        "recall": float(recall.mean()),
        "macro_f1": float(class_f1.mean()),
        "micro_f1": float(f1_score(expected, predictions, average="micro", zero_division=0)),
        "per_class": {
            label: {
                "f1": float(class_f1[index]),
                "support": int(support[index]),
            }
            for index, label in enumerate(labels)
        },
        "per_snr": {},
    }
    snr_values = snrs.cpu().numpy()
    for snr in SNRS:
        mask = snr_values == snr
        _, _, snr_f1, _ = precision_recall_fscore_support(
            expected[mask],
            predictions[mask],
            labels=np.arange(len(labels)),
            zero_division=0,
        )
        result["per_snr"][str(snr)] = {
            "samples": int(mask.sum()),
            "accuracy": float((predictions[mask] == expected[mask]).mean()),
            "macro_f1": float(snr_f1.mean()),
            "micro_f1": float(f1_score(expected[mask], predictions[mask], average="micro", zero_division=0)),
        }
    return result


def initialize_head(checkpoint, labels, label_to_mid, device):
    head = nn.Linear(768, len(labels)).to(device)
    mid_to_source = {
        mid: int(index) for index, mid in checkpoint["label_dict"].items()
    }
    source_indices = [mid_to_source[label_to_mid[label]] for label in labels]
    with torch.no_grad():
        source_weight = checkpoint["model"]["predictor.weight"]
        source_bias = checkpoint["model"]["predictor.bias"]
        head.weight.copy_(source_weight[source_indices])
        head.bias.copy_(source_bias[source_indices])
    return head


@torch.inference_mode()
def predict(head, data, device, batch_size=2048):
    outputs = []
    for start in range(0, len(data["x"]), batch_size):
        outputs.append(head(data["x"][start : start + batch_size].to(device)).cpu())
    return torch.cat(outputs)


def train_head(
    head,
    train_data,
    validation_data,
    labels,
    device,
    epochs,
    batch_size,
    learning_rate,
    patience,
):
    generator = torch.Generator().manual_seed(2026)
    loader = DataLoader(
        TensorDataset(train_data["x"], train_data["y"]),
        batch_size=batch_size,
        shuffle=True,
        generator=generator,
        pin_memory=device.type == "cuda",
    )
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        head.parameters(), lr=learning_rate, weight_decay=1e-4
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=2, min_lr=1e-6
    )

    initial_logits = predict(head, validation_data, device)
    best_metrics = metrics(
        initial_logits, validation_data["y"], validation_data["snr"], labels
    )
    best_state = {key: value.detach().cpu().clone() for key, value in head.state_dict().items()}
    best_epoch = 0
    stale = 0
    history = [{"epoch": 0, "validation": best_metrics}]
    print(
        f"epoch=0 val_accuracy={best_metrics['accuracy']:.4f} "
        f"val_macro_f1={best_metrics['macro_f1']:.4f}",
        flush=True,
    )

    for epoch in range(1, epochs + 1):
        head.train()
        total_loss = 0.0
        correct = 0
        seen = 0
        for features, targets in loader:
            features = features.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            logits = head(features)
            loss = criterion(logits, targets)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * targets.shape[0]
            correct += (logits.argmax(dim=1) == targets).sum().item()
            seen += targets.shape[0]

        head.eval()
        validation_logits = predict(head, validation_data, device)
        validation_metrics = metrics(
            validation_logits,
            validation_data["y"],
            validation_data["snr"],
            labels,
        )
        scheduler.step(validation_metrics["macro_f1"])
        epoch_result = {
            "epoch": epoch,
            "train_loss": total_loss / seen,
            "train_accuracy": correct / seen,
            "learning_rate": optimizer.param_groups[0]["lr"],
            "validation": validation_metrics,
        }
        history.append(epoch_result)
        print(
            f"epoch={epoch} train_loss={epoch_result['train_loss']:.4f} "
            f"train_accuracy={epoch_result['train_accuracy']:.4f} "
            f"val_accuracy={validation_metrics['accuracy']:.4f} "
            f"val_macro_f1={validation_metrics['macro_f1']:.4f} "
            f"lr={epoch_result['learning_rate']:.2e}",
            flush=True,
        )
        if validation_metrics["macro_f1"] > best_metrics["macro_f1"] + 1e-4:
            best_metrics = validation_metrics
            best_epoch = epoch
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in head.state_dict().items()
            }
            stale = 0
        else:
            stale += 1
            if stale >= patience:
                print(f"early_stop={epoch}", flush=True)
                break
    head.load_state_dict(best_state)
    return best_epoch, best_metrics, history


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("mix-dataset"))
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument("--output", type=Path, default=Path("checkpoints/beats_head.pt"))
    parser.add_argument(
        "--results", type=Path,
        default=Path("benchmark_results/beats_frozen_full.json"),
    )
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--extract-batch-size", type=int, default=16)
    parser.add_argument("--head-batch-size", type=int, default=512)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--train-per-class", type=int)
    parser.add_argument("--validation-per-class", type=int)
    parser.add_argument("--force-extract", action="store_true")
    parser.add_argument("--smoke-test", action="store_true")
    args = parser.parse_args()

    random.seed(2026)
    np.random.seed(2026)
    torch.manual_seed(2026)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.smoke_test:
        args.train_per_class = 6
        args.validation_per_class = 6
        args.epochs = 2
        args.workers = 0
        args.output = Path("checkpoints/beats_head_smoke.pt")
        args.results = Path("benchmark_results/beats_frozen_smoke.json")

    labels = (args.data_root / "labels.txt").read_text(encoding="utf-8").splitlines()
    rows = load_mix_manifest(args.data_root)
    label_to_mid = {}
    for row in rows:
        label_to_mid.setdefault(row["label_names"], row["label_mids"])

    print(f"device={device} labels={len(labels)}", flush=True)
    model, checkpoint = load_beats(device)
    train_data = load_or_extract(
        "train", args.data_root, rows, args.train_per_class, model, device,
        args.cache_dir, args.extract_batch_size, args.workers, args.force_extract,
    )
    validation_data = load_or_extract(
        "validation", args.data_root, rows, args.validation_per_class, model,
        device, args.cache_dir, args.extract_batch_size, args.workers,
        args.force_extract,
    )
    del model
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()

    head = initialize_head(checkpoint, labels, label_to_mid, device)
    best_epoch, best_metrics, history = train_head(
        head,
        train_data,
        validation_data,
        labels,
        device,
        args.epochs,
        args.head_batch_size,
        args.lr,
        args.patience,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    serializable_args = {
        key: str(value) if isinstance(value, Path) else value
        for key, value in vars(args).items()
    }
    torch.save(
        {
            "head": {key: value.cpu() for key, value in head.state_dict().items()},
            "labels": labels,
            "encoder": "BEATs_iter3_plus_AS2M_finetuned_cpt2",
            "args": serializable_args,
            "best_epoch": best_epoch,
            "validation_metrics": best_metrics,
        },
        args.output,
    )
    result = {
        "stage": "frozen_beats_full_head",
        "train_samples": len(train_data["y"]),
        "validation_samples": len(validation_data["y"]),
        "best_epoch": best_epoch,
        "best_validation": best_metrics,
        "embedding_extraction_seconds": {
            "train": float(train_data.get("seconds", 0.0)),
            "validation": float(validation_data.get("seconds", 0.0)),
        },
        "history": history,
        "checkpoint": str(args.output),
        "args": serializable_args,
    }
    args.results.parent.mkdir(parents=True, exist_ok=True)
    args.results.write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(
        f"best_epoch={best_epoch} val_accuracy={best_metrics['accuracy']:.4f} "
        f"val_macro_f1={best_metrics['macro_f1']:.4f}",
        flush=True,
    )
    print(f"checkpoint={args.output} results={args.results}", flush=True)


if __name__ == "__main__":
    main()

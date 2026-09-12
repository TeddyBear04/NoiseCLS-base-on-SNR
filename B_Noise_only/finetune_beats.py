"""Fine-tune the final BEATs transformer blocks on oracle-noise waveforms."""

from __future__ import annotations

import argparse
import gc
import json
import random
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from models.beats_loader import load_beats_classes
from utils.reporting36 import save_evaluation_artifacts, save_training_artifacts

from noise_pipeline.mix_data import load_float_audio, load_mix_manifest
from train_beats_head import BEATS_CHECKPOINT, balanced_rows, metrics


def mixed_precision_context(device: torch.device):
    """Use BF16 when available; fall back to numerically stable FP32."""
    return torch.autocast(
        device_type=device.type,
        dtype=torch.bfloat16,
        enabled=device.type == "cuda" and torch.cuda.is_bf16_supported(),
    )


def require_finite(value: torch.Tensor, name: str) -> None:
    """Fail immediately instead of saving a checkpoint with NaN weights."""
    if not torch.isfinite(value).all():
        raise FloatingPointError(f"Non-finite {name}; aborting training.")


class NoiseDataset(Dataset):
    """Noise-only view over the manifest."""

    def __init__(self, root: Path, split: str, rows: list[dict[str, str]]):
        self.root = root
        self.rows = [row for row in rows if row["split"] == split]
        self.labels = (root / "labels.txt").read_text(encoding="utf-8").splitlines()
        self.label_to_index = {
            label: index for index, label in enumerate(self.labels)
        }

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.rows[index]
        return {
            "waveform": load_float_audio(
                self.root / row["noise_path"], 16_000, 4 * 16_000
            ),
            "target": self.label_to_index[row["label_names"]],
            "snr": int(float(row["target_snr_db"])),
        }


def load_model(device: torch.device, head_checkpoint: Path, trainable_blocks: int):
    BEATs, BEATsConfig = load_beats_classes()

    source = torch.load(BEATS_CHECKPOINT, map_location="cpu", weights_only=True)
    model = BEATs(BEATsConfig(source["cfg"]))
    model.load_state_dict(source["model"])
    model.predictor = None
    for parameter in model.parameters():
        parameter.requires_grad = False
    for layer in model.encoder.layers[-trainable_blocks:]:
        for parameter in layer.parameters():
            parameter.requires_grad = True

    saved_head = torch.load(
        head_checkpoint, map_location="cpu", weights_only=True
    )
    labels = saved_head["labels"]
    head = nn.Linear(768, len(labels))
    head.load_state_dict(saved_head["head"])
    model.to(device)
    head.to(device)
    return model, head, labels


def set_training_mode(model, head, trainable_blocks: int) -> None:
    # Frozen blocks remain deterministic; dropout is active only in blocks that
    # receive gradient updates.
    model.eval()
    for layer in model.encoder.layers[-trainable_blocks:]:
        layer.train()
    head.train()


def make_loader(
    dataset: Dataset,
    batch_size: int,
    workers: int,
    shuffle: bool,
    device: torch.device,
):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        generator=torch.Generator().manual_seed(2026) if shuffle else None,
        num_workers=workers,
        pin_memory=device.type == "cuda",
        persistent_workers=workers > 0,
    )


@torch.inference_mode()
def evaluate(model, head, loader, device, labels, return_predictions: bool = False):
    model.eval()
    head.eval()
    all_logits, all_targets, all_snrs = [], [], []
    loss_total = 0.0
    criterion = nn.CrossEntropyLoss()
    for batch in loader:
        waveforms = batch["waveform"].to(device, non_blocking=True)
        targets = batch["target"].to(device, non_blocking=True)
        with mixed_precision_context(device):
            sequence, _ = model.extract_features(waveforms)
            logits = head(sequence.mean(dim=1))
            loss = criterion(logits, targets)
        require_finite(loss, "validation loss")
        loss_total += loss.item() * targets.shape[0]
        all_logits.append(logits.float().cpu())
        all_targets.append(targets.cpu())
        all_snrs.append(batch["snr"].cpu())
    result = metrics(
        torch.cat(all_logits),
        torch.cat(all_targets),
        torch.cat(all_snrs),
        labels,
    )
    result["loss"] = loss_total / len(loader.dataset)
    if return_predictions:
        logits = torch.cat(all_logits)
        return (
            result,
            torch.cat(all_targets).numpy(),
            logits.argmax(dim=1).numpy(),
        )
    return result


def train_epoch(
    model,
    head,
    loader,
    optimizer,
    device,
    trainable_blocks,
    accumulation_steps,
):
    set_training_mode(model, head, trainable_blocks)
    criterion = nn.CrossEntropyLoss()
    optimizer.zero_grad(set_to_none=True)
    total_loss = 0.0
    correct = 0
    seen = 0
    trainable_parameters = [
        parameter
        for parameter in list(model.parameters()) + list(head.parameters())
        if parameter.requires_grad
    ]
    for batch_index, batch in enumerate(loader, start=1):
        waveforms = batch["waveform"].to(device, non_blocking=True)
        targets = batch["target"].to(device, non_blocking=True)
        with mixed_precision_context(device):
            sequence, _ = model.extract_features(waveforms)
            logits = head(sequence.mean(dim=1))
            loss = criterion(logits, targets)
            scaled_loss = loss / accumulation_steps
        require_finite(loss, "training loss")
        scaled_loss.backward()
        if batch_index % accumulation_steps == 0 or batch_index == len(loader):
            nn.utils.clip_grad_norm_(trainable_parameters, 5.0)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
        total_loss += loss.item() * targets.shape[0]
        correct += (logits.argmax(dim=1) == targets).sum().item()
        seen += targets.shape[0]
        if batch_index == 1 or batch_index % 200 == 0 or batch_index == len(loader):
            memory = (
                torch.cuda.max_memory_allocated() / 1024**3
                if device.type == "cuda"
                else 0.0
            )
            print(
                f"train batch={batch_index}/{len(loader)} "
                f"loss={total_loss/seen:.4f} accuracy={correct/seen:.4f} "
                f"max_vram_gb={memory:.2f}",
                flush=True,
            )
    return {"loss": total_loss / seen, "accuracy": correct / seen}


def trainable_encoder_state(model, trainable_blocks: int):
    prefixes = tuple(
        f"encoder.layers.{index}."
        for index in range(
            len(model.encoder.layers) - trainable_blocks,
            len(model.encoder.layers),
        )
    )
    return {
        key: value.detach().cpu().clone()
        for key, value in model.state_dict().items()
        if key.startswith(prefixes)
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("../../36_labels"))
    parser.add_argument(
        "--head-checkpoint", type=Path, default=Path("checkpoint/beats_head_36_noise.pt")
    )
    parser.add_argument(
        "--output", type=Path, default=Path("checkpoint/audio_best_36_noise.pt")
    )
    parser.add_argument(
        "--results", type=Path,
        default=Path("checkpoint/summary_36_noise.json"),
    )
    parser.add_argument("--trainable-blocks", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--validation-batch-size", type=int, default=32)
    parser.add_argument("--accumulation-steps", type=int, default=2)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--head-lr", type=float, default=1e-4)
    parser.add_argument("--encoder-lr", type=float, default=1e-5)
    parser.add_argument("--train-per-class", type=int)
    parser.add_argument("--validation-per-class", type=int)
    parser.add_argument("--smoke-test", action="store_true")
    args = parser.parse_args()

    random.seed(2026)
    np.random.seed(2026)
    torch.manual_seed(2026)
    torch.backends.cuda.matmul.allow_tf32 = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.smoke_test:
        args.train_per_class = 6
        args.validation_per_class = 6
        args.epochs = 1
        args.patience = 1
        args.batch_size = 4
        args.validation_batch_size = 8
        args.accumulation_steps = 2
        args.workers = 0
        args.output = Path("checkpoint/audio_best_36_noise_smoke.pt")
        args.results = Path("checkpoint/summary_36_noise_smoke.json")

    rows = load_mix_manifest(args.data_root)
    train_rows = (
        rows
        if args.train_per_class is None
        else balanced_rows(rows, "train", args.train_per_class, 2026)
    )
    validation_rows = (
        rows
        if args.validation_per_class is None
        else balanced_rows(
            rows, "validation", args.validation_per_class, 2026
        )
    )
    train_dataset = NoiseDataset(args.data_root, "train", train_rows)
    validation_dataset = NoiseDataset(
        args.data_root, "validation", validation_rows
    )
    model, head, labels = load_model(
        device, args.head_checkpoint, args.trainable_blocks
    )
    train_loader = make_loader(
        train_dataset, args.batch_size, args.workers, True, device
    )
    validation_loader = make_loader(
        validation_dataset,
        args.validation_batch_size,
        args.workers,
        False,
        device,
    )
    encoder_parameters = [
        parameter for parameter in model.parameters() if parameter.requires_grad
    ]
    optimizer = torch.optim.AdamW(
        [
            {"params": encoder_parameters, "lr": args.encoder_lr},
            {"params": head.parameters(), "lr": args.head_lr},
        ],
        weight_decay=1e-4,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=1, min_lr=1e-7
    )
    print(
        f"device={device} train={len(train_dataset)} "
        f"validation={len(validation_dataset)} "
        f"trainable_encoder_params={sum(p.numel() for p in encoder_parameters)}",
        flush=True,
    )
    initial = evaluate(model, head, validation_loader, device, labels)
    print(
        f"epoch=0 val_loss={initial['loss']:.4f} "
        f"val_accuracy={initial['accuracy']:.4f} "
        f"val_macro_f1={initial['macro_f1']:.4f}",
        flush=True,
    )
    best_metrics = initial
    best_epoch = 0
    best_head = {
        key: value.detach().cpu().clone() for key, value in head.state_dict().items()
    }
    best_encoder = trainable_encoder_state(model, args.trainable_blocks)
    history = [{"epoch": 0, "validation": initial}]
    stale = 0
    started = time.perf_counter()

    for epoch in range(1, args.epochs + 1):
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats()
        train_result = train_epoch(
            model,
            head,
            train_loader,
            optimizer,
            device,
            args.trainable_blocks,
            args.accumulation_steps,
        )
        validation_result = evaluate(
            model, head, validation_loader, device, labels
        )
        scheduler.step(validation_result["macro_f1"])
        epoch_result = {
            "epoch": epoch,
            "train": train_result,
            "validation": validation_result,
            "encoder_lr": optimizer.param_groups[0]["lr"],
            "head_lr": optimizer.param_groups[1]["lr"],
        }
        history.append(epoch_result)
        print(
            f"epoch={epoch} train_loss={train_result['loss']:.4f} "
            f"train_accuracy={train_result['accuracy']:.4f} "
            f"val_loss={validation_result['loss']:.4f} "
            f"val_accuracy={validation_result['accuracy']:.4f} "
            f"val_macro_f1={validation_result['macro_f1']:.4f}",
            flush=True,
        )
        if validation_result["macro_f1"] > best_metrics["macro_f1"] + 1e-4:
            best_metrics = validation_result
            best_epoch = epoch
            best_head = {
                key: value.detach().cpu().clone()
                for key, value in head.state_dict().items()
            }
            best_encoder = trainable_encoder_state(
                model, args.trainable_blocks
            )
            stale = 0
        else:
            stale += 1
            if stale >= args.patience:
                print(f"early_stop={epoch}", flush=True)
                break

    args.output.parent.mkdir(parents=True, exist_ok=True)
    serializable_args = {
        key: str(value) if isinstance(value, Path) else value
        for key, value in vars(args).items()
    }
    torch.save(
        {
            "encoder_delta": best_encoder,
            "head": best_head,
            "labels": labels,
            "base_encoder_checkpoint": str(BEATS_CHECKPOINT),
            "base_head_checkpoint": str(args.head_checkpoint),
            "trainable_blocks": args.trainable_blocks,
            "best_epoch": best_epoch,
            "validation_metrics": best_metrics,
            "args": serializable_args,
        },
        args.output,
    )
    result = {
        "stage": "beats_noise_only_partial_finetune",
        "input_kind": "noise",
        "train_samples": len(train_dataset),
        "validation_samples": len(validation_dataset),
        "trainable_encoder_parameters": sum(
            parameter.numel() for parameter in encoder_parameters
        ),
        "best_epoch": best_epoch,
        "best_validation": best_metrics,
        "history": history,
        "elapsed_seconds": time.perf_counter() - started,
        "checkpoint": str(args.output),
        "args": serializable_args,
    }
    args.results.parent.mkdir(parents=True, exist_ok=True)
    args.results.write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    save_training_artifacts(args.output.parent, labels, result)
    model.load_state_dict(best_encoder, strict=False)
    head.load_state_dict(best_head)
    best_validation, expected, predicted = evaluate(
        model,
        head,
        validation_loader,
        device,
        labels,
        return_predictions=True,
    )
    save_evaluation_artifacts(
        args.output.parent,
        best_validation,
        labels,
        expected,
        predicted,
        split="validation",
    )
    print(
        f"best_epoch={best_epoch} val_accuracy={best_metrics['accuracy']:.4f} "
        f"val_macro_f1={best_metrics['macro_f1']:.4f}",
        flush=True,
    )
    print(f"checkpoint={args.output} results={args.results}", flush=True)

    del model, head
    gc.collect()


if __name__ == "__main__":
    main()

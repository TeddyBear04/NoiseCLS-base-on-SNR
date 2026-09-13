"""Jointly fine-tune separator, last BEATs blocks and fusion head.

Total loss: L = L_class + lambda * L_sep, where L_sep is negative SI-SDR between
the separator output and ``oracle_noise`` (training only).
"""

from __future__ import annotations

import argparse
import gc
import json
import time
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader

from config.paths import BEATS_CHECKPOINT
from models.fusion import FusionClassifier, encode_branches
from models.separator import build_separator, separation_loss, separator_payload, si_sdr
from noise_pipeline.mix_data import MixNoiseDataset, load_mix_manifest
from utils.reporting36 import save_evaluation_artifacts, save_training_artifacts
from utils.training36 import (
    SEED,
    add_si_sdr_metrics,
    load_beats,
    metrics,
    mixed_precision_context,
    require_finite,
    seed_everything,
    select_rows,
)


def load_model(device: torch.device, checkpoint: dict, trainable_blocks: int, dropout: float = 0.1):
    """Build encoder, separator and classifier from a head or fine-tuned checkpoint."""
    encoder, _ = load_beats(device)
    for layer in encoder.encoder.layers[-trainable_blocks:]:
        for parameter in layer.parameters():
            parameter.requires_grad = True
    if "encoder_delta" in checkpoint:
        encoder.load_state_dict(checkpoint["encoder_delta"], strict=False)
    separator = build_separator(checkpoint["separator"]).to(device)
    labels = checkpoint["labels"]
    classifier = FusionClassifier(len(labels), dropout=dropout)
    classifier.load_state_dict(checkpoint["classifier"])
    classifier.to(device)
    return encoder, separator, classifier, labels


def make_loader(dataset, batch_size: int, workers: int, shuffle: bool, device: torch.device):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        generator=torch.Generator().manual_seed(SEED) if shuffle else None,
        num_workers=workers,
        pin_memory=device.type == "cuda",
        persistent_workers=workers > 0,
    )


def set_training_mode(encoder, separator, classifier, trainable_blocks: int, train_separator: bool) -> None:
    # Frozen blocks remain deterministic; dropout is active only in blocks that
    # receive gradient updates.
    encoder.eval()
    for layer in encoder.encoder.layers[-trainable_blocks:]:
        layer.train()
    separator.train(train_separator)
    classifier.train()


@torch.inference_mode()
def evaluate(encoder, separator, classifier, loader, device, labels, sep_loss_weight: float,
             return_predictions: bool = False):
    encoder.eval()
    separator.eval()
    classifier.eval()
    criterion = nn.CrossEntropyLoss(reduction="sum")
    all_logits, all_targets, all_snrs, all_si_sdrs = [], [], [], []
    class_loss_total = 0.0
    for batch in loader:
        mixture = batch["mixture"].to(device, non_blocking=True)
        targets = batch["target"].to(device, non_blocking=True)
        oracle = batch["oracle_noise"].to(device, non_blocking=True)
        z_mix, z_noise, noise = encode_branches(encoder, separator, mixture, device)
        with mixed_precision_context(device):
            logits = classifier(z_mix, z_noise)
            loss = criterion(logits, targets)
        require_finite(loss, "validation loss")
        class_loss_total += loss.item()
        all_logits.append(logits.float().cpu())
        all_targets.append(targets.cpu())
        all_snrs.append(batch["snr"].cpu())
        all_si_sdrs.append(si_sdr(noise, oracle).cpu())
    logits = torch.cat(all_logits)
    targets = torch.cat(all_targets)
    snrs = torch.cat(all_snrs)
    si_sdrs = torch.cat(all_si_sdrs)
    result = add_si_sdr_metrics(metrics(logits, targets, snrs, labels), si_sdrs, snrs)
    result["class_loss"] = class_loss_total / len(loader.dataset)
    result["separation_loss"] = -result["si_sdr"]
    result["loss"] = result["class_loss"] + sep_loss_weight * result["separation_loss"]
    if return_predictions:
        return result, targets.numpy(), logits.argmax(dim=1).numpy()
    return result


def train_epoch(encoder, separator, classifier, loader, optimizer, device, args, train_separator: bool):
    set_training_mode(encoder, separator, classifier, args.trainable_blocks, train_separator)
    criterion = nn.CrossEntropyLoss()
    optimizer.zero_grad(set_to_none=True)
    totals = {"loss": 0.0, "class_loss": 0.0, "separation_loss": 0.0}
    correct = 0
    seen = 0
    trainable_parameters = [
        parameter
        for module in (encoder, separator, classifier)
        for parameter in module.parameters()
        if parameter.requires_grad
    ]
    for batch_index, batch in enumerate(loader, start=1):
        mixture = batch["mixture"].to(device, non_blocking=True)
        targets = batch["target"].to(device, non_blocking=True)
        oracle = batch["oracle_noise"].to(device, non_blocking=True)
        z_mix, z_noise, noise = encode_branches(encoder, separator, mixture, device)
        with mixed_precision_context(device):
            logits = classifier(z_mix, z_noise)
            class_loss = criterion(logits, targets)
        sep_loss = separation_loss(noise, oracle)
        loss = class_loss + args.sep_loss_weight * sep_loss
        require_finite(loss, "training loss")
        (loss / args.accumulation_steps).backward()
        if batch_index % args.accumulation_steps == 0 or batch_index == len(loader):
            nn.utils.clip_grad_norm_(trainable_parameters, 5.0)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
        size = targets.shape[0]
        totals["loss"] += loss.item() * size
        totals["class_loss"] += class_loss.item() * size
        totals["separation_loss"] += sep_loss.item() * size
        correct += (logits.argmax(dim=1) == targets).sum().item()
        seen += size
        if batch_index == 1 or batch_index % 200 == 0 or batch_index == len(loader):
            memory = torch.cuda.max_memory_allocated() / 1024**3 if device.type == "cuda" else 0.0
            print(
                f"train batch={batch_index}/{len(loader)} loss={totals['loss']/seen:.4f} "
                f"class_loss={totals['class_loss']/seen:.4f} "
                f"sep_loss={totals['separation_loss']/seen:.4f} "
                f"accuracy={correct/seen:.4f} max_vram_gb={memory:.2f}",
                flush=True,
            )
    return {**{key: value / seen for key, value in totals.items()}, "accuracy": correct / seen}


def trainable_encoder_state(encoder, trainable_blocks: int):
    layer_count = len(encoder.encoder.layers)
    prefixes = tuple(
        f"encoder.layers.{index}." for index in range(layer_count - trainable_blocks, layer_count)
    )
    return {
        key: value.detach().cpu().clone()
        for key, value in encoder.state_dict().items()
        if key.startswith(prefixes)
    }


def snapshot(encoder, separator, classifier, trainable_blocks: int) -> dict:
    return {
        "encoder_delta": trainable_encoder_state(encoder, trainable_blocks),
        "separator": separator_payload(separator),
        "classifier": {key: value.detach().cpu().clone() for key, value in classifier.state_dict().items()},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("../../36_labels"))
    parser.add_argument("--head-checkpoint", type=Path, default=Path("checkpoint/beats_fusion_head_36.pt"))
    parser.add_argument("--output", type=Path, default=Path("checkpoint/audio_best_36_fusion.pt"))
    parser.add_argument("--results", type=Path, default=Path("checkpoint/summary_36_fusion.json"))
    parser.add_argument("--trainable-blocks", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--validation-batch-size", type=int, default=32)
    parser.add_argument("--accumulation-steps", type=int, default=2)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--head-lr", type=float, default=1e-4)
    parser.add_argument("--encoder-lr", type=float, default=1e-5)
    parser.add_argument("--separator-lr", type=float, default=5e-5,
                        help="0 keeps the pre-trained separator frozen")
    parser.add_argument("--sep-loss-weight", type=float, default=0.05,
                        help="lambda in L = L_class + lambda * L_sep")
    parser.add_argument("--train-per-class", type=int)
    parser.add_argument("--validation-per-class", type=int)
    parser.add_argument("--smoke-test", action="store_true")
    args = parser.parse_args()

    seed_everything()
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
        args.head_checkpoint = Path("checkpoint/beats_fusion_head_36_smoke.pt")
        args.output = Path("checkpoint/audio_best_36_fusion_smoke.pt")
        args.results = Path("checkpoint/summary_36_fusion_smoke.json")

    rows = load_mix_manifest(args.data_root)
    train_dataset = MixNoiseDataset(
        args.data_root, "train", rows=select_rows(rows, "train", args.train_per_class)
    )
    validation_dataset = MixNoiseDataset(
        args.data_root, "validation",
        rows=select_rows(rows, "validation", args.validation_per_class),
    )
    head_checkpoint = torch.load(args.head_checkpoint, map_location="cpu", weights_only=True)
    encoder, separator, classifier, labels = load_model(
        device, head_checkpoint, args.trainable_blocks,
        dropout=head_checkpoint["args"].get("dropout", 0.1),
    )
    train_separator = args.separator_lr > 0
    for parameter in separator.parameters():
        parameter.requires_grad = train_separator

    train_loader = make_loader(train_dataset, args.batch_size, args.workers, True, device)
    validation_loader = make_loader(
        validation_dataset, args.validation_batch_size, args.workers, False, device
    )
    encoder_parameters = [parameter for parameter in encoder.parameters() if parameter.requires_grad]
    parameter_groups = [
        {"params": encoder_parameters, "lr": args.encoder_lr},
        {"params": classifier.parameters(), "lr": args.head_lr},
    ]
    if train_separator:
        parameter_groups.append({"params": separator.parameters(), "lr": args.separator_lr})
    optimizer = torch.optim.AdamW(parameter_groups, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=1, min_lr=1e-7
    )
    print(
        f"device={device} train={len(train_dataset)} validation={len(validation_dataset)} "
        f"trainable_encoder_params={sum(p.numel() for p in encoder_parameters)} "
        f"train_separator={train_separator} sep_loss_weight={args.sep_loss_weight}",
        flush=True,
    )

    initial = evaluate(encoder, separator, classifier, validation_loader, device, labels, args.sep_loss_weight)
    print(
        f"epoch=0 val_loss={initial['loss']:.4f} val_accuracy={initial['accuracy']:.4f} "
        f"val_macro_f1={initial['macro_f1']:.4f} val_si_sdr={initial['si_sdr']:.2f}",
        flush=True,
    )
    best_metrics = initial
    best_epoch = 0
    best_state = snapshot(encoder, separator, classifier, args.trainable_blocks)
    history = [{"epoch": 0, "validation": initial}]
    stale = 0
    started = time.perf_counter()

    for epoch in range(1, args.epochs + 1):
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats()
        train_result = train_epoch(
            encoder, separator, classifier, train_loader, optimizer, device, args, train_separator
        )
        validation_result = evaluate(
            encoder, separator, classifier, validation_loader, device, labels, args.sep_loss_weight
        )
        scheduler.step(validation_result["macro_f1"])
        history.append(
            {
                "epoch": epoch,
                "train": train_result,
                "validation": validation_result,
                "learning_rates": [group["lr"] for group in optimizer.param_groups],
            }
        )
        print(
            f"epoch={epoch} train_loss={train_result['loss']:.4f} "
            f"train_accuracy={train_result['accuracy']:.4f} "
            f"val_loss={validation_result['loss']:.4f} "
            f"val_accuracy={validation_result['accuracy']:.4f} "
            f"val_macro_f1={validation_result['macro_f1']:.4f} "
            f"val_si_sdr={validation_result['si_sdr']:.2f}",
            flush=True,
        )
        if validation_result["macro_f1"] > best_metrics["macro_f1"] + 1e-4:
            best_metrics = validation_result
            best_epoch = epoch
            best_state = snapshot(encoder, separator, classifier, args.trainable_blocks)
            stale = 0
        else:
            stale += 1
            if stale >= args.patience:
                print(f"early_stop={epoch}", flush=True)
                break

    serializable_args = {
        key: str(value) if isinstance(value, Path) else value
        for key, value in vars(args).items()
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            **best_state,
            "labels": labels,
            "base_encoder_checkpoint": str(BEATS_CHECKPOINT),
            "base_head_checkpoint": str(args.head_checkpoint),
            "trainable_blocks": args.trainable_blocks,
            "dropout": head_checkpoint["args"].get("dropout", 0.1),
            "best_epoch": best_epoch,
            "validation_metrics": best_metrics,
            "args": serializable_args,
        },
        args.output,
    )
    result = {
        "stage": "beats_mixture_noise_fusion_joint_finetune",
        "input_kind": "mixture+separated_noise",
        "train_samples": len(train_dataset),
        "validation_samples": len(validation_dataset),
        "trainable_encoder_parameters": sum(parameter.numel() for parameter in encoder_parameters),
        "best_epoch": best_epoch,
        "best_validation": best_metrics,
        "history": history,
        "elapsed_seconds": time.perf_counter() - started,
        "checkpoint": str(args.output),
        "args": serializable_args,
    }
    args.results.parent.mkdir(parents=True, exist_ok=True)
    args.results.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    save_training_artifacts(args.output.parent, labels, result)

    encoder.load_state_dict(best_state["encoder_delta"], strict=False)
    separator.load_state_dict(best_state["separator"]["state"])
    classifier.load_state_dict(best_state["classifier"])
    best_validation, expected, predicted = evaluate(
        encoder, separator, classifier, validation_loader, device, labels,
        args.sep_loss_weight, return_predictions=True,
    )
    save_evaluation_artifacts(
        args.output.parent, best_validation, labels, expected, predicted, split="validation"
    )
    print(
        f"best_epoch={best_epoch} val_accuracy={best_metrics['accuracy']:.4f} "
        f"val_macro_f1={best_metrics['macro_f1']:.4f} val_si_sdr={best_metrics['si_sdr']:.2f}",
        flush=True,
    )
    print(f"checkpoint={args.output} results={args.results}", flush=True)

    del encoder, separator, classifier
    gc.collect()


if __name__ == "__main__":
    main()

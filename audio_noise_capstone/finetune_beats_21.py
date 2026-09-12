"""Fine-tune BEATs for the 21-label multi-label dataset."""

from __future__ import annotations

import argparse
import ast
import gc
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, WeightedRandomSampler

from benchmark_pretrained import BEATS_CHECKPOINT
from noise_pipeline.audio21 import Audio21Dataset, load_label_metadata, multilabel_metrics
from noise_pipeline.data import load_pcm16


class OraclePairAudio21Dataset(Audio21Dataset):
    """Training view that returns a time-aligned oracle-noise crop as teacher input."""

    def __getitem__(self, index: int):
        result = super().__getitem__(index)
        sample_id = self.rows[index]["sample_id"]
        oracle_path = self.split_dir / "oracle_noise" / f"{sample_id}.wav"
        oracle = self._crop(load_pcm16(oracle_path, self.sample_rate), index)
        result["oracle_waveform"] = oracle.squeeze(0)
        return result


class AttentionPoolingHead(nn.Module):
    """Classifier with a learned temporal attention distribution.

    Zero-initialising ``attention_score`` makes the initial softmax uniform, so
    this head starts equivalently to mean pooling while retaining the pretrained
    predictor's class weights.
    """

    requires_sequence = True

    def __init__(self, dimension: int, output_dimension: int) -> None:
        super().__init__()
        self.attention_score = nn.Linear(dimension, 1)
        self.classifier = nn.Linear(dimension, output_dimension)
        nn.init.zeros_(self.attention_score.weight)
        nn.init.zeros_(self.attention_score.bias)

    def forward(self, sequence: torch.Tensor) -> torch.Tensor:
        weights = torch.softmax(self.attention_score(sequence).squeeze(-1), dim=1)
        pooled = torch.sum(sequence * weights.unsqueeze(-1), dim=1)
        return self.classifier(pooled)


def classify(head: nn.Module, sequence: torch.Tensor) -> torch.Tensor:
    if getattr(head, "requires_sequence", False):
        return head(sequence)
    return head(sequence.mean(dim=1))


def load_model(device: torch.device, metadata, trainable_blocks: int, pooling: str = "mean"):
    beats_source = Path("third_party/beats").resolve()
    sys.path.insert(0, str(beats_source))
    from BEATs import BEATs, BEATsConfig

    source = torch.load(BEATS_CHECKPOINT, map_location="cpu", weights_only=True)
    model = BEATs(BEATsConfig(source["cfg"]))
    model.load_state_dict(source["model"])
    mid_to_source = {
        mid: int(index) for index, mid in source["label_dict"].items()
    }
    columns = [mid_to_source[row["mid"]] for row in metadata]
    if pooling == "mean":
        head = nn.Linear(768, len(metadata))
        classifier = head
    elif pooling == "attention":
        head = AttentionPoolingHead(768, len(metadata))
        classifier = head.classifier
    else:
        raise ValueError(f"Unsupported pooling mode: {pooling}")
    with torch.no_grad():
        classifier.weight.copy_(source["model"]["predictor.weight"][columns])
        classifier.bias.copy_(source["model"]["predictor.bias"][columns])
    model.predictor = None
    for parameter in model.parameters():
        parameter.requires_grad = False
    trainable_layers = list(model.encoder.layers[-trainable_blocks:])
    model.to(device)
    head.to(device)
    return model, head, trainable_layers


def load_oracle_teacher(device: torch.device, metadata):
    """Load frozen AudioSet BEATs to create soft labels from oracle noise."""

    beats_source = Path("third_party/beats").resolve()
    sys.path.insert(0, str(beats_source))
    from BEATs import BEATs, BEATsConfig

    source = torch.load(BEATS_CHECKPOINT, map_location="cpu", weights_only=True)
    teacher = BEATs(BEATsConfig(source["cfg"]))
    teacher.load_state_dict(source["model"])
    for parameter in teacher.parameters():
        parameter.requires_grad = False
    teacher.to(device).eval()
    mid_to_source = {mid: int(index) for index, mid in source["label_dict"].items()}
    columns = [mid_to_source[row["mid"]] for row in metadata]
    return teacher, columns


def set_mode(model, head, trainable_layers, encoder_trainable: bool):
    model.eval()
    for layer in trainable_layers:
        layer.train(encoder_trainable)
        for parameter in layer.parameters():
            parameter.requires_grad = encoder_trainable
    head.train()


def make_loader(dataset, batch_size, workers, shuffle, device, sample_weights=None):
    sampler = None
    if sample_weights is not None:
        sampler = WeightedRandomSampler(
            sample_weights,
            num_samples=len(dataset),
            replacement=True,
            generator=torch.Generator().manual_seed(2026),
        )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle and sampler is None,
        sampler=sampler,
        generator=torch.Generator().manual_seed(2026) if shuffle and sampler is None else None,
        num_workers=workers,
        pin_memory=device.type == "cuda",
        persistent_workers=workers > 0,
    )


def class_counts(dataset: Audio21Dataset) -> torch.Tensor:
    counts = torch.zeros(dataset.num_classes, dtype=torch.float32)
    for row in dataset.rows:
        for original_index in ast.literal_eval(row["label_indices"]):
            index = dataset.original_to_model.get(int(original_index))
            if index is not None:
                counts[index] += 1
    return counts


def unique_parameters(parameters):
    """Preserve parameter order while removing shared tensors.

    BEATs shares the relative-attention-bias table across encoder blocks. AdamW
    must receive that tensor once; otherwise it applies its update repeatedly
    in each optimizer step.
    """

    seen: set[int] = set()
    result = []
    for parameter in parameters:
        if id(parameter) not in seen:
            seen.add(id(parameter))
            result.append(parameter)
    return result


@torch.inference_mode()
def evaluate(model, head, loader, device, labels, pos_weight, threshold):
    model.eval()
    head.eval()
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    probabilities, targets, snrs = [], [], []
    total_loss = 0.0
    for batch in loader:
        waveforms = batch["waveform"].to(device, non_blocking=True)
        target = batch["target"].to(device, non_blocking=True)
        with torch.autocast(
            device_type=device.type,
            dtype=torch.bfloat16,
            enabled=device.type == "cuda",
        ):
            sequence, _ = model.extract_features(waveforms)
            logits = classify(head, sequence)
            loss = criterion(logits, target)
        total_loss += loss.item() * target.shape[0]
        probabilities.append(torch.sigmoid(logits.float()).cpu())
        targets.append(target.cpu())
        snrs.append(batch["snr"])
    result = multilabel_metrics(
        torch.cat(probabilities), torch.cat(targets), torch.cat(snrs),
        labels, threshold,
    )
    result["loss"] = total_loss / len(loader.dataset)
    return result


def train_epoch(
    model, head, loader, optimizer, device, trainable_layers,
    encoder_trainable, pos_weight, oracle_teacher=None, teacher_columns=None,
    oracle_distillation_weight: float = 0.0,
):
    set_mode(model, head, trainable_layers, encoder_trainable)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    trainable = [
        parameter
        for parameter in list(model.parameters()) + list(head.parameters())
        if parameter.requires_grad
    ]
    total_loss = 0.0
    seen = 0
    for batch_index, batch in enumerate(loader, start=1):
        waveforms = batch["waveform"].to(device, non_blocking=True)
        targets = batch["target"].to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(
            device_type=device.type,
            dtype=torch.bfloat16,
            enabled=device.type == "cuda",
        ):
            sequence, _ = model.extract_features(waveforms)
            logits = classify(head, sequence)
            classification_loss = criterion(logits, targets)
            loss = classification_loss
            if oracle_teacher is not None:
                oracle_waveforms = batch["oracle_waveform"].to(
                    device, non_blocking=True
                )
                with torch.inference_mode():
                    teacher_probabilities, _ = oracle_teacher.extract_features(
                        oracle_waveforms
                    )
                    teacher_probabilities = teacher_probabilities[:, teacher_columns]
                distillation_loss = nn.functional.binary_cross_entropy_with_logits(
                    logits, teacher_probabilities.float()
                )
                loss = loss + oracle_distillation_weight * distillation_loss
        loss.backward()
        nn.utils.clip_grad_norm_(trainable, 5.0)
        optimizer.step()
        total_loss += loss.item() * targets.shape[0]
        seen += targets.shape[0]
        if batch_index == 1 or batch_index % 100 == 0 or batch_index == len(loader):
            memory = (
                torch.cuda.max_memory_allocated() / 1024**3
                if device.type == "cuda" else 0.0
            )
            print(
                f"train batch={batch_index}/{len(loader)} "
                f"loss={total_loss/seen:.4f} max_vram_gb={memory:.2f}",
                flush=True,
            )
    return {"loss": total_loss / seen}


def encoder_delta(model, trainable_blocks):
    start = len(model.encoder.layers) - trainable_blocks
    prefixes = tuple(
        f"encoder.layers.{index}." for index in range(start, len(model.encoder.layers))
    )
    return {
        key: value.detach().cpu().clone()
        for key, value in model.state_dict().items()
        if key.startswith(prefixes)
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("21_labels_dataset"))
    parser.add_argument("--seconds", type=float, default=6.0)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--head-only-epochs", type=int, default=1)
    parser.add_argument("--trainable-blocks", type=int, default=4)
    parser.add_argument(
        "--pooling", choices=("mean", "attention"), default="mean",
        help="Temporal pooling before the classifier (default: mean).",
    )
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--validation-batch-size", type=int, default=64)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--head-lr", type=float, default=5e-5)
    parser.add_argument("--encoder-lr", type=float, default=2e-6)
    parser.add_argument("--max-pos-weight", type=float, default=20.0)
    parser.add_argument(
        "--oracle-distillation-weight", type=float, default=0.0,
        help="Train-only soft-label loss from frozen BEATs run on oracle noise.",
    )
    parser.add_argument(
        "--high-snr-sampling-weight", type=float, default=1.0,
        help="Relative train sampling weight for examples with SNR >= 15 dB.",
    )
    parser.add_argument(
        "--targeted-pos-weight-cap", type=float,
        help="Cap positive BCE weight only for four high-false-positive labels.",
    )
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--patience", type=int, default=2)
    parser.add_argument("--limit", type=int)
    parser.add_argument(
        "--output", type=Path, default=Path("checkpoints/beats_21_last4.pt")
    )
    parser.add_argument(
        "--results", type=Path,
        default=Path("benchmark_results/beats_21_last4.json"),
    )
    args = parser.parse_args()

    random.seed(2026)
    np.random.seed(2026)
    torch.manual_seed(2026)
    torch.backends.cuda.matmul.allow_tf32 = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    metadata = load_label_metadata(args.data_root)
    labels = [row["display_name"] for row in metadata]
    train_dataset_class = (
        OraclePairAudio21Dataset
        if args.oracle_distillation_weight > 0 else Audio21Dataset
    )
    train_dataset = train_dataset_class(
        args.data_root, "train", args.seconds, "random", args.limit
    )
    validation_dataset = Audio21Dataset(
        args.data_root, "validation", args.seconds, "center", args.limit
    )
    high_snr_weights = None
    if args.high_snr_sampling_weight != 1.0:
        high_snr_weights = torch.tensor(
            [
                args.high_snr_sampling_weight
                if float(row["target_snr_db"]) >= 15.0 else 1.0
                for row in train_dataset.rows
            ],
            dtype=torch.double,
        )
    train_loader = make_loader(
        train_dataset, args.batch_size, args.workers, True, device, high_snr_weights
    )
    validation_loader = make_loader(
        validation_dataset, args.validation_batch_size, args.workers, False, device
    )
    model, head, trainable_layers = load_model(
        device, metadata, args.trainable_blocks, args.pooling
    )
    oracle_teacher = None
    teacher_columns = None
    if args.oracle_distillation_weight > 0:
        oracle_teacher, teacher_columns = load_oracle_teacher(device, metadata)
    counts = class_counts(train_dataset)
    pos_weight = ((len(train_dataset) - counts) / counts.clamp_min(1)).clamp(
        1.0, args.max_pos_weight
    ).to(device)
    if args.targeted_pos_weight_cap is not None:
        targeted_labels = {
            "Speech", "Inside, small room", "Car", "Musical instrument",
        }
        for index, label in enumerate(labels):
            if label in targeted_labels:
                pos_weight[index] = pos_weight[index].clamp(
                    max=args.targeted_pos_weight_cap
                )
    encoder_parameters = unique_parameters(
        parameter for layer in trainable_layers for parameter in layer.parameters()
    )
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
        f"validation={len(validation_dataset)} labels={len(labels)} "
        f"trainable_encoder_params={sum(p.numel() for p in encoder_parameters)}",
        flush=True,
    )

    initial = evaluate(
        model, head, validation_loader, device, labels, pos_weight, args.threshold
    )
    print(
        f"epoch=0 val_loss={initial['loss']:.4f} "
        f"micro_f1={initial['micro_f1']:.4f} "
        f"macro_f1={initial['macro_f1']:.4f} "
        f"mAP={initial['macro_average_precision']:.4f}",
        flush=True,
    )
    best_epoch = 0
    best_metrics = initial
    best_head = {k: v.detach().cpu().clone() for k, v in head.state_dict().items()}
    best_encoder = encoder_delta(model, args.trainable_blocks)
    history = [{"epoch": 0, "validation": initial}]
    stale = 0
    started = time.perf_counter()
    for epoch in range(1, args.epochs + 1):
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats()
        encoder_trainable = epoch > args.head_only_epochs
        train_result = train_epoch(
            model, head, train_loader, optimizer, device, trainable_layers,
            encoder_trainable, pos_weight, oracle_teacher, teacher_columns,
            args.oracle_distillation_weight,
        )
        validation = evaluate(
            model, head, validation_loader, device, labels, pos_weight, args.threshold
        )
        scheduler.step(validation["macro_average_precision"])
        history.append(
            {
                "epoch": epoch,
                "encoder_trainable": encoder_trainable,
                "train": train_result,
                "validation": validation,
                "encoder_lr": optimizer.param_groups[0]["lr"],
                "head_lr": optimizer.param_groups[1]["lr"],
            }
        )
        print(
            f"epoch={epoch} encoder_trainable={encoder_trainable} "
            f"train_loss={train_result['loss']:.4f} "
            f"val_loss={validation['loss']:.4f} "
            f"micro_f1={validation['micro_f1']:.4f} "
            f"macro_f1={validation['macro_f1']:.4f} "
            f"mAP={validation['macro_average_precision']:.4f}",
            flush=True,
        )
        if validation["macro_average_precision"] > best_metrics["macro_average_precision"] + 1e-4:
            best_epoch = epoch
            best_metrics = validation
            best_head = {
                k: v.detach().cpu().clone() for k, v in head.state_dict().items()
            }
            best_encoder = encoder_delta(model, args.trainable_blocks)
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
            "encoder_delta": best_encoder,
            "head": best_head,
            "labels": labels,
            "metadata": metadata,
            "base_encoder_checkpoint": str(BEATS_CHECKPOINT),
            "trainable_blocks": args.trainable_blocks,
            "best_epoch": best_epoch,
            "validation_metrics": best_metrics,
            "args": serializable_args,
        },
        args.output,
    )
    result = {
        "stage": "beats_21_multilabel_partial_finetune",
        "train_samples": len(train_dataset),
        "validation_samples": len(validation_dataset),
        "best_epoch": best_epoch,
        "best_validation": best_metrics,
        "history": history,
        "elapsed_seconds": time.perf_counter() - started,
        "checkpoint": str(args.output),
        "args": serializable_args,
        "test_used": False,
    }
    args.results.parent.mkdir(parents=True, exist_ok=True)
    args.results.write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(
        f"best_epoch={best_epoch} micro_f1={best_metrics['micro_f1']:.4f} "
        f"macro_f1={best_metrics['macro_f1']:.4f} "
        f"mAP={best_metrics['macro_average_precision']:.4f}",
        flush=True,
    )
    print(f"checkpoint={args.output} results={args.results}", flush=True)
    del model, head
    gc.collect()


if __name__ == "__main__":
    main()

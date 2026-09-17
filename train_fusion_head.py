"""Train the fusion head on cached (z_mix, z_noise) from frozen BEATs + separator."""

from __future__ import annotations

import argparse
import gc
import json
import time
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from models.fusion import FusionClassifier, encode_branches
from models.separator import build_separator, separator_payload
from noise_pipeline.mix_data import MixNoiseDataset, load_mix_manifest
from utils.reporting36 import save_evaluation_artifacts, save_training_artifacts
from utils.training36 import (
    SEED,
    initialize_head_from_audioset,
    label_to_mid_map,
    load_beats,
    metrics,
    seed_everything,
    select_rows,
)

DEFAULT_CACHE_DIR = Path("artifacts/embedding_cache_36_fusion")


def validation_score(result: dict, metric_name: str) -> float:
    if metric_name == "macro_f1":
        return result["macro_f1"]
    if metric_name == "worst_group_macro_f1":
        return min(values["macro_f1"] for values in result["per_snr"].values())
    raise ValueError("selection_metric must be 'macro_f1' or 'worst_group_macro_f1'")


def file_signature(path: Path) -> str:
    stat = path.stat()
    return f"{path.name}:{stat.st_size}:{int(stat.st_mtime)}"


@torch.inference_mode()
def extract_split(encoder, separator, dataset, device, batch_size: int, workers: int):
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=workers,
        pin_memory=device.type == "cuda",
        persistent_workers=workers > 0,
    )
    mix_embeddings, noise_embeddings, conditioned_embeddings = [], [], []
    noise_ratios, speech_ratios, targets, snrs = [], [], [], []
    started = time.perf_counter()
    for batch_index, batch in enumerate(loader, start=1):
        mixture = batch["mixture"].to(device, non_blocking=True)
        z_mix, z_noise, z_conditioned, _, noise_ratio_db, speech_ratio_db = encode_branches(
            encoder, separator, mixture, device
        )
        mix_embeddings.append(z_mix.float().cpu())
        noise_embeddings.append(z_noise.float().cpu())
        conditioned_embeddings.append(z_conditioned.float().cpu())
        noise_ratios.append(noise_ratio_db.float().cpu())
        speech_ratios.append(speech_ratio_db.float().cpu())
        targets.append(batch["target"].long())
        snrs.append(batch["snr"].long())
        if batch_index == 1 or batch_index % 100 == 0 or batch_index == len(loader):
            print(f"extract split={dataset.split} batch={batch_index}/{len(loader)}", flush=True)
    return {
        "x_mix": torch.cat(mix_embeddings),
        "x_noise": torch.cat(noise_embeddings),
        "x_conditioned": torch.cat(conditioned_embeddings),
        "noise_ratio_db": torch.cat(noise_ratios),
        "speech_ratio_db": torch.cat(speech_ratios),
        "y": torch.cat(targets),
        "snr": torch.cat(snrs),
        "seconds": time.perf_counter() - started,
    }


def load_or_extract(
    split, root, rows, per_class, encoder, separator, separator_signature,
    device, cache_dir, batch_size, workers, force,
):
    suffix = "full" if per_class is None else f"pc{per_class}"
    cache_path = cache_dir / f"beats_fusion_{split}_{suffix}_seed{SEED}.pt"
    if cache_path.exists() and not force:
        data = torch.load(cache_path, map_location="cpu", weights_only=True)
        if data.get("separator_signature") == separator_signature:
            print(f"cache split={split} path={cache_path}", flush=True)
            return data
        print(f"cache split={split} was built with another separator; re-extracting", flush=True)
    dataset = MixNoiseDataset(
        root, split, rows=select_rows(rows, split, per_class), load_oracle_noise=False
    )
    data = extract_split(encoder, separator, dataset, device, batch_size, workers)
    data["separator_signature"] = separator_signature
    cache_dir.mkdir(parents=True, exist_ok=True)
    torch.save(data, cache_path)
    print(
        f"cached split={split} samples={len(dataset)} seconds={data['seconds']:.1f} path={cache_path}",
        flush=True,
    )
    return data


@torch.inference_mode()
def predict(classifier, data, device, batch_size: int = 2048):
    classifier.eval()
    outputs = []
    for start in range(0, len(data["y"]), batch_size):
        stop = start + batch_size
        outputs.append(
            classifier(
                data["x_mix"][start:stop].to(device),
                data["x_noise"][start:stop].to(device),
                data["x_conditioned"][start:stop].to(device),
                data["noise_ratio_db"][start:stop].to(device),
                data["speech_ratio_db"][start:stop].to(device),
            ).cpu()
        )
    return torch.cat(outputs)


def train_classifier(classifier, train_data, validation_data, labels, device, args):
    loader = DataLoader(
        TensorDataset(
            train_data["x_mix"], train_data["x_noise"], train_data["x_conditioned"],
            train_data["noise_ratio_db"], train_data["speech_ratio_db"], train_data["y"]
        ),
        batch_size=args.head_batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(args.seed),
        pin_memory=device.type == "cuda",
    )
    criterion = nn.CrossEntropyLoss()
    # The 36-class head learns at the usual rate; the 1.2M-parameter fusion
    # projection learns slowly with strong decay toward its zero (mixture-only) start.
    fusion_parameters = list(classifier.fusion.parameters())
    fusion_ids = {id(parameter) for parameter in fusion_parameters}
    optimizer = torch.optim.AdamW(
        [
            {
                "params": [p for p in classifier.parameters() if id(p) not in fusion_ids],
                "lr": args.lr,
                "weight_decay": 1e-4,
            },
            {
                "params": fusion_parameters,
                "lr": args.fusion_lr,
                "weight_decay": args.fusion_weight_decay,
            },
        ]
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=2, min_lr=1e-6
    )

    def validate():
        logits = predict(classifier, validation_data, device)
        result = metrics(logits, validation_data["y"], validation_data["snr"], labels)
        result["worst_group_macro_f1"] = min(
            values["macro_f1"] for values in result["per_snr"].values()
        )
        result["loss"] = float(criterion(logits, validation_data["y"]))
        return result

    best_metrics = validate()
    best_state = {key: value.detach().cpu().clone() for key, value in classifier.state_dict().items()}
    best_epoch = 0
    stale = 0
    history = [{"epoch": 0, "validation": best_metrics}]
    print(
        f"epoch=0 val_accuracy={best_metrics['accuracy']:.4f} val_macro_f1={best_metrics['macro_f1']:.4f} val_mAP={best_metrics['mAP']:.4f}",
        flush=True,
    )
    for epoch in range(1, args.epochs + 1):
        classifier.train()
        total_loss = 0.0
        correct = 0
        seen = 0
        for z_mix, z_noise, z_conditioned, noise_ratio_db, speech_ratio_db, targets in loader:
            z_mix = z_mix.to(device, non_blocking=True)
            z_noise = z_noise.to(device, non_blocking=True)
            z_conditioned = z_conditioned.to(device, non_blocking=True)
            noise_ratio_db = noise_ratio_db.to(device, non_blocking=True)
            speech_ratio_db = speech_ratio_db.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            logits = classifier(z_mix, z_noise, z_conditioned, noise_ratio_db, speech_ratio_db)
            loss = criterion(logits, targets)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * targets.shape[0]
            correct += (logits.argmax(dim=1) == targets).sum().item()
            seen += targets.shape[0]

        validation_metrics = validate()
        score = validation_score(validation_metrics, args.selection_metric)
        scheduler.step(score)
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
            f"val_loss={validation_metrics['loss']:.4f} "
            f"val_accuracy={validation_metrics['accuracy']:.4f} "
            f"val_macro_f1={validation_metrics['macro_f1']:.4f} val_mAP={validation_metrics['mAP']:.4f} "
            f"lr={epoch_result['learning_rate']:.2e}",
            flush=True,
        )
        if score > validation_score(best_metrics, args.selection_metric) + 1e-4:
            best_metrics = validation_metrics
            best_epoch = epoch
            best_state = {key: value.detach().cpu().clone() for key, value in classifier.state_dict().items()}
            stale = 0
        else:
            stale += 1
            if stale >= args.patience:
                print(f"early_stop={epoch}", flush=True)
                break
    classifier.load_state_dict(best_state)
    return best_epoch, best_metrics, history


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("../../36_labels"))
    parser.add_argument("--separator-checkpoint", type=Path, default=Path("checkpoint/noise_separator.pt"))
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument("--output", type=Path, default=Path("checkpoint/beats_fusion_head_36.pt"))
    parser.add_argument("--results", type=Path, default=Path("checkpoint/beats_fusion_head_36_summary.json"))
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--extract-batch-size", type=int, default=32)
    parser.add_argument("--head-batch-size", type=int, default=512)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--fusion-lr", type=float, default=1e-4)
    parser.add_argument("--fusion-weight-decay", type=float, default=0.05)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--noise-dropout", type=float, default=0.2)
    parser.add_argument("--conditioned-dropout", type=float, default=0.2)
    parser.add_argument("--selection-metric", choices=("macro_f1", "worst_group_macro_f1"),
                        default="worst_group_macro_f1")
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--seed", type=int, default=SEED,
                        help="training randomness only; data selection and embedding cache stay fixed")
    parser.add_argument("--target-rms", type=float, help="override the separator checkpoint value")
    parser.add_argument("--max-gain-db", type=float, help="override the separator checkpoint value")
    parser.add_argument("--speech-threshold-db", type=float, help="WDRC compression knee in dBFS")
    parser.add_argument("--speech-ratio", type=float, help="WDRC ratio; must be at least 1")
    parser.add_argument("--speech-target-rms", type=float, help="post-WDRC make-up target RMS")
    parser.add_argument("--speech-max-gain-db", type=float, help="cap post-WDRC make-up gain")
    parser.add_argument("--speech-attack-ms", type=float, help="WDRC attack time")
    parser.add_argument("--speech-release-ms", type=float, help="WDRC release time")
    parser.add_argument("--train-per-class", type=int)
    parser.add_argument("--validation-per-class", type=int)
    parser.add_argument("--force-extract", action="store_true")
    parser.add_argument("--smoke-test", action="store_true")
    args = parser.parse_args()

    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.smoke_test:
        args.train_per_class = 6
        args.validation_per_class = 6
        args.epochs = 2
        args.workers = 0
        args.separator_checkpoint = Path("checkpoint/noise_separator_smoke.pt")
        args.cache_dir = Path("artifacts/embedding_cache_36_fusion_smoke")
        args.output = Path("checkpoint/beats_fusion_head_36_smoke.pt")
        args.results = Path("checkpoint/beats_fusion_head_36_smoke.json")

    labels = (args.data_root / "labels.txt").read_text(encoding="utf-8").splitlines()
    rows = load_mix_manifest(args.data_root)
    print(f"device={device} labels={len(labels)}", flush=True)

    separator_checkpoint = torch.load(args.separator_checkpoint, map_location="cpu", weights_only=True)
    separator = build_separator(separator_checkpoint["separator"]).to(device).eval()
    separator.set_amplification(args.target_rms, args.max_gain_db)
    separator.set_speech_conditioning(
        args.speech_threshold_db, args.speech_ratio, args.speech_target_rms,
        args.speech_max_gain_db, args.speech_attack_ms, args.speech_release_ms,
    )
    # Amplification changes z_noise, so it is part of the cache key.
    signature = (
        f"{file_signature(args.separator_checkpoint)}"
        f"|separator={separator.config}"
    )
    print(
        f"noise_target_rms={separator.config['target_rms']} "
        f"speech_threshold_db={separator.config['speech_threshold_db']} "
        f"speech_ratio={separator.config['speech_ratio']}",
        flush=True,
    )
    encoder, beats_checkpoint = load_beats(device)
    extraction = (encoder, separator, signature, device, args.cache_dir,
                  args.extract_batch_size, args.workers, args.force_extract)
    train_data = load_or_extract("train", args.data_root, rows, args.train_per_class, *extraction)
    validation_data = load_or_extract(
        "validation", args.data_root, rows, args.validation_per_class, *extraction
    )
    del encoder
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()

    classifier = FusionClassifier(
        len(labels), dropout=args.dropout, noise_dropout=args.noise_dropout,
        conditioned_dropout=args.conditioned_dropout,
    )
    initialize_head_from_audioset(classifier.head, beats_checkpoint, labels, label_to_mid_map(rows))
    classifier.to(device)
    best_epoch, best_metrics, history = train_classifier(
        classifier, train_data, validation_data, labels, device, args
    )

    serializable_args = {
        key: str(value) if isinstance(value, Path) else value
        for key, value in vars(args).items()
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "classifier": {key: value.cpu() for key, value in classifier.state_dict().items()},
            "separator": separator_payload(separator),
            "labels": labels,
            "encoder": "BEATs_iter3_plus_AS2M_finetuned_cpt2",
            "args": serializable_args,
            "best_epoch": best_epoch,
            "validation_metrics": best_metrics,
        },
        args.output,
    )
    result = {
        "stage": "frozen_beats_mixture_noise_fusion_head",
        "input_kind": "mixture+separated_noise",
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
    args.results.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    save_training_artifacts(args.output.parent, labels, result)
    validation_logits = predict(classifier, validation_data, device)
    save_evaluation_artifacts(
        args.output.parent,
        best_metrics,
        labels,
        validation_data["y"].numpy(),
        validation_logits.argmax(dim=1).numpy(),
        split="validation",
    )
    print(
        f"best_epoch={best_epoch} val_accuracy={best_metrics['accuracy']:.4f} "
        f"val_macro_f1={best_metrics['macro_f1']:.4f} val_mAP={best_metrics['mAP']:.4f}",
        flush=True,
    )
    print(f"checkpoint={args.output} results={args.results}", flush=True)


if __name__ == "__main__":
    main()

"""Pre-train the noise separator on (mixture, oracle_noise) pairs with SI-SDR."""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader

from models.separator import NoiseSeparator, separation_loss, separator_payload, si_sdr
from noise_pipeline.mix_data import MixNoiseDataset, load_mix_manifest
from utils.training36 import SEED, SNRS, require_finite, seed_everything, select_rows


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


@torch.inference_mode()
def evaluate(separator: NoiseSeparator, loader, device: torch.device) -> dict:
    separator.eval()
    estimates, baselines, snrs = [], [], []
    for batch in loader:
        mixture = batch["mixture"].to(device, non_blocking=True)
        oracle = batch["oracle_noise"].to(device, non_blocking=True)
        estimates.append(si_sdr(separator(mixture), oracle).cpu())
        baselines.append(si_sdr(mixture, oracle).cpu())
        snrs.append(batch["snr"])
    estimates = torch.cat(estimates)
    baselines = torch.cat(baselines)
    snrs = torch.cat(snrs)
    require_finite(estimates, "validation SI-SDR")
    result = {
        "si_sdr": float(estimates.mean()),
        "mixture_si_sdr": float(baselines.mean()),
        "si_sdr_improvement": float((estimates - baselines).mean()),
        "per_snr": {},
    }
    for snr in SNRS:
        mask = snrs == snr
        if not mask.any():
            continue
        result["per_snr"][str(snr)] = {
            "samples": int(mask.sum()),
            "si_sdr": float(estimates[mask].mean()),
            "mixture_si_sdr": float(baselines[mask].mean()),
            "si_sdr_improvement": float((estimates[mask] - baselines[mask]).mean()),
        }
    return result


def train_epoch(separator, loader, optimizer, device) -> float:
    separator.train()
    total_loss = 0.0
    seen = 0
    for batch_index, batch in enumerate(loader, start=1):
        mixture = batch["mixture"].to(device, non_blocking=True)
        oracle = batch["oracle_noise"].to(device, non_blocking=True)
        loss = separation_loss(separator(mixture), oracle)
        require_finite(loss, "separation loss")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(separator.parameters(), 5.0)
        optimizer.step()
        total_loss += loss.item() * mixture.shape[0]
        seen += mixture.shape[0]
        if batch_index == 1 or batch_index % 200 == 0 or batch_index == len(loader):
            print(
                f"train batch={batch_index}/{len(loader)} sep_loss={total_loss/seen:.4f}",
                flush=True,
            )
    return total_loss / seen


def save_reports(output_dir: Path, history: list[dict], best: dict) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "separator_history.csv").open("w", newline="", encoding="utf-8") as handle:
        fields = ["epoch", "train_sep_loss", "validation_si_sdr", "validation_si_sdr_improvement"]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in history:
            writer.writerow(
                {
                    "epoch": row["epoch"],
                    "train_sep_loss": row.get("train_sep_loss"),
                    "validation_si_sdr": row["validation"]["si_sdr"],
                    "validation_si_sdr_improvement": row["validation"]["si_sdr_improvement"],
                }
            )
    with (output_dir / "separator_snr_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        fields = ["snr_db", "samples", "si_sdr", "mixture_si_sdr", "si_sdr_improvement"]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for snr, values in best["per_snr"].items():
            writer.writerow({"snr_db": snr, **{key: values[key] for key in fields[1:]}})


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("../../36_labels"))
    parser.add_argument("--output", type=Path, default=Path("checkpoint/noise_separator.pt"))
    parser.add_argument("--results", type=Path, default=Path("checkpoint/noise_separator_summary.json"))
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--validation-batch-size", type=int, default=64)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--n-fft", type=int, default=512)
    parser.add_argument("--hop-length", type=int, default=160)
    parser.add_argument("--win-length", type=int, default=400)
    parser.add_argument("--channels", type=int, default=256)
    parser.add_argument("--blocks", type=int, default=8)
    parser.add_argument("--kernel-size", type=int, default=3)
    parser.add_argument("--target-rms", type=float, default=0.1)
    parser.add_argument("--max-gain-db", type=float, default=40.0)
    parser.add_argument("--train-per-class", type=int)
    parser.add_argument("--validation-per-class", type=int)
    parser.add_argument("--smoke-test", action="store_true")
    args = parser.parse_args()

    seed_everything()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.smoke_test:
        args.train_per_class = 6
        args.validation_per_class = 6
        args.epochs = 1
        args.batch_size = 4
        args.validation_batch_size = 8
        args.workers = 0
        args.output = Path("checkpoint/noise_separator_smoke.pt")
        args.results = Path("checkpoint/noise_separator_smoke.json")

    rows = load_mix_manifest(args.data_root)
    train_dataset = MixNoiseDataset(
        args.data_root, "train", rows=select_rows(rows, "train", args.train_per_class)
    )
    validation_dataset = MixNoiseDataset(
        args.data_root, "validation",
        rows=select_rows(rows, "validation", args.validation_per_class),
    )
    train_loader = make_loader(train_dataset, args.batch_size, args.workers, True, device)
    validation_loader = make_loader(
        validation_dataset, args.validation_batch_size, args.workers, False, device
    )
    separator = NoiseSeparator(
        n_fft=args.n_fft,
        hop_length=args.hop_length,
        win_length=args.win_length,
        channels=args.channels,
        blocks=args.blocks,
        kernel_size=args.kernel_size,
        target_rms=args.target_rms,
        max_gain_db=args.max_gain_db,
    ).to(device)
    optimizer = torch.optim.AdamW(separator.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=2, min_lr=1e-6
    )
    print(
        f"device={device} train={len(train_dataset)} validation={len(validation_dataset)} "
        f"separator_params={sum(p.numel() for p in separator.parameters())}",
        flush=True,
    )

    best = evaluate(separator, validation_loader, device)
    print(
        f"epoch=0 val_si_sdr={best['si_sdr']:.2f} mixture_si_sdr={best['mixture_si_sdr']:.2f}",
        flush=True,
    )
    best_epoch = 0
    best_payload = separator_payload(separator)
    history = [{"epoch": 0, "validation": best}]
    stale = 0
    started = time.perf_counter()
    for epoch in range(1, args.epochs + 1):
        train_loss = train_epoch(separator, train_loader, optimizer, device)
        validation = evaluate(separator, validation_loader, device)
        scheduler.step(validation["si_sdr"])
        history.append(
            {
                "epoch": epoch,
                "train_sep_loss": train_loss,
                "learning_rate": optimizer.param_groups[0]["lr"],
                "validation": validation,
            }
        )
        print(
            f"epoch={epoch} train_sep_loss={train_loss:.4f} "
            f"val_si_sdr={validation['si_sdr']:.2f} "
            f"val_si_sdri={validation['si_sdr_improvement']:.2f}",
            flush=True,
        )
        if validation["si_sdr"] > best["si_sdr"] + 1e-3:
            best = validation
            best_epoch = epoch
            best_payload = separator_payload(separator)
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
            "separator": best_payload,
            "best_epoch": best_epoch,
            "validation_metrics": best,
            "args": serializable_args,
        },
        args.output,
    )
    result = {
        "stage": "noise_separator_pretraining",
        "train_samples": len(train_dataset),
        "validation_samples": len(validation_dataset),
        "best_epoch": best_epoch,
        "best_validation": best,
        "history": history,
        "elapsed_seconds": time.perf_counter() - started,
        "checkpoint": str(args.output),
        "args": serializable_args,
    }
    args.results.parent.mkdir(parents=True, exist_ok=True)
    args.results.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    save_reports(args.output.parent, history, best)
    print(
        f"best_epoch={best_epoch} val_si_sdr={best['si_sdr']:.2f} "
        f"val_si_sdri={best['si_sdr_improvement']:.2f}",
        flush=True,
    )
    print(f"checkpoint={args.output} results={args.results}", flush=True)


if __name__ == "__main__":
    main()

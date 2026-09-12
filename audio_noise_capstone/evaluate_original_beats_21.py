"""Evaluate the official BEATs AudioSet model on the 21-label dataset.

This is a strict zero-shot evaluation: the official 527-label predictor is
restricted to the 21 matching AudioSet class IDs. No optimizer, backward pass,
threshold calibration, or update on the 21-label dataset is performed.
"""

from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from benchmark_pretrained import BEATS_CHECKPOINT
from noise_pipeline.audio21 import (
    Audio21Dataset,
    load_label_metadata,
    multilabel_metrics,
)


class OriginalBEATs21:
    """Official BEATs iter3+ AudioSet model with a restricted output view."""

    def __init__(
        self,
        device: torch.device,
        metadata: list[dict[str, str]],
    ) -> None:
        beats_source = Path("third_party/beats").resolve()
        sys.path.insert(0, str(beats_source))
        from BEATs import BEATs, BEATsConfig

        checkpoint = torch.load(
            BEATS_CHECKPOINT, map_location="cpu", weights_only=True
        )
        self.config = dict(checkpoint["cfg"])
        self.model = BEATs(BEATsConfig(self.config))
        self.model.load_state_dict(checkpoint["model"], strict=True)
        self.model.to(device).eval()
        for parameter in self.model.parameters():
            parameter.requires_grad = False

        mid_to_source = {
            mid: int(index) for index, mid in checkpoint["label_dict"].items()
        }
        missing = [row["mid"] for row in metadata if row["mid"] not in mid_to_source]
        if missing:
            raise ValueError(f"AudioSet IDs absent from BEATs checkpoint: {missing}")
        self.columns = torch.tensor(
            [mid_to_source[row["mid"]] for row in metadata],
            dtype=torch.long,
            device=device,
        )
        self.device = device
        self.parameter_count = sum(p.numel() for p in self.model.parameters())
        self.trainable_parameter_count = sum(
            p.numel() for p in self.model.parameters() if p.requires_grad
        )

    @torch.inference_mode()
    def predict(self, waveforms: torch.Tensor) -> torch.Tensor:
        probabilities_527, _ = self.model.extract_features(
            waveforms.to(self.device, non_blocking=True)
        )
        return probabilities_527.index_select(1, self.columns).float().cpu()


def evaluate_split(
    predictor: OriginalBEATs21,
    dataset: Audio21Dataset,
    labels: list[str],
    batch_size: int,
    workers: int,
    device: torch.device,
    threshold: float,
) -> dict:
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        pin_memory=device.type == "cuda",
        persistent_workers=workers > 0,
    )
    probabilities, targets, snrs = [], [], []
    started = time.perf_counter()
    for batch_index, batch in enumerate(loader, start=1):
        probabilities.append(predictor.predict(batch["waveform"]))
        targets.append(batch["target"])
        snrs.append(batch["snr"])
        if batch_index == 1 or batch_index % 25 == 0 or batch_index == len(loader):
            print(
                f"split={dataset.split} batch={batch_index}/{len(loader)}",
                flush=True,
            )
    metrics = multilabel_metrics(
        torch.cat(probabilities),
        torch.cat(targets),
        torch.cat(snrs),
        labels,
        threshold,
    )
    metrics["samples"] = len(dataset)
    metrics["elapsed_seconds"] = time.perf_counter() - started
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("21_labels_dataset"))
    parser.add_argument(
        "--splits",
        nargs="+",
        choices=("validation", "test"),
        default=("validation", "test"),
    )
    parser.add_argument("--seconds", type=float, default=6.0)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("benchmark_results/beats_original_21_zero_shot.json"),
    )
    args = parser.parse_args()

    torch.manual_seed(2026)
    torch.backends.cuda.matmul.allow_tf32 = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    metadata = load_label_metadata(args.data_root)
    labels = [row["display_name"] for row in metadata]
    predictor = OriginalBEATs21(device, metadata)
    result = {
        "protocol": "official_beats_audioset_restricted_21_zero_shot",
        "checkpoint": str(BEATS_CHECKPOINT),
        "data_root": str(args.data_root),
        "labels": labels,
        "seconds": args.seconds,
        "crop": "center",
        "threshold": args.threshold,
        "threshold_calibrated": False,
        "optimizer_used": False,
        "backward_passes": 0,
        "parameter_updates": 0,
        "parameter_count": predictor.parameter_count,
        "trainable_parameter_count": predictor.trainable_parameter_count,
        "architecture": {
            "encoder_layers": predictor.config["encoder_layers"],
            "encoder_embed_dim": predictor.config["encoder_embed_dim"],
            "encoder_attention_heads": predictor.config["encoder_attention_heads"],
            "encoder_ffn_embed_dim": predictor.config["encoder_ffn_embed_dim"],
            "input_patch_size": predictor.config["input_patch_size"],
            "upstream_predictor_classes": predictor.config["predictor_class"],
            "reported_classes": len(labels),
        },
        "splits": {},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    print(
        f"device={device} parameters={predictor.parameter_count} "
        f"trainable={predictor.trainable_parameter_count} labels={len(labels)}",
        flush=True,
    )
    for split in args.splits:
        dataset = Audio21Dataset(
            args.data_root, split, args.seconds, "center", args.limit
        )
        metrics = evaluate_split(
            predictor,
            dataset,
            labels,
            args.batch_size,
            args.workers,
            device,
            args.threshold,
        )
        result["splits"][split] = metrics
        args.output.write_text(
            json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(
            f"split={split} micro_f1={metrics['micro_f1']:.4f} "
            f"macro_f1={metrics['macro_f1']:.4f} "
            f"mAP={metrics['macro_average_precision']:.4f} "
            f"exact_match={metrics['exact_match']:.4f}",
            flush=True,
        )
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()
    print(f"results={args.output}", flush=True)


if __name__ == "__main__":
    main()

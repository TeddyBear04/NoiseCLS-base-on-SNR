"""Evaluate AudioSet heads from AST, PANNs and BEATs on 21 labels."""

from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torchaudio
from torch.utils.data import DataLoader

from benchmark_pretrained import AST_CHECKPOINT, BEATS_CHECKPOINT, PANNS_CHECKPOINT
from noise_pipeline.audio21 import Audio21Dataset, load_label_metadata, multilabel_metrics


BATCH_SIZES = {"ast": 8, "panns": 16, "beats": 16}


class ASTPredictor:
    name = "ast"

    def __init__(self, device: torch.device, metadata: list[dict[str, str]]):
        from transformers import ASTFeatureExtractor, ASTForAudioClassification

        self.device = device
        self.extractor = ASTFeatureExtractor.from_pretrained(
            AST_CHECKPOINT, local_files_only=True
        )
        self.model = ASTForAudioClassification.from_pretrained(
            AST_CHECKPOINT, local_files_only=True
        ).to(device).eval()
        label_to_source = {
            label: int(index) for index, label in self.model.config.id2label.items()
        }
        self.columns = torch.tensor(
            [label_to_source[row["display_name"]] for row in metadata],
            device=device,
        )
        self.parameter_count = sum(p.numel() for p in self.model.parameters())

    @torch.inference_mode()
    def __call__(self, waveforms: torch.Tensor) -> torch.Tensor:
        inputs = self.extractor(
            [waveform.numpy() for waveform in waveforms],
            sampling_rate=16_000,
            return_tensors="pt",
        )["input_values"].to(self.device)
        logits = self.model(input_values=inputs).logits
        return torch.sigmoid(logits.index_select(1, self.columns)).float().cpu()


class PANNsPredictor:
    name = "panns"

    def __init__(self, device: torch.device, metadata: list[dict[str, str]]):
        from panns_inference.config import labels as source_labels
        from panns_inference.models import Cnn14

        self.device = device
        self.resample = torchaudio.transforms.Resample(16_000, 32_000).to(device)
        self.model = Cnn14(
            sample_rate=32_000,
            window_size=1024,
            hop_size=320,
            mel_bins=64,
            fmin=50,
            fmax=14_000,
            classes_num=527,
        )
        checkpoint = torch.load(PANNS_CHECKPOINT, map_location="cpu", weights_only=True)
        self.model.load_state_dict(checkpoint["model"])
        self.model.to(device).eval()
        label_to_source = {label: index for index, label in enumerate(source_labels)}
        self.columns = torch.tensor(
            [label_to_source[row["display_name"]] for row in metadata],
            device=device,
        )
        self.parameter_count = sum(p.numel() for p in self.model.parameters())

    @torch.inference_mode()
    def __call__(self, waveforms: torch.Tensor) -> torch.Tensor:
        probabilities = self.model(
            self.resample(waveforms.to(self.device)), None
        )["clipwise_output"]
        return probabilities.index_select(1, self.columns).float().cpu()


class BEATsPredictor:
    name = "beats"

    def __init__(self, device: torch.device, metadata: list[dict[str, str]]):
        beats_source = Path("third_party/beats").resolve()
        sys.path.insert(0, str(beats_source))
        from BEATs import BEATs, BEATsConfig

        checkpoint = torch.load(BEATS_CHECKPOINT, map_location="cpu", weights_only=True)
        self.model = BEATs(BEATsConfig(checkpoint["cfg"]))
        self.model.load_state_dict(checkpoint["model"])
        self.model.to(device).eval()
        self.device = device
        mid_to_source = {
            mid: int(index) for index, mid in checkpoint["label_dict"].items()
        }
        self.columns = torch.tensor(
            [mid_to_source[row["mid"]] for row in metadata], device=device
        )
        self.parameter_count = sum(p.numel() for p in self.model.parameters())

    @torch.inference_mode()
    def __call__(self, waveforms: torch.Tensor) -> torch.Tensor:
        probabilities, _ = self.model.extract_features(waveforms.to(self.device))
        return probabilities.index_select(1, self.columns).float().cpu()


def create_predictor(name, device, metadata):
    return {"ast": ASTPredictor, "panns": PANNsPredictor, "beats": BEATsPredictor}[
        name
    ](device, metadata)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("21_labels_dataset"))
    parser.add_argument(
        "--encoders", nargs="+", choices=("ast", "panns", "beats"),
        default=("ast", "panns", "beats"),
    )
    parser.add_argument("--seconds", type=float, default=6.0)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument(
        "--output", type=Path,
        default=Path("benchmark_results/pretrained_21_zero_shot.json"),
    )
    args = parser.parse_args()

    torch.manual_seed(2026)
    torch.backends.cuda.matmul.allow_tf32 = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    metadata = load_label_metadata(args.data_root)
    labels = [row["display_name"] for row in metadata]
    dataset = Audio21Dataset(
        args.data_root, "validation", args.seconds, "center", args.limit
    )
    print(
        f"device={device} validation={len(dataset)} labels={len(labels)} ",
        flush=True,
    )
    result = {
        "protocol": "restricted_audioset_head_multilabel",
        "validation_samples": len(dataset),
        "seconds": args.seconds,
        "threshold": args.threshold,
        "test_used": False,
        "encoders": {},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)

    for name in args.encoders:
        predictor = create_predictor(name, device, metadata)
        loader = DataLoader(
            dataset,
            batch_size=BATCH_SIZES[name],
            num_workers=args.workers,
            pin_memory=device.type == "cuda",
            persistent_workers=args.workers > 0,
        )
        probabilities, targets, snrs = [], [], []
        started = time.perf_counter()
        for batch_index, batch in enumerate(loader, start=1):
            probabilities.append(predictor(batch["waveform"]))
            targets.append(batch["target"])
            snrs.append(batch["snr"])
            if batch_index == 1 or batch_index % 25 == 0 or batch_index == len(loader):
                print(
                    f"encoder={name} batch={batch_index}/{len(loader)}", flush=True
                )
        metrics = multilabel_metrics(
            torch.cat(probabilities), torch.cat(targets), torch.cat(snrs),
            labels, args.threshold,
        )
        metrics["parameter_count"] = predictor.parameter_count
        metrics["elapsed_seconds"] = time.perf_counter() - started
        result["encoders"][name] = metrics
        args.output.write_text(
            json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(
            f"encoder={name} micro_f1={metrics['micro_f1']:.4f} "
            f"macro_f1={metrics['macro_f1']:.4f} "
            f"mAP={metrics['macro_average_precision']:.4f}",
            flush=True,
        )
        del predictor, loader, probabilities, targets, snrs
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()
    print(f"results={args.output}", flush=True)


if __name__ == "__main__":
    main()

"""Benchmark frozen pretrained audio encoders on ``mix-dataset``.

The benchmark intentionally uses a shared linear-probe protocol so that the
comparison reflects pretrained representation quality rather than differences
in fine-tuning recipes. Samples are balanced by both class and SNR.
"""

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
import soundfile as sf
import torch
import torchaudio
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset


AST_CHECKPOINT = "MIT/ast-finetuned-audioset-10-10-0.4593"
PANNS_CHECKPOINT = Path("checkpoints/pretrained/Cnn14_mAP=0.431.pth")
BEATS_CHECKPOINT = Path(
    "checkpoints/pretrained/"
    "BEATs_iter3_plus_AS2M_finetuned_on_AS2M_cpt2.pt"
)
ENCODER_BATCH_SIZES = {"ast": 8, "panns": 16, "beats": 16}


class ManifestAudioDataset(Dataset):
    def __init__(self, root: Path, rows: list[dict[str, str]], label_to_index: dict[str, int]):
        self.root = root
        self.rows = rows
        self.label_to_index = label_to_index

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int):
        row = self.rows[index]
        waveform, sample_rate = sf.read(
            self.root / row["mixture_path"], dtype="float32", always_2d=True
        )
        waveform = waveform.mean(axis=1)
        if sample_rate != 16_000:
            waveform = torchaudio.functional.resample(
                torch.from_numpy(waveform), sample_rate, 16_000
            ).numpy()
        target_length = 4 * 16_000
        waveform = waveform[:target_length]
        if waveform.shape[0] < target_length:
            waveform = np.pad(waveform, (0, target_length - waveform.shape[0]))
        return {
            "waveform": torch.from_numpy(waveform.copy()),
            "target": self.label_to_index[row["label_names"]],
            "snr": int(float(row["target_snr_db"])),
            "sample_id": row["sample_id"],
        }


def load_manifest(root: Path) -> list[dict[str, str]]:
    with (root / "manifest.csv").open(encoding="utf-8-sig", newline="") as file:
        return list(csv.DictReader(file))


def balanced_subset(
    rows: list[dict[str, str]], split: str, per_class: int, seed: int
) -> list[dict[str, str]]:
    groups: dict[tuple[str, int], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        if row["split"] == split:
            key = (row["label_names"], int(float(row["target_snr_db"])))
            groups[key].append(row)

    labels = sorted({label for label, _ in groups})
    snrs = sorted({snr for _, snr in groups})
    if per_class % len(snrs):
        raise ValueError(
            f"--{split}-per-class must be divisible by {len(snrs)} SNR levels"
        )
    per_group = per_class // len(snrs)
    rng = random.Random(seed + (0 if split == "train" else 10_000))
    selected = []
    for label in labels:
        for snr in snrs:
            candidates = groups[(label, snr)].copy()
            rng.shuffle(candidates)
            if len(candidates) < per_group:
                raise ValueError(
                    f"Not enough rows for {split}/{label}/{snr}: "
                    f"need {per_group}, found {len(candidates)}"
                )
            selected.extend(candidates[:per_group])
    rng.shuffle(selected)
    return selected


class ASTEncoder:
    name = "ast"

    def __init__(self, device: torch.device):
        from transformers import ASTFeatureExtractor, ASTModel

        self.device = device
        self.feature_extractor = ASTFeatureExtractor.from_pretrained(AST_CHECKPOINT)
        self.model = ASTModel.from_pretrained(AST_CHECKPOINT).to(device).eval()
        self.parameter_count = sum(parameter.numel() for parameter in self.model.parameters())

    @torch.inference_mode()
    def __call__(self, waveforms: torch.Tensor) -> torch.Tensor:
        features = self.feature_extractor(
            [waveform.numpy() for waveform in waveforms],
            sampling_rate=16_000,
            return_tensors="pt",
        )["input_values"].to(self.device)
        return self.model(input_values=features).pooler_output.float().cpu()


class PANNsEncoder:
    name = "panns"

    def __init__(self, device: torch.device):
        from panns_inference.models import Cnn14

        if not PANNS_CHECKPOINT.exists():
            raise FileNotFoundError(PANNS_CHECKPOINT)
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
        self.parameter_count = sum(parameter.numel() for parameter in self.model.parameters())

    @torch.inference_mode()
    def __call__(self, waveforms: torch.Tensor) -> torch.Tensor:
        waveforms = self.resample(waveforms.to(self.device))
        return self.model(waveforms, None)["embedding"].float().cpu()


class BEATsEncoder:
    name = "beats"

    def __init__(self, device: torch.device):
        if not BEATS_CHECKPOINT.exists():
            raise FileNotFoundError(BEATS_CHECKPOINT)
        beats_source = Path("third_party/beats").resolve()
        sys.path.insert(0, str(beats_source))
        from BEATs import BEATs, BEATsConfig

        checkpoint = torch.load(BEATS_CHECKPOINT, map_location="cpu", weights_only=True)
        self.model = BEATs(BEATsConfig(checkpoint["cfg"]))
        self.model.load_state_dict(checkpoint["model"])
        # Keep the AudioSet-fine-tuned backbone, but expose its representation
        # instead of the original 527-class prediction head.
        self.model.predictor = None
        self.model.to(device).eval()
        self.device = device
        self.parameter_count = sum(parameter.numel() for parameter in self.model.parameters())

    @torch.inference_mode()
    def __call__(self, waveforms: torch.Tensor) -> torch.Tensor:
        sequence, _ = self.model.extract_features(waveforms.to(self.device))
        return sequence.mean(dim=1).float().cpu()


def create_encoder(name: str, device: torch.device):
    factories = {"ast": ASTEncoder, "panns": PANNsEncoder, "beats": BEATsEncoder}
    return factories[name](device)


def extract_embeddings(encoder, loader: DataLoader, split: str):
    embeddings, targets, snrs, sample_ids = [], [], [], []
    started = time.perf_counter()
    for batch_index, batch in enumerate(loader, start=1):
        embeddings.append(encoder(batch["waveform"]).numpy())
        targets.append(batch["target"].numpy())
        snrs.append(batch["snr"].numpy())
        sample_ids.extend(batch["sample_id"])
        if batch_index == 1 or batch_index % 25 == 0 or batch_index == len(loader):
            print(
                f"encoder={encoder.name} split={split} "
                f"batch={batch_index}/{len(loader)}",
                flush=True,
            )
    return {
        "x": np.concatenate(embeddings),
        "y": np.concatenate(targets),
        "snr": np.concatenate(snrs),
        "sample_id": np.asarray(sample_ids),
        "seconds": time.perf_counter() - started,
    }


def evaluate_probe(train_data, validation_data, seed: int):
    classifier = make_pipeline(
        StandardScaler(),
        LogisticRegression(
            C=1.0,
            max_iter=500,
            solver="lbfgs",
            random_state=seed,
        ),
    )
    started = time.perf_counter()
    classifier.fit(train_data["x"], train_data["y"])
    training_seconds = time.perf_counter() - started
    predictions = classifier.predict(validation_data["x"])
    metrics = {
        "accuracy": accuracy_score(validation_data["y"], predictions),
        "macro_f1": f1_score(validation_data["y"], predictions, average="macro"),
        "weighted_f1": f1_score(validation_data["y"], predictions, average="weighted"),
        "probe_training_seconds": training_seconds,
        "per_snr": {},
    }
    for snr in sorted(np.unique(validation_data["snr"])):
        mask = validation_data["snr"] == snr
        metrics["per_snr"][str(int(snr))] = {
            "samples": int(mask.sum()),
            "accuracy": accuracy_score(validation_data["y"][mask], predictions[mask]),
            "macro_f1": f1_score(
                validation_data["y"][mask], predictions[mask], average="macro"
            ),
        }
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("mix-dataset"))
    parser.add_argument(
        "--encoders", nargs="+", choices=("ast", "panns", "beats"),
        default=("ast", "panns", "beats"),
    )
    parser.add_argument("--train-per-class", type=int, default=96)
    parser.add_argument("--validation-per-class", type=int, default=30)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument(
        "--output", type=Path,
        default=Path("benchmark_results/pretrained_linear_probe.json"),
    )
    parser.add_argument(
        "--cache-dir", type=Path, default=Path("benchmark_results/embedding_cache")
    )
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    labels = (args.data_root / "labels.txt").read_text(encoding="utf-8").splitlines()
    label_to_index = {label: index for index, label in enumerate(labels)}
    rows = load_manifest(args.data_root)
    train_rows = balanced_subset(rows, "train", args.train_per_class, args.seed)
    validation_rows = balanced_subset(
        rows, "validation", args.validation_per_class, args.seed
    )
    print(
        f"device={device} labels={len(labels)} train={len(train_rows)} "
        f"validation={len(validation_rows)}",
        flush=True,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.cache_dir.mkdir(parents=True, exist_ok=True)
    results = {
        "protocol": "frozen_encoder_linear_probe",
        "seed": args.seed,
        "train_per_class": args.train_per_class,
        "validation_per_class": args.validation_per_class,
        "train_samples": len(train_rows),
        "validation_samples": len(validation_rows),
        "encoders": {},
    }

    for encoder_name in args.encoders:
        cache_path = args.cache_dir / (
            f"{encoder_name}_train{args.train_per_class}_"
            f"validation{args.validation_per_class}_seed{args.seed}.npz"
        )
        encoder = None
        if cache_path.exists():
            print(f"encoder={encoder_name} cache={cache_path}", flush=True)
            cached = np.load(cache_path)
            train_data = {key[6:]: cached[key] for key in cached if key.startswith("train_")}
            validation_data = {
                key[11:]: cached[key] for key in cached if key.startswith("validation_")
            }
            parameter_count = int(cached["parameter_count"])
        else:
            encoder = create_encoder(encoder_name, device)
            parameter_count = encoder.parameter_count
            batch_size = ENCODER_BATCH_SIZES[encoder_name]
            train_loader = DataLoader(
                ManifestAudioDataset(args.data_root, train_rows, label_to_index),
                batch_size=batch_size,
                num_workers=args.workers,
                pin_memory=device.type == "cuda",
                persistent_workers=args.workers > 0,
            )
            validation_loader = DataLoader(
                ManifestAudioDataset(args.data_root, validation_rows, label_to_index),
                batch_size=batch_size,
                num_workers=args.workers,
                pin_memory=device.type == "cuda",
                persistent_workers=args.workers > 0,
            )
            train_data = extract_embeddings(encoder, train_loader, "train")
            validation_data = extract_embeddings(encoder, validation_loader, "validation")
            np.savez_compressed(
                cache_path,
                parameter_count=np.asarray(parameter_count),
                **{f"train_{key}": value for key, value in train_data.items()},
                **{
                    f"validation_{key}": value
                    for key, value in validation_data.items()
                },
            )

        metrics = evaluate_probe(train_data, validation_data, args.seed)
        metrics.update(
            {
                "parameter_count": parameter_count,
                "embedding_dimension": int(train_data["x"].shape[1]),
                "train_extraction_seconds": float(train_data.get("seconds", 0.0)),
                "validation_extraction_seconds": float(
                    validation_data.get("seconds", 0.0)
                ),
            }
        )
        results["encoders"][encoder_name] = metrics
        args.output.write_text(
            json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(
            f"encoder={encoder_name} accuracy={metrics['accuracy']:.4f} "
            f"macro_f1={metrics['macro_f1']:.4f}",
            flush=True,
        )

        del encoder, train_data, validation_data
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

    print(f"results={args.output}", flush=True)


if __name__ == "__main__":
    main()

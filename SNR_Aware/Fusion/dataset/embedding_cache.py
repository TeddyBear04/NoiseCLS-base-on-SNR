"""Freeze the three SNR branches and turn each clip into one embedding row."""

from __future__ import annotations

import csv
import importlib.util
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor, nn
from torch.utils.data import Dataset

SNR_AWARE_ROOT = Path(__file__).resolve().parents[2]
BEATS_ROOT = SNR_AWARE_ROOT / "BEATs_Experts"
DPCRN_ROOT = SNR_AWARE_ROOT / "DPCRN_Noise_Target"

BRANCH_DIMS: dict[str, int] = {"beats_low": 768, "beats_mid": 768, "dpcrn_high": 256}


def _load_module(path: Path, alias: str) -> Any:
    """Load a module by file path under a unique name.

    ``BEATs_Experts`` and ``DPCRN_Noise_Target`` each contain a package called
    ``models``. Importing one by package name caches it as ``sys.modules["models"]``
    and makes the other unreachable, so both branches could never be loaded in one
    process. Loading by path under an alias sidesteps the shared name entirely.
    """
    if alias in sys.modules:
        return sys.modules[alias]
    spec = importlib.util.spec_from_file_location(alias, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {alias} from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[alias] = module
    spec.loader.exec_module(module)
    return module


def load_beats_branch(
    pretrained: Path, expert_checkpoint: Path, device: torch.device
) -> tuple[nn.Module, list[str]]:
    """Rebuild a fine-tuned BEATs expert: base weights plus its encoder delta."""
    loader = _load_module(BEATS_ROOT / "models" / "beats_loader.py", "fusion_beats_loader")
    BEATs, BEATsConfig = loader.load_beats_classes()
    base = torch.load(pretrained, map_location="cpu", weights_only=True)
    model = BEATs(BEATsConfig(base["cfg"]))
    model.load_state_dict(base["model"])
    model.predictor = None

    expert = torch.load(expert_checkpoint, map_location="cpu", weights_only=True)
    missing, unexpected = model.load_state_dict(expert["encoder_delta"], strict=False)
    if unexpected:
        raise ValueError(f"Unexpected keys in {expert_checkpoint}: {sorted(unexpected)[:5]}")
    model.to(device).eval()
    for parameter in model.parameters():
        parameter.requires_grad = False
    return model, list(expert["labels"])


def load_dpcrn_branch(checkpoint: Path, device: torch.device) -> tuple[nn.Module, list[str]]:
    module = _load_module(DPCRN_ROOT / "models" / "dpcrn_noise.py", "fusion_dpcrn_noise")
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    model_config = payload["config"]["model"]
    model = module.DPCRNNoiseClassifier(
        classes_num=len(payload["labels"]),
        n_fft=model_config["n_fft"],
        hop_length=model_config["hop_length"],
        encoder_channels=tuple(model_config["encoder_channels"]),
        dprnn_blocks=model_config["dprnn_blocks"],
        embedding_dim=model_config["embedding_dim"],
        classifier_dropout=model_config["classifier_dropout"],
        bidirectional_time=model_config["bidirectional_time"],
    )
    model.load_state_dict(payload["model"])
    model.to(device).eval()
    for parameter in model.parameters():
        parameter.requires_grad = False
    return model, list(payload["labels"])


@torch.inference_mode()
def embed_beats(model: nn.Module, waveform: Tensor) -> Tensor:
    sequence, _ = model.extract_features(waveform)
    return sequence.mean(dim=1).float()


@torch.inference_mode()
def embed_dpcrn(model: nn.Module, waveform: Tensor) -> Tensor:
    return model(waveform)["embedding"].float()


BRANCH_ORDER = ("beats_low", "beats_mid", "dpcrn_high")
EMBEDDERS = {"beats_low": embed_beats, "beats_mid": embed_beats, "dpcrn_high": embed_dpcrn}


def write_cache(
    branches: dict[str, nn.Module], loader: Any, split: str, out_dir: Path, device: torch.device
) -> None:
    """Embed every clip with every branch, preserving loader order."""
    out_dir.mkdir(parents=True, exist_ok=True)
    collected: dict[str, list[np.ndarray]] = {name: [] for name in BRANCH_ORDER}
    targets: list[int] = []
    snrs: list[float] = []

    for index, batch in enumerate(loader, start=1):
        waveform = batch["mixture"].to(device, non_blocking=True)
        for name in BRANCH_ORDER:
            vectors = EMBEDDERS[name](branches[name], waveform)
            collected[name].append(vectors.cpu().numpy().astype(np.float16))
        targets.extend(batch["target"].tolist())
        snrs.extend(batch["snr"].tolist())
        if index == 1 or index % 50 == 0:
            print(f"cache split={split} batch={index}/{len(loader)}", flush=True)

    for name in BRANCH_ORDER:
        stacked = np.concatenate(collected[name])
        if stacked.shape != (len(targets), BRANCH_DIMS[name]):
            raise ValueError(
                f"branch {name} produced {stacked.shape}, expected "
                f"{(len(targets), BRANCH_DIMS[name])}"
            )
        # Cheap insurance, not a suspected bug: DPCRN embeddings are bounded
        # to (-1, 1) by construction and BEATs embeddings sit around 1-10, so
        # neither overflow nor NaN should occur in the float16 cast below.
        # But the cache costs GPU-hours to regenerate, and float16 silently
        # turns an overflow into inf and silently propagates NaN, so a single
        # poisoned row would corrupt every downstream metric with no signal.
        non_finite_rows = int((~np.isfinite(stacked)).any(axis=1).sum())
        if non_finite_rows:
            raise ValueError(
                f"branch {name} produced {non_finite_rows} row(s) with non-finite "
                "values (inf/NaN) after casting to float16"
            )
        np.save(out_dir / f"{split}_{name}.npy", stacked)

    with (out_dir / f"{split}_meta.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["row_index", "label_index", "snr_db"])
        writer.writerows(zip(range(len(targets)), targets, snrs))
    print(f"cached split={split} rows={len(targets)}", flush=True)


class EmbeddingCacheDataset(Dataset[dict[str, Any]]):
    """Read one split of the cache. Branch order is fixed by BRANCH_ORDER."""

    def __init__(self, cache_dir: str | Path, split: str) -> None:
        cache_dir = Path(cache_dir)
        self.split = split
        matrices = []
        for name in BRANCH_ORDER:
            matrix = np.load(cache_dir / f"{split}_{name}.npy")
            if matrix.shape[1] != BRANCH_DIMS[name]:
                raise ValueError(f"branch {name} has width {matrix.shape[1]}")
            matrices.append(matrix)

        row_counts = {matrix.shape[0] for matrix in matrices}
        with (cache_dir / f"{split}_meta.csv").open(encoding="utf-8") as handle:
            meta = list(csv.DictReader(handle))
        row_counts.add(len(meta))
        if len(row_counts) != 1:
            raise ValueError(
                f"{split} cache has mismatched row counts across branches and meta: "
                f"{sorted(row_counts)}"
            )

        self.features = torch.from_numpy(np.concatenate(matrices, axis=1)).float()
        self.targets = torch.tensor([int(row["label_index"]) for row in meta], dtype=torch.long)
        self.snrs = torch.tensor([float(row["snr_db"]) for row in meta], dtype=torch.float32)
        self.feature_dim = self.features.shape[1]
        labels_file = cache_dir / "labels.txt"
        self.labels = (
            labels_file.read_text(encoding="utf-8").splitlines() if labels_file.exists() else []
        )

    def __len__(self) -> int:
        return self.features.shape[0]

    def __getitem__(self, index: int) -> dict[str, Any]:
        return {
            "features": self.features[index],
            "target": self.targets[index],
            "snr": self.snrs[index],
        }

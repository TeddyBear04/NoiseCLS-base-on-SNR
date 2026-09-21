"""Freeze the three SNR branches and turn each clip into one embedding row."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn

SNR_AWARE_ROOT = Path(__file__).resolve().parents[2]
BEATS_ROOT = SNR_AWARE_ROOT / "BEATs_Experts"
DPCRN_ROOT = SNR_AWARE_ROOT / "DPCRN_Noise_Target"

BRANCH_DIMS: dict[str, int] = {"beats_low": 768, "beats_mid": 768, "dpcrn_high": 256}


def _import_from(root: Path, module: str) -> Any:
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    return __import__(module, fromlist=["*"])


def load_beats_branch(
    pretrained: Path, expert_checkpoint: Path, device: torch.device
) -> tuple[nn.Module, list[str]]:
    """Rebuild a fine-tuned BEATs expert: base weights plus its encoder delta."""
    loader = _import_from(BEATS_ROOT, "models.beats_loader")
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
    module = _import_from(DPCRN_ROOT, "models.dpcrn_noise")
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

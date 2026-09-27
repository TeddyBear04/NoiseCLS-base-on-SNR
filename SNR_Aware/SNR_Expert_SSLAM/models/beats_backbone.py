"""BEATs backbone behind the same interface as `models.sslam.SSLAMEncoder`.

`forward(waveform) -> (patch tokens, pooled vector)`, `blocks()` for unfreezing and
layer-wise lr decay, and an optional train-time augmentation hook. Everything else
in this project - FiLM, KD, CRD, band-only evaluation, reports - stays unchanged.

The BEATs source is the vendored copy in `../BEATs_Experts/models/beats`, the same
one the mid-expert project used, so the two backbones' numbers are comparable.

Why a separate forward instead of `BEATs.extract_features`: SpecAugment has to act
on the fbank, and `extract_features` computes the fbank and feeds it straight to the
patch embedding with no hook in between. The steps below are the same as its own,
minus the padding-mask branches this project never uses (every clip is 4.0 s).
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
from torch import Tensor, nn

from expert_lib import roll_waveform, spec_augment

_VENDOR = Path(__file__).resolve().parents[2] / "BEATs_Experts" / "models"


def _beats_classes():
    if str(_VENDOR) not in sys.path:
        sys.path.insert(0, str(_VENDOR))
    from beats_loader import load_beats_classes
    return load_beats_classes()


def load_beats(device: torch.device, checkpoint_path: Path, dropout: float,
               layerdrop: float):
    """Frozen-by-default BEATs plus its raw checkpoint (for the AudioSet head rows).

    `dropout` and `layerdrop` override the checkpoint's cfg: the AS2M-finetuned
    checkpoint ships dropout 0.0, while the BEATs paper fine-tunes with 0.1 / 0.1.
    `layer_wise_gradient_decay_ratio` is forced to 1.0 because layer-wise decay is
    done with optimizer groups here (see `expert_lib.layer_decay_scales`).
    """
    BEATs, BEATsConfig = _beats_classes()
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    cfg = dict(checkpoint["cfg"])
    cfg.update(dropout=dropout, encoder_layerdrop=layerdrop,
               layer_wise_gradient_decay_ratio=1.0)
    model = BEATs(BEATsConfig(cfg))
    model.load_state_dict(checkpoint["model"])
    model.predictor = None
    model.to(device).eval()
    for parameter in model.parameters():
        parameter.requires_grad = False
    return model, checkpoint


def beats_head_rows(checkpoint: dict, label_mids: list[str]) -> tuple[Tensor, Tensor]:
    """The checkpoint's AudioSet predictor rows for our labels, in our label order.

    BEATs predicts per token and averages the logits; a linear head on the
    mean-pooled vector is the same function, so these rows warm-start it exactly.
    """
    source = {mid: int(index) for index, mid in checkpoint["label_dict"].items()}
    missing = [mid for mid in label_mids if mid not in source]
    if missing:
        raise KeyError(f"labels missing from the BEATs label_dict: {missing}")
    rows = torch.tensor([source[mid] for mid in label_mids])
    weight = checkpoint["model"]["predictor.weight"][rows].clone()
    bias = checkpoint["model"]["predictor.bias"][rows].clone()
    return weight, bias


class BEATsEncoder(nn.Module):
    """BEATs with an optional train-time augmentation hook, mean-pooled."""

    def __init__(self, model):
        super().__init__()
        self.model = model
        self.pooling = "mean_patches"
        self.augment: dict = {}
        self.generator = torch.Generator().manual_seed(0)

    def blocks(self) -> nn.ModuleList:
        return self.model.encoder.layers

    def forward(self, waveform: Tensor) -> tuple[Tensor, Tensor]:
        augment = self.training and bool(self.augment)
        if augment and self.augment.get("roll"):
            waveform = roll_waveform(waveform, self.generator)
        fbank = self.model.preprocess(waveform)
        if augment and self.augment.get("spec_ratio", 0) > 0:
            fbank = spec_augment(fbank, self.augment["spec_ratio"], self.generator)

        features = self.model.patch_embedding(fbank.unsqueeze(1))
        features = features.reshape(features.shape[0], features.shape[1], -1)
        features = self.model.layer_norm(features.transpose(1, 2))
        if self.model.post_extract_proj is not None:
            features = self.model.post_extract_proj(features)
        x, _ = self.model.encoder(self.model.dropout_input(features), padding_mask=None)
        return x, x.mean(dim=1)

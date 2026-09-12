"""Controlled BEATs fine-tuning with Asymmetric Loss instead of weighted BCE."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import torch
from torch import nn

import finetune_beats_21 as experiment


class AsymmetricLoss(nn.Module):
    """Numerically stable ASL for multi-label classification.

    The ``pos_weight`` argument is accepted for compatibility with the baseline
    training loop but intentionally ignored. ASL controls positive/negative
    imbalance through asymmetric focusing and negative probability clipping.
    """

    def __init__(
        self,
        pos_weight=None,
        gamma_negative: float = 4.0,
        gamma_positive: float = 1.0,
        clip: float = 0.05,
        eps: float = 1e-8,
    ) -> None:
        super().__init__()
        self.gamma_negative = gamma_negative
        self.gamma_positive = gamma_positive
        self.clip = clip
        self.eps = eps

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        positive_probability = torch.sigmoid(logits)
        negative_probability = 1.0 - positive_probability
        if self.clip > 0:
            negative_probability = (negative_probability + self.clip).clamp(max=1.0)
        log_likelihood = (
            targets * torch.log(positive_probability.clamp(min=self.eps))
            + (1.0 - targets)
            * torch.log(negative_probability.clamp(min=self.eps))
        )
        with torch.no_grad():
            probability = (
                positive_probability * targets
                + negative_probability * (1.0 - targets)
            )
            gamma = (
                self.gamma_positive * targets
                + self.gamma_negative * (1.0 - targets)
            )
            focusing_weight = torch.pow(1.0 - probability, gamma)
        return -(log_likelihood * focusing_weight).mean()


_adamw = torch.optim.AdamW


def deduplicated_adamw(parameter_groups, *args, **kwargs):
    seen: set[int] = set()
    groups = []
    for source in parameter_groups:
        group = dict(source)
        unique = []
        for parameter in source["params"]:
            identity = id(parameter)
            if identity not in seen:
                seen.add(identity)
                unique.append(parameter)
        group["params"] = unique
        groups.append(group)
    return _adamw(groups, *args, **kwargs)


def argument_path(flag: str, default: str) -> Path:
    if flag in sys.argv:
        return Path(sys.argv[sys.argv.index(flag) + 1])
    return Path(default)


if __name__ == "__main__":
    experiment.nn.BCEWithLogitsLoss = AsymmetricLoss
    experiment.torch.optim.AdamW = deduplicated_adamw
    print("loss=ASL gamma_negative=4 gamma_positive=1 clip=0.05", flush=True)
    experiment.main()
    results_path = argument_path(
        "--results", "benchmark_results/beats_21_last4.json"
    )
    if results_path.exists():
        result = json.loads(results_path.read_text(encoding="utf-8"))
        result["loss_configuration"] = {
            "name": "AsymmetricLoss",
            "gamma_negative": 4.0,
            "gamma_positive": 1.0,
            "negative_probability_clip": 0.05,
            "pos_weight_used": False,
        }
        results_path.write_text(
            json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
        )

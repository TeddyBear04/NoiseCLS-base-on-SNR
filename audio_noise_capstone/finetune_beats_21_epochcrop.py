"""Fine-tune BEATs with a new random long-clip crop on every sample access."""

from __future__ import annotations

import torch

import finetune_beats_21 as experiment
from noise_pipeline.audio21_crops import EpochRandomCropAudio21Dataset


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


if __name__ == "__main__":
    experiment.Audio21Dataset = EpochRandomCropAudio21Dataset
    experiment.torch.optim.AdamW = deduplicated_adamw
    experiment.main()

import json
from pathlib import Path

import torch

from tasks.run_36 import build_scheduler, load_config, validate_config

CONFIG = Path(__file__).resolve().parents[1] / "config" / "train_config.json"


def test_shipped_config_is_valid():
    config = load_config(CONFIG)
    validate_config(config)
    assert config["model"]["encoder_channels"] == [32, 32, 32, 64, 128]
    assert config["model"]["embedding_dim"] == 256
    assert config["training"]["patience"] == 15


def test_scheduler_warms_up_then_decays():
    config = load_config(CONFIG)
    parameter = torch.nn.Parameter(torch.zeros(1))
    optimizer = torch.optim.AdamW([parameter], lr=config["training"]["learning_rate"])
    scheduler = build_scheduler(optimizer, config, steps_per_epoch=10)
    start = optimizer.param_groups[0]["lr"]
    for _ in range(10 * config["training"]["warmup_epochs"]):
        optimizer.step()
        scheduler.step()
    peak = optimizer.param_groups[0]["lr"]
    for _ in range(10 * config["training"]["epochs"]):
        optimizer.step()
        scheduler.step()
    assert start < peak
    assert optimizer.param_groups[0]["lr"] < peak

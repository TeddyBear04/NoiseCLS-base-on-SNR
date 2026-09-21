import json
from pathlib import Path

import torch

from dataset.embedding_cache import BRANCH_DIMS
from models.fusion_head import FusionHead
from tasks.run_36 import ABLATION_ROWS, load_config, set_seed, validate_config

CONFIG = Path(__file__).resolve().parents[1] / "config" / "train_config.json"


def test_shipped_config_is_valid():
    config = load_config(CONFIG)
    validate_config(config)
    assert config["training"]["lambda_snr"] == 0.1


def test_ablation_covers_each_branch_alone_and_all_together():
    names = [name for name, _ in ABLATION_ROWS]
    assert names == ["beats_low", "beats_mid", "dpcrn_high", "fusion"]
    assert ABLATION_ROWS[-1][1] == ("beats_low", "beats_mid", "dpcrn_high")


def test_set_seed_makes_fusion_head_init_reproducible():
    """Pins the property ablation's per-row reseed depends on.

    Without a fresh set_seed() before each ABLATION_ROWS iteration, a row's
    weight init depends on how many previous rows already trained -- this
    checks that seeding alone is enough to make two builds identical.
    """
    set_seed(2026)
    first = FusionHead(BRANCH_DIMS)
    set_seed(2026)
    second = FusionHead(BRANCH_DIMS)
    for key in first.state_dict():
        assert torch.equal(first.state_dict()[key], second.state_dict()[key])

import json
from pathlib import Path

from tasks.run_36 import ABLATION_ROWS, load_config, validate_config

CONFIG = Path(__file__).resolve().parents[1] / "config" / "train_config.json"


def test_shipped_config_is_valid():
    config = load_config(CONFIG)
    validate_config(config)
    assert config["training"]["lambda_snr"] == 0.1


def test_ablation_covers_each_branch_alone_and_all_together():
    names = [name for name, _ in ABLATION_ROWS]
    assert names == ["beats_low", "beats_mid", "dpcrn_high", "fusion"]
    assert ABLATION_ROWS[-1][1] == ("beats_low", "beats_mid", "dpcrn_high")

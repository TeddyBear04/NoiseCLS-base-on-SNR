"""Run the existing 36-label single-label BEATs stages from one JSON config."""

from __future__ import annotations

import argparse
import json
import runpy
import sys
from pathlib import Path

from config.paths import PROJECT_ROOT


STAGES = {
    "head36": ("train_beats_head.py", "head_training"),
    "finetune36": ("finetune_beats.py", "finetuning"),
    "test36": ("evaluate_beats_36_test.py", "evaluation"),
}


def option_list(values: dict) -> list[str]:
    arguments: list[str] = []
    for key, value in values.items():
        if value is None:
            continue
        arguments.extend([f"--{key.replace('_', '-')}", str(value)])
    return arguments


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=STAGES)
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "config" / "train_config.json")
    arguments, overrides = parser.parse_known_args()
    config = json.loads(arguments.config.read_text(encoding="utf-8"))
    dataset_root = config["dataset_splitter"]["dataset_path"]
    script, section = STAGES[arguments.stage]
    stage_values = {"data_root": dataset_root, **config[section]}
    for key in ("snr_min_db", "snr_max_db"):
        if key in config["dataset_splitter"]:
            stage_values[key] = config["dataset_splitter"][key]
    if "pretrained" in config:
        stage_values["pretrained_checkpoint"] = config["pretrained"]["checkpoint"]
    sys.argv = [script, *option_list(stage_values), *overrides]
    runpy.run_path(str(PROJECT_ROOT / script), run_name="__main__")


if __name__ == "__main__":
    main()

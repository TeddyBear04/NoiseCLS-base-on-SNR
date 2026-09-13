"""Run the 36-label mixture/noise fusion stages from one JSON config."""

from __future__ import annotations

import argparse
import json
import runpy
import sys
from pathlib import Path

from config.paths import PROJECT_ROOT


# Each stage merges its config sections in order; later sections win.
STAGES = {
    "separator36": ("train_separator.py", ("separator", "separator_training")),
    "head36": ("train_fusion_head.py", ("head_training",)),
    "finetune36": ("finetune_fusion.py", ("finetuning",)),
    "test36": ("evaluate_fusion_test.py", ("evaluation",)),
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
    script, sections = STAGES[arguments.stage]
    stage_values = {"data_root": dataset_root}
    for section in sections:
        stage_values.update(config[section])
    configured_output_dir = stage_values.get("output_dir")
    output_dir = (
        Path(configured_output_dir)
        if configured_output_dir is not None
        else Path(stage_values["output"]).parent
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "train_config.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    sys.argv = [script, *option_list(stage_values), *overrides]
    runpy.run_path(str(PROJECT_ROOT / script), run_name="__main__")


if __name__ == "__main__":
    main()

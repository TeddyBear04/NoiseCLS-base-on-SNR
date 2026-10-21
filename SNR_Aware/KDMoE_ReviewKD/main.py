from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from online_pipeline import (
    HERE,
    load_config,
    load_frcrn,
    resolve_device,
    train,
    validate_config,
)


def command_check(config: dict) -> None:
    paths = validate_config(config)
    device = resolve_device(config["runtime"]["device"])
    separator = load_frcrn(paths["frcrn_config"], paths["frcrn_checkpoint"], device)
    report = {
        "status": "ok",
        "device": str(device),
        "frcrn_checkpoint": str(paths["frcrn_checkpoint"]),
        "frcrn_parameters": separator.num_parameters(),
        "frcrn_trainable_parameters": separator.num_parameters(trainable_only=True),
        "frcrn_input": "fp32 cache" if "frcrn_cache_dir" in paths else "online (student autocast)",
    }
    import torch

    packed = torch.load(paths["teacher_cache"], map_location="cpu", weights_only=False)
    report["teacher_mode"] = "cached_logits"
    report["teacher_cache"] = str(paths["teacher_cache"])
    report["teacher_logits"] = {split: list(packed[split]["logits36"].shape) for split in ("train", "validation")}
    print(json.dumps(report, indent=2, ensure_ascii=False))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Frozen FRCRN + cached BEATs teacher logits + KD-MoE student"
    )
    parser.add_argument("command", choices=("check", "train", "smoke"))
    parser.add_argument(
        "--config", type=Path, default=HERE / "config" / "train_panns_reviewkd.json"
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s"
    )
    config = load_config(args.config)
    if args.command == "check":
        command_check(config)
        return
    result = train(config, smoke=args.command == "smoke")
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

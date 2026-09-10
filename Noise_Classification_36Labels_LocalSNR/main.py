from __future__ import annotations

import argparse
import json
import logging
import os
import random
import tempfile
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.optim as optim

from config import TrainConfig
from dataset import NoiseDataLoaderManager
from models import build_local_snr_model
from tasks import LocalSNRTrainer, split_parameter_groups
from utils import format_snr_table, log_model_profile

PROJECT_ROOT = Path(__file__).resolve().parent

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = torch.cuda.is_available()


def resolve_device(requested: Optional[str] = None) -> torch.device:
    if requested:
        if requested.startswith("cuda") and not torch.cuda.is_available():
            logger.warning("CUDA was requested but is unavailable; using CPU")
            return torch.device("cpu")
        return torch.device(requested)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_config(config_path: str, dataset_path: Optional[str] = None) -> TrainConfig:
    """Load a config, letting the caller redirect where the dataset lives.

    Relative dataset paths in the JSON resolve against the project directory
    that holds the config file. A config copied next to a checkpoint therefore
    resolves "../36_labels" from the checkpoint directory, which is the wrong
    place, so re-reading a finished run needs an explicit override.
    """
    config = TrainConfig.from_json(config_path)
    override = dataset_path or os.environ.get("NOISE_DATASET_PATH")
    if override:
        config.dataset_splitter.dataset_path = str(Path(override).expanduser().resolve())
    return config


def check_dataset(config_path: str, dataset_path: Optional[str] = None) -> None:
    """Validate manifests, labels, paths, shapes, and one batch per split."""
    config = load_config(config_path, dataset_path)
    manager = NoiseDataLoaderManager(
        dataset_config=config.dataset_splitter,
        audio_config=config.audio_features,
        batch_size=2,
        num_workers=0,
        cache_audio=False,
        pin_memory=False,
        seed=config.random_seed,
        classes_num=config.model.classes_num,
        augmentation_config=config.augmentation,
    )
    logger.info("Labels (%d): %s", len(manager.label_names), manager.label_names)
    for split in ("train", "val", "test"):
        batch = next(iter(manager.get_dataloader(split, shuffle=False)))
        reconstruction_error = (
            batch["waveform"] - batch["clean_waveform"] - batch["noise_waveform"]
        ).abs().max()
        if float(reconstruction_error) > 1e-5:
            raise ValueError(
                f"{split} violates mixture = clean + noise; max error "
                f"{float(reconstruction_error):.3e}"
            )
        logger.info(
            "%s check: clips=%d windows=%d waveform=%s target=%s local_snr=%s "
            "component_error=%.3e",
            split,
            len(manager.datasets[split].records),
            len(manager.datasets[split]),
            tuple(batch["waveform"].shape),
            tuple(batch["target"].shape),
            tuple(batch["local_snr_db"].shape),
            float(reconstruction_error),
        )
    logger.info("Dataset check passed")


def run_evaluation(
    config_path: str,
    checkpoint_path: str,
    device_name: Optional[str] = None,
    split: str = "test",
    output_path: Optional[str] = None,
    dataset_path: Optional[str] = None,
) -> dict:
    """Score a saved checkpoint on one split, without retraining.

    summary.json is only written once all three stages and the test pass have
    finished, so a run killed in its last epoch leaves a perfectly good model
    and no metrics at all. Point this at that checkpoint to recover them, and
    at the train_config.json the trainer stored beside it to be sure the
    architecture matches.
    """
    config = load_config(config_path, dataset_path)
    checkpoint_file = Path(checkpoint_path).expanduser().resolve()
    if not checkpoint_file.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_file}")
    device = resolve_device(device_name)

    loaders = NoiseDataLoaderManager(
        dataset_config=config.dataset_splitter,
        audio_config=config.audio_features,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        cache_audio=False,
        pin_memory=config.pin_memory,
        seed=config.random_seed,
        classes_num=config.model.classes_num,
    )
    model = build_local_snr_model(config.audio_features, config.model).to(device)
    checkpoint = torch.load(checkpoint_file, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint.get("model_state_dict", checkpoint), strict=True)
    label_names = list(checkpoint.get("label_names") or loaders.label_names)
    logger.info(
        "Scoring %s (stage %s, epoch %s) on the %s split",
        checkpoint_file.name,
        checkpoint.get("stage", "?"),
        checkpoint.get("epoch", "?"),
        split,
    )

    # The trainer owns the metric code, so reuse it rather than restate it. It
    # is given a scratch home because instantiating it writes labels.json and
    # the config, and scoring must not touch the directory being scored.
    with tempfile.TemporaryDirectory() as scratch:
        trainer = LocalSNRTrainer(
            model=model,
            optimizer=optim.AdamW(model.parameters()),
            device=device,
            config=config,
            checkpoint_directory=scratch,
            label_names=label_names,
            train_config_path=config_path,
        )
        metrics = trainer.evaluate(loaders.get_dataloader(split, shuffle=False))

    scalar_keys = (
        "loss", "classification_loss", "separation_loss", "local_snr_loss",
        "top1_accuracy", "top3_accuracy", "balanced_accuracy",
        "f1_macro", "f1_micro", "f1_weighted", "precision_macro", "recall_macro",
        "mAP", "macro_auc", "subset_accuracy",
        "local_snr_mae_db", "local_snr_rmse_db", "noise_si_sdr_db",
    )
    report = {
        "checkpoint": str(checkpoint_file),
        "config": str(Path(config_path).expanduser().resolve()),
        "split": split,
        "stage": checkpoint.get("stage"),
        "epoch": checkpoint.get("epoch"),
        "num_clips": int(metrics["num_clips"]),
        "num_windows": int(metrics["num_windows"]),
        **{key: float(metrics[key]) for key in scalar_keys if key in metrics},
        "per_label_top1_recall": {
            name: float(value)
            for name, value in zip(label_names, metrics["per_label_top1_recall"])
        },
        "snr_metrics": metrics.get("snr_metrics", {}),
    }
    destination = (
        Path(output_path).expanduser().resolve()
        if output_path
        else checkpoint_file.parent / f"evaluation_{split}.json"
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    logger.info(
        "%d clips | top-1 %.4f | top-3 %.4f | macro-F1 %.4f | mAP %.4f",
        report["num_clips"],
        report["top1_accuracy"],
        report["top3_accuracy"],
        report["f1_macro"],
        report["mAP"],
    )
    logger.info(
        "Local-SNR MAE %.3f dB | RMSE %.3f dB | noise SI-SDR %.3f dB",
        report["local_snr_mae_db"],
        report["local_snr_rmse_db"],
        report["noise_si_sdr_db"],
    )
    logger.info("Per-SNR breakdown:\n%s", format_snr_table(report["snr_metrics"]))
    logger.info("Wrote %s", destination)
    return metrics


def run_training(
    config_path: str,
    device_name: Optional[str] = None,
    dataset_path: Optional[str] = None,
) -> dict:
    config = load_config(config_path, dataset_path)
    dataset_root = Path(config.dataset_splitter.dataset_path)
    if not dataset_root.is_dir():
        raise FileNotFoundError(
            f"Dataset directory not found: {dataset_root}. "
            "Edit dataset_splitter.dataset_path or set NOISE_DATASET_PATH."
        )

    seed_everything(config.random_seed)
    device = resolve_device(device_name)
    logger.info("Device: %s", device)
    logger.info("Dataset: %s", dataset_root)
    logger.info(
        "Audio: %d Hz, %.1f-second windows, %.1f-second inference hop",
        config.audio_features.sample_rate,
        config.audio_features.clip_seconds,
        config.audio_features.inference_hop_seconds,
    )
    logger.info(
        "Task: supervised noise extraction + %d-class classification + Local-SNR; "
        "waveform augmentation=%s; dynamic-SNR augmentation=%s (p=%.2f)",
        config.model.classes_num,
        config.augmentation.enabled,
        config.dataset_splitter.dynamic_snr_enabled,
        config.dataset_splitter.dynamic_snr_probability,
    )

    loaders = NoiseDataLoaderManager(
        dataset_config=config.dataset_splitter,
        audio_config=config.audio_features,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        cache_audio=config.cache_audio,
        pin_memory=config.pin_memory,
        seed=config.random_seed,
        classes_num=config.model.classes_num,
        augmentation_config=config.augmentation,
    )
    train_loader = loaders.get_dataloader("train", shuffle=True)
    val_loader = loaders.get_dataloader("val", shuffle=False)
    test_loader = loaders.get_dataloader("test", shuffle=False)

    if config.dataset_splitter.signal_type != "mixture":
        raise ValueError("BlackFeatherLocalSNR requires dataset_splitter.signal_type='mixture'")
    model = build_local_snr_model(config.audio_features, config.model).to(device)
    logger.info(
        "Noise extractor: %s, %s parameters (%s in the rest of the model)",
        config.model.extractor_type,
        f"{sum(p.numel() for p in model.noise_extractor.parameters()):,}",
        f"{sum(p.numel() for p in model.parameters()) - sum(p.numel() for p in model.noise_extractor.parameters()):,}",
    )
    clip_samples = int(round(config.audio_features.sample_rate * config.audio_features.clip_seconds))
    if config.profile_model:
        dummy = torch.zeros(1, clip_samples, device=device)
        log_model_profile(model, dummy, "BlackFeatherLocalSNR")

    # AdamW's decoupled weight decay is the only L2 in play; nothing is added
    # to the loss. The extractor carries its own decay so that regularising the
    # classifier does not shrink the noise it is supposed to predict.
    regularization = config.regularization
    parameter_groups = split_parameter_groups(model, regularization)
    optimizer = optim.AdamW(parameter_groups, lr=config.learning_rate)
    for group in parameter_groups:
        logger.info(
            "Weight decay %.2e on %d tensors (%s parameters)",
            group["weight_decay"],
            len(group["params"]),
            f"{sum(p.numel() for p in group['params']):,}",
        )
    if regularization.l1_lambda > 0.0:
        # Only AudioTrainer adds an L1 term; this pipeline never reads it.
        logger.warning(
            "regularization.l1_lambda=%.2e has no effect in the Local-SNR trainer",
            regularization.l1_lambda,
        )
    if config.use_pos_weight:
        # pos_weight re-weights the positive side of an independent binary
        # decision, which cross-entropy does not have. Class weights would be
        # the equivalent knob, and the balanced 36-label splits do not need one.
        logger.warning("use_pos_weight has no effect with cross-entropy; ignoring it")

    # Use the registry/config name so training, feature extraction, and inference
    # resolve exactly the same checkpoint directory on every platform.
    model_output_dir = Path(config.ckpt_dir) / "BlackFeatherLocalSNR"
    trainer = LocalSNRTrainer(
        model=model,
        optimizer=optimizer,
        device=device,
        config=config,
        checkpoint_directory=str(model_output_dir),
        label_names=loaders.label_names,
        train_config_path=config_path,
    )
    return trainer.train(train_loader, val_loader, test_loader)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the noise classifier")
    parser.add_argument(
        "--config",
        default=str(PROJECT_ROOT / "config" / "train_config.json"),
        help="Path to train_config.json",
    )
    parser.add_argument("--device", default=None, help="Optional device override, e.g. cuda, cuda:0, cpu")
    parser.add_argument(
        "--check-data",
        action="store_true",
        help="Validate manifests and one batch per split without starting training",
    )
    parser.add_argument(
        "--evaluate",
        metavar="CHECKPOINT",
        default=None,
        help="Score a saved checkpoint on one split instead of training",
    )
    parser.add_argument(
        "--split",
        default="test",
        choices=("train", "val", "test"),
        help="Split to score with --evaluate",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Where --evaluate writes its report; defaults to evaluation_<split>.json "
        "beside the checkpoint",
    )
    parser.add_argument(
        "--dataset",
        default=None,
        help="Dataset directory, overriding the config and NOISE_DATASET_PATH. Needed "
        "when re-reading a train_config.json stored beside a checkpoint, whose "
        "relative dataset path no longer resolves from there",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.check_data:
        check_dataset(args.config, args.dataset)
        return
    if args.evaluate:
        run_evaluation(
            args.config,
            args.evaluate,
            args.device,
            args.split,
            args.output,
            args.dataset,
        )
        return
    result = run_training(args.config, args.device, args.dataset)
    logger.info("Training complete. Best checkpoint: %s", result["final_checkpoint"])


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import logging
import os
import random
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.optim as optim

from config import TrainConfig
from dataset import NoiseDataLoaderManager
from features import AudioFrontend
from models import AudioModel, build_backbone
from tasks import AudioTrainer
from utils import log_model_profile

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


def load_config(config_path: str) -> TrainConfig:
    config = TrainConfig.from_json(config_path)
    dataset_override = os.environ.get("NOISE_DATASET_PATH")
    if dataset_override:
        config.dataset_splitter.dataset_path = str(Path(dataset_override).expanduser().resolve())
    return config


def check_dataset(config_path: str) -> None:
    """Validate manifests, labels, paths, shapes, and one batch per split."""
    config = load_config(config_path)
    manager = NoiseDataLoaderManager(
        dataset_config=config.dataset_splitter,
        audio_config=config.audio_features,
        batch_size=2,
        num_workers=0,
        cache_audio=False,
        pin_memory=False,
        seed=config.random_seed,
        classes_num=config.model.classes_num,
    )
    logger.info("Labels (%d): %s", len(manager.label_names), manager.label_names)
    for split in ("train", "val", "test"):
        batch = next(iter(manager.get_dataloader(split, shuffle=False)))
        logger.info(
            "%s check: clips=%d windows=%d waveform=%s target=%s",
            split,
            len(manager.datasets[split].records),
            len(manager.datasets[split]),
            tuple(batch["waveform"].shape),
            tuple(batch["target"].shape),
        )
    logger.info("Dataset check passed")


def run_training(config_path: str, device_name: Optional[str] = None) -> dict:
    config = load_config(config_path)
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
        "Task: %d-label multi-label classification; dynamic-SNR augmentation=%s (p=%.2f)",
        config.model.classes_num,
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
    )
    train_loader = loaders.get_dataloader("train", shuffle=True)
    val_loader = loaders.get_dataloader("val", shuffle=False)
    test_loader = loaders.get_dataloader("test", shuffle=False)

    frontend = AudioFrontend(config.audio_features)
    backbone = build_backbone(config.model)
    model = AudioModel(frontend, backbone).to(device)
    clip_samples = int(round(config.audio_features.sample_rate * config.audio_features.clip_seconds))
    if config.profile_model:
        dummy = torch.zeros(1, clip_samples, device=device)
        log_model_profile(model, dummy, backbone.get_name())

    optimizer = optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    pos_weight = (
        loaders.positive_class_weights(config.max_pos_weight).to(device)
        if config.use_pos_weight
        else None
    )
    if pos_weight is not None:
        logger.info("Positive-class weights: %s", [round(value, 3) for value in pos_weight.tolist()])

    # Use the registry/config name so training, feature extraction, and inference
    # resolve exactly the same checkpoint directory on every platform.
    model_output_dir = Path(config.ckpt_dir) / config.model.backbone
    trainer = AudioTrainer(
        model=model,
        optimizer=optimizer,
        device=device,
        ckpt_dir=str(model_output_dir),
        label_names=loaders.label_names,
        threshold=config.threshold,
        monitor=config.monitor,
        early_stopping=config.early_stopping,
        patience=config.patience,
        delta=config.delta,
        pos_weight=pos_weight,
        clip_samples=clip_samples,
        train_config_path=config_path,
        snr_bands=[(band.name, band.min_db, band.max_db) for band in config.snr_bands],
    )
    return trainer.train(train_loader, val_loader, test_loader, config.epochs)


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
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.check_data:
        check_dataset(args.config)
        return
    result = run_training(args.config, args.device)
    logger.info("Training complete. Best checkpoint: %s", result["checkpoint_path"])


if __name__ == "__main__":
    main()

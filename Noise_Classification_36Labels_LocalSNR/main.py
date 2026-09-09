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
from models import build_local_snr_model
from tasks import LocalSNRTrainer, split_parameter_groups
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
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.check_data:
        check_dataset(args.config)
        return
    result = run_training(args.config, args.device)
    logger.info("Training complete. Best checkpoint: %s", result["final_checkpoint"])


if __name__ == "__main__":
    main()

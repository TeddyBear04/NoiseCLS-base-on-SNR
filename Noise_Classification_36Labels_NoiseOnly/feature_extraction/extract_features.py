from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
from collections import OrderedDict
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import TrainConfig
from dataset import NoiseDataLoaderManager
from features import AudioFrontend
from models import AudioModel, build_backbone

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class ClassifierInputHook:
    def __init__(self, module: torch.nn.Module) -> None:
        self.features: torch.Tensor | None = None
        self.handle = module.register_forward_hook(self._capture)

    def _capture(self, module, inputs, output) -> None:
        self.features = inputs[0].detach()

    def close(self) -> None:
        self.handle.remove()


def load_feature_config(path: Path) -> Dict[str, object]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def resolve_project_path(value: str) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def aggregate_by_clip(
    sample_ids: List[str], features: np.ndarray, targets: np.ndarray
) -> tuple[List[str], np.ndarray, np.ndarray]:
    groups: "OrderedDict[str, List[int]]" = OrderedDict()
    for index, sample_id in enumerate(sample_ids):
        groups.setdefault(sample_id, []).append(index)
    clip_features = np.stack([features[indexes].mean(axis=0) for indexes in groups.values()])
    clip_targets = np.stack([targets[indexes[0]] for indexes in groups.values()])
    return list(groups), clip_features.astype(np.float32), clip_targets.astype(np.float32)


def extract_split(model, loader, device, hook) -> tuple[List[str], np.ndarray, np.ndarray]:
    ids: List[str] = []
    feature_batches = []
    target_batches = []
    with torch.no_grad():
        for batch in tqdm(loader, desc="Extracting features", unit="batch", dynamic_ncols=True):
            hook.features = None
            _ = model(batch["waveform"].to(device, non_blocking=True))
            if hook.features is None:
                raise RuntimeError("Classifier hook did not capture features")
            features = hook.features
            if features.ndim > 2:
                features = features.flatten(start_dim=1)
            ids.extend(list(batch["audio_name"]))
            feature_batches.append(features.cpu().numpy().astype(np.float32))
            target_batches.append(batch["target"].numpy().astype(np.float32))
    return aggregate_by_clip(ids, np.concatenate(feature_batches), np.concatenate(target_batches))


def run(feature_config_path: str) -> None:
    feature_config = load_feature_config(Path(feature_config_path))
    train_config_path = resolve_project_path(str(feature_config["train_config_path"]))
    config = TrainConfig.from_json(str(train_config_path))
    torch.manual_seed(config.random_seed)
    config.dataset_splitter.dynamic_snr_enabled = False
    config.batch_size = int(feature_config.get("batch_size", config.batch_size))
    requested_device = str(feature_config.get("device", "cuda"))
    device = torch.device(
        requested_device if not requested_device.startswith("cuda") or torch.cuda.is_available() else "cpu"
    )

    manager = NoiseDataLoaderManager(
        dataset_config=config.dataset_splitter,
        audio_config=config.audio_features,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        cache_audio=False,
        pin_memory=config.pin_memory,
        seed=config.random_seed,
        classes_num=config.model.classes_num,
    )
    model = AudioModel(AudioFrontend(config.audio_features), build_backbone(config.model)).to(device)
    checkpoint_path = resolve_project_path(str(feature_config["checkpoint_path"]))
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.eval()
    hook = ClassifierInputHook(model.backbone.fc_audioset)
    output_dir = resolve_project_path(str(feature_config.get("output_dir", "outputs/features")))
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        for split in ("train", "val", "test"):
            sample_ids, features, targets = extract_split(
                model, manager.get_dataloader(split, shuffle=False), device, hook
            )
            np.save(output_dir / f"{split}_features.npy", features)
            np.save(output_dir / f"{split}_targets.npy", targets)
            with (output_dir / f"{split}_metadata.csv").open(
                "w", newline="", encoding="utf-8"
            ) as handle:
                writer = csv.writer(handle)
                writer.writerow(["feature_index", "sample_id"])
                writer.writerows(enumerate(sample_ids))
            logger.info("%s: features=%s targets=%s", split, features.shape, targets.shape)
    finally:
        hook.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Export clip-level classifier input features")
    parser.add_argument(
        "--config", default=str(PROJECT_ROOT / "config" / "feature_config.json")
    )
    args = parser.parse_args()
    run(args.config)


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import csv
import json
import logging
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch

from config import TrainConfig
from dataset import fixed_window, load_audio_file, read_label_catalog, sliding_window_starts
from features import AudioFrontend
from models import AudioModel, build_backbone

PROJECT_ROOT = Path(__file__).resolve().parent
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def _check_checkpoint_classes(
    checkpoint_path: Path,
    checkpoint: dict,
    state_dict: dict,
    classes_num: int,
) -> None:
    """Turn a head-size mismatch into an actionable message.

    The 21-label checkpoints under `checkpoint/` cannot be loaded into a
    36-label model; without this the failure is a raw state_dict size error.
    """
    label_names = checkpoint.get("label_names")
    checkpoint_classes = len(label_names) if label_names else None
    if checkpoint_classes is None:
        weight = state_dict.get("backbone.fc_audioset.weight")
        checkpoint_classes = int(weight.shape[0]) if weight is not None else None
    if checkpoint_classes is not None and checkpoint_classes != classes_num:
        raise ValueError(
            f"{checkpoint_path} was trained for {checkpoint_classes} labels, but the "
            f"config asks for {classes_num}. Use a checkpoint from a matching run, or "
            f"set model.classes_num (and the dataset) to match this checkpoint."
        )


def load_model(
    config: TrainConfig,
    checkpoint_path: Path,
    device: torch.device,
) -> tuple[AudioModel, List[str], float]:
    model = AudioModel(AudioFrontend(config.audio_features), build_backbone(config.model)).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    _check_checkpoint_classes(checkpoint_path, checkpoint, state_dict, config.model.classes_num)
    model.load_state_dict(state_dict, strict=True)
    label_names = checkpoint.get("label_names")
    if label_names is None:
        label_names = [
            item.display_name
            for item in read_label_catalog(
                Path(config.dataset_splitter.dataset_path),
                config.dataset_splitter.selected_labels_file,
            )
        ]
    threshold = float(checkpoint.get("threshold", config.threshold))
    model.eval()
    return model, list(label_names), threshold


def predict_audio(
    audio_path: str,
    config_path: str,
    checkpoint_path: Optional[str] = None,
    output_csv: Optional[str] = None,
    device_name: Optional[str] = None,
) -> dict:
    config = TrainConfig.from_json(config_path)
    device = torch.device(
        device_name if device_name else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    if device.type == "cuda" and not torch.cuda.is_available():
        device = torch.device("cpu")
    default_checkpoint = Path(config.ckpt_dir) / config.model.backbone / "audio_best.pt"
    checkpoint = Path(checkpoint_path).expanduser().resolve() if checkpoint_path else default_checkpoint
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")

    model, label_names, threshold = load_model(config, checkpoint, device)
    waveform = load_audio_file(Path(audio_path).expanduser().resolve(), config.audio_features.sample_rate)
    clip_samples = int(round(config.audio_features.sample_rate * config.audio_features.clip_seconds))
    hop_samples = int(
        round(config.audio_features.sample_rate * config.audio_features.inference_hop_seconds)
    )
    starts = sliding_window_starts(waveform.numel(), clip_samples, hop_samples)
    probabilities = []
    batch_size = config.batch_size
    with torch.no_grad():
        for offset in range(0, len(starts), batch_size):
            batch_starts = starts[offset : offset + batch_size]
            batch = torch.stack(
                [fixed_window(waveform, start, clip_samples) for start in batch_starts]
            ).to(device)
            logits = model(batch)["clipwise_output"]
            probabilities.append(torch.sigmoid(logits).cpu().numpy())
    window_probability = np.concatenate(probabilities, axis=0)
    clip_probability = window_probability.mean(axis=0)

    rows = []
    duration_seconds = waveform.numel() / config.audio_features.sample_rate
    for start, probability in zip(starts, window_probability):
        start_seconds = start / config.audio_features.sample_rate
        row = {
            "start_seconds": round(start_seconds, 6),
            "end_seconds": round(min(duration_seconds, start_seconds + config.audio_features.clip_seconds), 6),
        }
        row.update({label: float(value) for label, value in zip(label_names, probability)})
        rows.append(row)

    csv_path = (
        Path(output_csv).expanduser().resolve()
        if output_csv
        else Path(audio_path).expanduser().resolve().with_suffix(".noise_predictions.csv")
    )
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    detected = [
        {"label": label, "probability": float(value)}
        for label, value in sorted(
            zip(label_names, clip_probability), key=lambda item: item[1], reverse=True
        )
        if value >= threshold
    ]
    result = {
        "audio_path": str(Path(audio_path).expanduser().resolve()),
        "checkpoint_path": str(checkpoint),
        "sample_rate": config.audio_features.sample_rate,
        "window_seconds": config.audio_features.clip_seconds,
        "hop_seconds": config.audio_features.inference_hop_seconds,
        "threshold": threshold,
        "num_windows": len(starts),
        "detected_labels": detected,
        "window_predictions_csv": str(csv_path),
    }
    summary_path = csv_path.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info("Detected labels: %s", detected)
    logger.info("Window predictions: %s", csv_path)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sliding-window inference for the noise classifier")
    parser.add_argument("audio_path")
    parser.add_argument("--config", default=str(PROJECT_ROOT / "config" / "train_config.json"))
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--output-csv", default=None)
    parser.add_argument("--device", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    predict_audio(
        audio_path=args.audio_path,
        config_path=args.config,
        checkpoint_path=args.checkpoint,
        output_csv=args.output_csv,
        device_name=args.device,
    )


if __name__ == "__main__":
    main()

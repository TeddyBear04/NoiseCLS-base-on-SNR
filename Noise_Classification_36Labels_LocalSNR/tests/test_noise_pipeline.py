import csv
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import AudioFeaturesConfig, LocalSNRConfig, SplitterConfig
from dataset import (
    NoiseDataLoaderManager,
    compute_local_snr_targets,
    local_snr_segment_starts,
    mix_with_dynamic_snr,
    sliding_window_starts,
)
from features import AudioFrontend
from models import Cnn14MobileV2LocalSNR, LocalSNRAudioModel
from utils import MultiTaskNoiseSNRLoss
from utils.evaluate import aggregate_windows, compute_local_snr_metrics, compute_multilabel_metrics

# Both fixtures describe the same three labels and the same two clips per split;
# only the on-disk encoding differs.
LABEL_NAMES = ["A", "B", "C"]
ORIGINAL_INDICES = [10, 20, 30]
CLIP_LABELS = ([10], [20, 30])


def _write_clip(split_path: Path, sample_id: str, duration: float, noise_directory: str) -> None:
    samples = int(16_000 * duration)
    time = np.arange(samples, dtype=np.float32) / 16_000
    clean = 0.1 * np.sin(2 * np.pi * 300 * time)
    noise = 0.03 * np.sin(2 * np.pi * 900 * time)
    for directory, waveform in (
        ("clean", clean),
        (noise_directory, noise),
        ("mixture", clean + noise),
    ):
        sf.write(split_path / directory / f"{sample_id}.wav", waveform, 16_000)


def build_multi_hot_dataset(root: Path) -> None:
    """labels.txt + a multi_hot_<N> manifest column, as in the 36-label dataset."""
    (root / "labels.txt").write_text("\n".join(LABEL_NAMES) + "\n", encoding="utf-8")
    multi_hot_width = len(LABEL_NAMES)

    for split in ("train", "validation", "test"):
        split_path = root / split
        for directory in ("mixture", "clean", "noise"):
            (split_path / directory).mkdir(parents=True, exist_ok=True)
        rows = []
        for index, labels in enumerate(CLIP_LABELS):
            sample_id = f"{split}_{index:02d}"
            duration = 1.5 + index
            _write_clip(split_path, sample_id, duration, "noise")
            multi_hot = [0] * multi_hot_width
            for original_index in labels:
                multi_hot[ORIGINAL_INDICES.index(original_index)] = 1
            rows.append(
                {
                    "sample_id": sample_id,
                    "multi_hot_3": json.dumps(multi_hot),
                    "target_snr_db": "5.0",
                    "duration_seconds": str(duration),
                }
            )
        with (split_path / "manifest.csv").open("w", newline="", encoding="utf-8-sig") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=["sample_id", "multi_hot_3", "target_snr_db", "duration_seconds"],
            )
            writer.writeheader()
            writer.writerows(rows)


def build_label_indices_dataset(root: Path) -> None:
    """selected_labels.csv + a label_indices manifest column, as in 21_labels_dataset."""
    with (root / "selected_labels.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["model_index", "original_index", "mid", "display_name"])
        writer.writerows(
            [
                [model_index, original_index, f"/m/{name.lower()}", name]
                for model_index, (original_index, name) in enumerate(
                    zip(ORIGINAL_INDICES, LABEL_NAMES)
                )
            ]
        )

    for split in ("train_single", "validation_single", "test_single"):
        split_path = root / split
        for directory in ("mixture", "clean", "oracle_noise"):
            (split_path / directory).mkdir(parents=True, exist_ok=True)
        rows = []
        for index, labels in enumerate(CLIP_LABELS):
            sample_id = f"{split}_{index:02d}"
            duration = 1.5 + index
            _write_clip(split_path, sample_id, duration, "oracle_noise")
            rows.append(
                {
                    "sample_id": sample_id,
                    "label_indices": json.dumps(labels),
                    "target_snr_db": "5.0",
                    "duration_seconds": str(duration),
                }
            )
        with (split_path / "manifest.csv").open("w", newline="", encoding="utf-8-sig") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=["sample_id", "label_indices", "target_snr_db", "duration_seconds"],
            )
            writer.writeheader()
            writer.writerows(rows)


AUDIO_CONFIG = AudioFeaturesConfig(
    sample_rate=16_000,
    clip_seconds=1.0,
    inference_hop_seconds=0.5,
    window_size=512,
    hop_size=160,
    mel_bins=64,
    fmax=8_000,
)


class ManifestLoaderTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def _assert_loads(self, dataset_config: SplitterConfig) -> None:
        manager = NoiseDataLoaderManager(
            dataset_config,
            AUDIO_CONFIG,
            batch_size=2,
            num_workers=0,
            cache_audio=False,
            pin_memory=False,
            classes_num=3,
        )
        self.assertEqual(manager.label_names, LABEL_NAMES)
        train_item = manager.datasets["train"][0]
        self.assertEqual(tuple(train_item["waveform"].shape), (16_000,))
        self.assertEqual(tuple(train_item["target"].shape), (3,))
        self.assertEqual(tuple(train_item["local_snr_db"].shape), (3,))
        self.assertEqual(tuple(train_item["local_snr_mask"].shape), (3,))
        self.assertTrue(bool(torch.isfinite(train_item["local_snr_db"]).all()))
        self.assertTrue(bool(train_item["dynamic_snr"]))
        self.assertTrue(bool(torch.isfinite(train_item["waveform"]).all()))
        # The first clip carries label A, the second carries B and C.
        targets = [record.target for record in manager.datasets["train"].records]
        self.assertEqual(targets, [(1.0, 0.0, 0.0), (0.0, 1.0, 1.0)])
        # Val/test expand each clip into several sliding windows.
        self.assertGreater(len(manager.datasets["val"]), len(manager.datasets["val"].records))

    def test_multi_hot_layout(self) -> None:
        build_multi_hot_dataset(self.root)
        self._assert_loads(
            SplitterConfig(
                dataset_path=str(self.root),
                dynamic_snr_enabled=True,
                dynamic_snr_probability=1.0,
            )
        )

    def test_label_indices_layout(self) -> None:
        build_label_indices_dataset(self.root)
        self._assert_loads(
            SplitterConfig(
                dataset_path=str(self.root),
                train_directory="train_single",
                validation_directory="validation_single",
                test_directory="test_single",
                noise_directory="oracle_noise",
                selected_labels_file="selected_labels.csv",
                dynamic_snr_enabled=True,
                dynamic_snr_probability=1.0,
            )
        )

    def test_missing_label_column_is_rejected(self) -> None:
        build_multi_hot_dataset(self.root)
        manifest_path = self.root / "train" / "manifest.csv"
        with manifest_path.open("r", encoding="utf-8-sig", newline="") as handle:
            rows = list(csv.DictReader(handle))
        with manifest_path.open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(
                handle, fieldnames=["sample_id", "target_snr_db", "duration_seconds"]
            )
            writer.writeheader()
            for row in rows:
                row.pop("multi_hot_3")
                writer.writerow(row)
        with self.assertRaisesRegex(ValueError, "no label column"):
            NoiseDataLoaderManager(
                SplitterConfig(dataset_path=str(self.root)),
                AUDIO_CONFIG,
                batch_size=2,
                num_workers=0,
                pin_memory=False,
                classes_num=3,
            )


class WindowingAndMetricsTest(unittest.TestCase):
    def test_windowing_and_multilabel_metrics(self) -> None:
        self.assertEqual(sliding_window_starts(25, 10, 6), [0, 6, 12, 15])
        target = np.asarray([[1, 0, 0], [0, 1, 1]], dtype=np.float32)
        probability = np.asarray([[0.9, 0.1, 0.2], [0.1, 0.8, 0.9]], dtype=np.float32)
        metrics = compute_multilabel_metrics(target, probability, 0.5, LABEL_NAMES)
        self.assertAlmostEqual(metrics["mAP"], 1.0)
        self.assertAlmostEqual(metrics["f1_macro"], 1.0)
        ids, clip_probability, _, _ = aggregate_windows(
            ["one", "one", "two"],
            np.ones((3, 3), dtype=np.float32),
            np.zeros((3, 3), dtype=np.float32),
            np.asarray([0.0, 0.0, 5.0], dtype=np.float32),
        )
        self.assertEqual(ids, ["one", "two"])
        self.assertEqual(clip_probability.shape, (2, 3))

    def test_local_snr_targets_and_dynamic_mix_are_aligned(self) -> None:
        sample_rate = 16_000
        time = torch.arange(sample_rate, dtype=torch.float32) / sample_rate
        clean = 0.2 * torch.sin(2 * torch.pi * 300 * time)
        noise = 0.05 * torch.sin(2 * torch.pi * 900 * time)
        mixture, clean_component, noise_component = mix_with_dynamic_snr(
            clean, noise, sample_rate, -5.0, 20.0, 0.5
        )
        self.assertTrue(torch.allclose(mixture, clean_component + noise_component, atol=1e-6))
        config = LocalSNRConfig()
        target, mask, centers = compute_local_snr_targets(
            clean_component, noise_component, sample_rate, config
        )
        self.assertEqual(target.shape, mask.shape)
        self.assertEqual(target.shape, centers.shape)
        self.assertEqual(target.numel(), 3)
        self.assertTrue(bool(mask.all()))

    def test_multitask_model_and_loss_shapes(self) -> None:
        local_config = LocalSNRConfig()
        clip_samples = int(AUDIO_CONFIG.sample_rate * AUDIO_CONFIG.clip_seconds)
        segment_count = len(
            local_snr_segment_starts(clip_samples, AUDIO_CONFIG.sample_rate, local_config)
        )
        model = LocalSNRAudioModel(
            AudioFrontend(AUDIO_CONFIG),
            Cnn14MobileV2LocalSNR(classes_num=3),
            local_config,
            segment_count,
        )
        waveform = torch.randn(2, clip_samples)
        output = model(waveform)
        self.assertEqual(tuple(output["clipwise_output"].shape), (2, 3))
        self.assertEqual(tuple(output["local_snr_db"].shape), (2, segment_count))
        loss = MultiTaskNoiseSNRLoss(
            snr_weight=local_config.loss_weight,
            target_offset_db=local_config.target_offset_db,
            target_scale_db=local_config.target_scale_db,
        )(
            output,
            {
                "target": torch.zeros(2, 3),
                "local_snr_db": torch.zeros(2, segment_count),
                "local_snr_mask": torch.ones(2, segment_count, dtype=torch.bool),
            },
        )
        self.assertTrue(bool(torch.isfinite(loss)))
        metrics = compute_local_snr_metrics(
            np.zeros((2, segment_count), dtype=np.float32),
            np.ones((2, segment_count), dtype=np.float32),
            np.ones((2, segment_count), dtype=bool),
        )
        self.assertAlmostEqual(metrics["local_snr_mae_db"], 1.0)


if __name__ == "__main__":
    unittest.main()

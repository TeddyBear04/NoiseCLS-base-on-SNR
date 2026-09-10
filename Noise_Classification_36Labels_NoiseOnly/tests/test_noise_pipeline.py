import csv
import json
import shutil
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

from config import AudioFeaturesConfig, ModelConfig, SplitterConfig, TrainAugmentationConfig, TrainConfig
from dataset import (
    NoiseDataLoaderManager,
    mix_with_snr,
    random_pad_crop,
    random_time_shift,
    same_class_mixup,
    sliding_window_starts,
)
from features import AudioFrontend
from models import MODEL_REGISTRY, AudioModel, build_backbone
from tasks.audio_train import AudioTrainer, l1_penalty, split_decay_parameters
from utils import SingleLabelCELoss
from utils.evaluate import aggregate_windows, compute_metrics

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
    # With labels.txt the line order is the label index, so the multi-hot row is
    # as wide as the catalog and its flags sit at those positions - not at the
    # AudioSet indices, which only the selected_labels.csv layout uses.
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
        for directory in ("mixture", "clean", "noise"):
            (split_path / directory).mkdir(parents=True, exist_ok=True)
        rows = []
        for index, labels in enumerate(CLIP_LABELS):
            sample_id = f"{split}_{index:02d}"
            duration = 1.5 + index
            _write_clip(split_path, sample_id, duration, "noise")
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
        self.assertFalse(bool(train_item["dynamic_snr"]))
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
                selected_labels_file="selected_labels.csv",
            )
        )

    def test_noise_only_config_rejects_mixture(self) -> None:
        with self.assertRaises(ValueError):
            SplitterConfig(dataset_path=str(self.root), signal_type="mixture")
        with self.assertRaises(ValueError):
            SplitterConfig(dataset_path=str(self.root), dynamic_snr_enabled=True)

    def test_noise_only_loads_when_clean_and_mixture_are_absent(self) -> None:
        build_multi_hot_dataset(self.root)
        for split in ("train", "validation", "test"):
            shutil.rmtree(self.root / split / "clean")
            shutil.rmtree(self.root / split / "mixture")

        manager = NoiseDataLoaderManager(
            SplitterConfig(dataset_path=str(self.root)),
            AUDIO_CONFIG,
            batch_size=2,
            num_workers=0,
            pin_memory=False,
            classes_num=3,
        )
        item = manager.datasets["train"][0]
        self.assertEqual(tuple(item["waveform"].shape), (16_000,))
        self.assertFalse(bool(item["dynamic_snr"]))

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
    def test_waveform_augmentation_helpers_preserve_shape_and_finiteness(self) -> None:
        torch.manual_seed(7)
        np.random.seed(7)
        waveform = torch.linspace(-0.25, 0.25, 16_000)
        shifted = random_time_shift(waveform, max_shift_samples=2_000)
        cropped = random_pad_crop(waveform, length=16_000, padding_samples=2_000)
        mixed, coefficient = same_class_mixup(waveform, waveform.flip(0), alpha=0.4)
        mixture = mix_with_snr(waveform, mixed, snr_db=5.0)

        for result in (shifted, cropped, mixed, mixture):
            self.assertEqual(tuple(result.shape), (16_000,))
            self.assertTrue(bool(torch.isfinite(result).all()))
        self.assertGreaterEqual(coefficient, 0.0)
        self.assertLessEqual(coefficient, 1.0)
        self.assertFalse(torch.equal(shifted, waveform))
        self.assertFalse(torch.equal(cropped, waveform))

    def test_online_augmentation_is_train_only_and_keeps_same_class_label(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            build_multi_hot_dataset(root)

            # Give both train records label A so same-class Mixup has a valid,
            # distinct partner while validation/test remain untouched.
            manifest_path = root / "train" / "manifest.csv"
            with manifest_path.open("r", encoding="utf-8-sig", newline="") as handle:
                rows = list(csv.DictReader(handle))
                fieldnames = list(rows[0])
            rows[1]["multi_hot_3"] = json.dumps([1, 0, 0])
            with manifest_path.open("w", encoding="utf-8-sig", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(rows)

            augmentation = TrainAugmentationConfig(
                enabled=True,
                random_clean_probability=1.0,
                clean_speed_probability=0.0,
                clean_gain_probability=0.0,
                clean_reverb_probability=0.0,
                noise_time_shift_probability=0.0,
                noise_gain_probability=0.0,
                noise_eq_probability=0.0,
                noise_reverb_probability=0.0,
                noise_time_stretch_probability=0.0,
                noise_polarity_probability=0.0,
                same_class_mixup_probability=1.0,
            )
            manager = NoiseDataLoaderManager(
                SplitterConfig(
                    dataset_path=str(root),
                ),
                AUDIO_CONFIG,
                batch_size=2,
                num_workers=0,
                pin_memory=False,
                classes_num=3,
                augmentation_config=augmentation,
            )

            train_item = manager.datasets["train"][0]
            validation_item = manager.datasets["val"][0]
            self.assertTrue(bool(train_item["waveform_augmented"]))
            self.assertFalse(bool(train_item["clean_replaced"]))
            self.assertTrue(bool(train_item["same_class_mixup"]))
            self.assertEqual(train_item["target"].tolist(), [1.0, 0.0, 0.0])
            self.assertFalse(bool(validation_item["waveform_augmented"]))
            self.assertFalse(bool(validation_item["clean_replaced"]))
            self.assertFalse(bool(validation_item["same_class_mixup"]))

    def test_windowing_and_single_label_metrics(self) -> None:
        self.assertEqual(sliding_window_starts(25, 10, 6), [0, 6, 12, 15])
        # One label per clip, and the argmax picks the right one every time.
        target = np.asarray([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float32)
        # Real softmax rows: they sum to one and no winner clears 0.5.
        probability = np.asarray(
            [[0.45, 0.35, 0.20], [0.30, 0.42, 0.28], [0.33, 0.25, 0.42]], dtype=np.float32
        )
        metrics = compute_metrics(target, probability, LABEL_NAMES)
        self.assertAlmostEqual(metrics["mAP"], 1.0)
        self.assertAlmostEqual(metrics["f1_macro"], 1.0)
        self.assertAlmostEqual(metrics["top1_accuracy"], 1.0)
        # Softmax outputs rarely clear 0.5, so the prediction must come from the
        # argmax: a threshold rule would have scored these three clips as zero.
        self.assertAlmostEqual(metrics["subset_accuracy"], 1.0)

    def test_argmax_metrics_on_a_wrong_prediction(self) -> None:
        target = np.asarray([[1, 0, 0], [0, 1, 0]], dtype=np.float32)
        # Clip 2 is wrong: B is the truth but C wins, with B ranked second.
        probability = np.asarray([[0.48, 0.30, 0.22], [0.10, 0.42, 0.48]], dtype=np.float32)
        metrics = compute_metrics(target, probability, LABEL_NAMES)
        self.assertAlmostEqual(metrics["top1_accuracy"], 0.5)
        self.assertAlmostEqual(metrics["top3_accuracy"], 1.0)
        # Under a single label per clip these three collapse onto top-1.
        self.assertAlmostEqual(metrics["subset_accuracy"], metrics["top1_accuracy"])
        self.assertAlmostEqual(metrics["f1_micro"], metrics["top1_accuracy"])
        self.assertAlmostEqual(metrics["balanced_accuracy"], metrics["top1_accuracy"])
        ids, clip_probability, _, _ = aggregate_windows(
            ["one", "one", "two"],
            np.ones((3, 3), dtype=np.float32),
            np.zeros((3, 3), dtype=np.float32),
            np.asarray([0.0, 0.0, 5.0], dtype=np.float32),
        )
        self.assertEqual(ids, ["one", "two"])
        self.assertEqual(clip_probability.shape, (2, 3))


class _TinyClassifier(torch.nn.Module):
    """Conv + BatchNorm + Linear, i.e. one of every parameter shape that matters."""

    def __init__(self, classes_num: int = 3, input_size: int = 4) -> None:
        super().__init__()
        self.conv = torch.nn.Conv1d(1, 2, kernel_size=1)
        self.norm = torch.nn.BatchNorm1d(2)
        self.linear = torch.nn.Linear(2 * input_size, classes_num)

    def forward(self, waveform: torch.Tensor) -> dict:
        hidden = self.norm(self.conv(waveform.unsqueeze(1)))
        return {"clipwise_output": self.linear(hidden.flatten(1))}


class RegularizationTest(unittest.TestCase):
    def test_regularization_defaults_to_the_legacy_weight_decay(self) -> None:
        # Configs written before the regularization block still have to train
        # with the L2 strength they asked for.
        config = TrainConfig(weight_decay=0.02)
        self.assertAlmostEqual(config.regularization.l2_lambda, 0.02)
        self.assertAlmostEqual(config.regularization.l1_lambda, 0.0)
        self.assertTrue(config.regularization.exclude_bias_and_norm)

    def test_regularization_block_overrides_the_legacy_weight_decay(self) -> None:
        config = TrainConfig(
            weight_decay=0.02,
            regularization={"l1_lambda": 1e-5, "l2_lambda": 0.01},
        )
        self.assertAlmostEqual(config.regularization.l2_lambda, 0.01)
        self.assertAlmostEqual(config.regularization.l1_lambda, 1e-5)
        # weight_decay is kept in step so nothing reads a stale L2 strength.
        self.assertAlmostEqual(config.weight_decay, 0.01)

    def test_decayed_group_excludes_bias_and_normalisation(self) -> None:
        model = _TinyClassifier()
        decayed, skipped = split_decay_parameters(model, exclude_bias_and_norm=True)
        self.assertEqual([tuple(p.shape) for p in decayed], [(2, 1, 1), (3, 8)])
        # conv bias, BatchNorm weight, BatchNorm bias, linear bias.
        self.assertEqual(len(skipped), 4)
        self.assertTrue(all(p.ndim == 1 for p in skipped))
        total = sum(1 for _ in model.parameters())
        self.assertEqual(len(decayed) + len(skipped), total)

    def test_decayed_group_can_cover_every_parameter(self) -> None:
        model = _TinyClassifier()
        decayed, skipped = split_decay_parameters(model, exclude_bias_and_norm=False)
        self.assertEqual(skipped, [])
        self.assertEqual(len(decayed), sum(1 for _ in model.parameters()))

    def test_l1_penalty_sums_absolute_weights(self) -> None:
        parameters = [
            torch.nn.Parameter(torch.tensor([[1.0, -2.0]])),
            torch.nn.Parameter(torch.tensor([[-3.0]])),
        ]
        self.assertAlmostEqual(float(l1_penalty(parameters, 0.5).detach()), 3.0)
        # A disabled penalty must not build a graph node at all.
        zero = l1_penalty(parameters, 0.0)
        self.assertAlmostEqual(float(zero), 0.0)
        self.assertFalse(zero.requires_grad)

    def test_l1_penalty_reaches_gradients_but_not_the_logged_loss(self) -> None:
        torch.manual_seed(0)
        model = _TinyClassifier()
        waveform = torch.randn(4, 4)
        target = torch.eye(3)[torch.tensor([0, 1, 2, 0])]
        batches = [{"waveform": waveform, "target": target}]

        # BatchNorm normalises with batch statistics in train mode and running
        # statistics in eval mode, so the reference has to use the same mode the
        # training loop does or the two losses are not comparable.
        model.train()
        with torch.no_grad():
            expected_ce = float(
                SingleLabelCELoss()(model(waveform), {"target": target})
            )

        with tempfile.TemporaryDirectory() as directory:
            trainer = AudioTrainer(
                model=model,
                optimizer=torch.optim.AdamW(model.parameters(), lr=0.0),
                device=torch.device("cpu"),
                ckpt_dir=directory,
                label_names=LABEL_NAMES,
                early_stopping=False,
                l1_lambda=1.0,
            )
            logged_loss, _ = trainer._train_epoch(batches, epoch=1)

        # A learning rate of zero keeps the weights fixed, so the logged number
        # is comparable with the reference cross-entropy above.
        self.assertAlmostEqual(logged_loss, expected_ce, places=5)
        # The penalty still has to steer training: its gradient is the sign of
        # each decayed weight, which cross-entropy alone would never produce.
        gradients = [p.grad for p in trainer.l1_parameters]
        self.assertTrue(all(gradient is not None for gradient in gradients))
        self.assertTrue(
            any(torch.all(gradient.abs() >= 1.0 - 1e-6) for gradient in gradients)
        )

    def test_trainer_without_l1_leaves_the_parameter_list_empty(self) -> None:
        model = _TinyClassifier()
        with tempfile.TemporaryDirectory() as directory:
            trainer = AudioTrainer(
                model=model,
                optimizer=torch.optim.AdamW(model.parameters(), lr=0.0),
                device=torch.device("cpu"),
                ckpt_dir=directory,
                label_names=LABEL_NAMES,
                early_stopping=False,
            )
        self.assertEqual(trainer.l1_parameters, [])


class TestBackboneRegistry(unittest.TestCase):
    """Every registered backbone has to be a drop-in swap for the others.

    Training only ever reaches a backbone through ``model.backbone`` in the
    config, so the contract that matters is the one BaseBackbone states: take
    the frontend's [Batch, 1, Time, Mel] features, return [Batch, Classes]
    logits. A backbone that breaks it is unusable no matter how it trains.
    """

    # Shorter than the 4-second clip (401 frames) to keep the sweep quick; the
    # frequency axis stays at the configured 128 mel bins because the stem
    # cares about it.
    FEATURES = (2, 1, 256, 128)
    CLASSES_NUM = 36

    # The mid-size CNNs added for backbone comparison, alongside the PANNs and
    # mobile families that were here first.
    MID_SIZE_BACKBONES = ("ResNet18", "ResNet34", "DenseNet121", "EfficientNetB2")

    def test_mid_size_backbones_are_registered(self) -> None:
        for name in self.MID_SIZE_BACKBONES:
            with self.subTest(backbone=name):
                self.assertIn(name, MODEL_REGISTRY)

    def test_every_backbone_maps_frontend_features_to_class_logits(self) -> None:
        features = torch.randn(*self.FEATURES)
        for name in sorted(MODEL_REGISTRY):
            with self.subTest(backbone=name):
                backbone = build_backbone(
                    ModelConfig(backbone=name, pretrained=False, classes_num=self.CLASSES_NUM)
                )
                backbone.eval()
                with torch.no_grad():
                    logits = backbone(features)
                self.assertEqual(logits.shape, (self.FEATURES[0], self.CLASSES_NUM))
                self.assertTrue(torch.isfinite(logits).all())

    def test_every_backbone_reports_a_distinct_name(self) -> None:
        names = [
            build_backbone(ModelConfig(backbone=name, pretrained=False, classes_num=4)).get_name()
            for name in sorted(MODEL_REGISTRY)
        ]
        self.assertEqual(len(names), len(set(names)))
        self.assertNotIn("base_backbone", names)

    def test_the_audio_model_wrapper_accepts_any_registered_backbone(self) -> None:
        # The wrapper asserts on the BaseBackbone type and reshapes nothing, so
        # a backbone that passes here is ready for tasks/audio_train.py.
        features_config = AudioFeaturesConfig()
        for name in ("ResNet18", "DenseNet121"):
            with self.subTest(backbone=name):
                model = AudioModel(
                    frontend=AudioFrontend(features_config),
                    backbone=build_backbone(
                        ModelConfig(backbone=name, pretrained=False, classes_num=self.CLASSES_NUM)
                    ),
                )
                model.eval()
                waveform = torch.randn(2, int(16_000 * features_config.clip_seconds))
                with torch.no_grad():
                    output = model(waveform)
                self.assertEqual(
                    output["clipwise_output"].shape, (2, self.CLASSES_NUM)
                )


if __name__ == "__main__":
    unittest.main()

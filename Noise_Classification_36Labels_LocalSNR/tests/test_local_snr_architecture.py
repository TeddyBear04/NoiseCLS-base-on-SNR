import sys
import tempfile
import unittest
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import (
    AudioFeaturesConfig,
    ModelConfig,
    MultiTaskLossConfig,
    TrainConfig,
    TrainingStagesConfig,
)
from dataset import (
    local_snr_from_stems,
    mix_stems_with_dynamic_snr,
    mix_stems_with_snr,
)
from models import build_local_snr_model
from tasks import LocalSNRTrainer
from utils import BlackFeatherMultiTaskLoss


class ConsistentMixingTest(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(3)
        self.clean = 0.2 * torch.randn(8_000)
        self.noise = 0.2 * torch.randn(8_000)

    def test_fixed_snr_returns_aligned_components(self) -> None:
        mixture, clean, noise = mix_stems_with_snr(self.clean, self.noise, 10.0)
        self.assertTrue(torch.allclose(mixture, clean + noise, atol=1e-7))
        measured = 10.0 * torch.log10(clean.square().mean() / noise.square().mean())
        self.assertAlmostEqual(float(measured), 10.0, places=4)

    def test_dynamic_snr_returns_aligned_components(self) -> None:
        mixture, clean, noise, envelope = mix_stems_with_dynamic_snr(
            self.clean, self.noise, 16_000, -5.0, 20.0, 0.25
        )
        self.assertTrue(torch.allclose(mixture, clean + noise, atol=1e-7))
        self.assertEqual(envelope.shape, mixture.shape)
        local_snr, mask = local_snr_from_stems(clean, noise, frame_samples=2_000)
        self.assertEqual(tuple(local_snr.shape), (4,))
        self.assertEqual(tuple(mask.shape), (4,))
        self.assertTrue(bool(torch.isfinite(local_snr).all()))


class LocalSNRModelTest(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(5)
        self.audio_config = AudioFeaturesConfig(
            sample_rate=8_000,
            clip_seconds=0.256,
            inference_hop_seconds=0.128,
            window_size=128,
            hop_size=32,
            mel_bins=32,
            fmin=20,
            fmax=4_000,
            time_drop_width=0,
            time_stripes_num=0,
            freq_drop_width=0,
            freq_stripes_num=0,
            local_snr_segment_seconds=0.128,
        )
        self.model_config = ModelConfig(
            classes_num=4,
            embedding_dim=32,
            encoder_channels=8,
            extractor_channels=8,
            extractor_depth=2,
            mixture_branch_dropout=0.0,
        )
        self.model = build_local_snr_model(self.audio_config, self.model_config)

    def test_forward_shapes_and_mixture_consistency(self) -> None:
        waveform = torch.randn(2, 2_048)
        output = self.model(waveform)
        self.assertEqual(tuple(output["clipwise_output"].shape), (2, 4))
        self.assertEqual(tuple(output["estimated_noise"].shape), (2, 2_048))
        self.assertEqual(tuple(output["local_snr_output"].shape), (2, 2))
        self.assertTrue(
            torch.allclose(
                waveform,
                output["estimated_speech"] + output["estimated_noise"],
                atol=1e-6,
            )
        )
        for value in output.values():
            self.assertTrue(bool(torch.isfinite(value).all()))

    def test_all_fusion_ablation_modes(self) -> None:
        waveform = torch.randn(1, 2_048)
        for mode in ("fusion", "mixture", "noise"):
            output = self.model(waveform, fusion_mode=mode)
            self.assertEqual(tuple(output["clipwise_output"].shape), (1, 4))

    def test_multitask_loss_backpropagates(self) -> None:
        waveform = torch.randn(2, 2_048)
        target_noise = 0.2 * torch.randn_like(waveform)
        target_clean = waveform - target_noise
        snr_rows = []
        mask_rows = []
        for clean, noise in zip(target_clean, target_noise):
            snr, mask = local_snr_from_stems(clean, noise, frame_samples=1_024)
            snr_rows.append(snr)
            mask_rows.append(mask)
        output = self.model(waveform)
        objective = BlackFeatherMultiTaskLoss(MultiTaskLossConfig())
        losses = objective(
            output,
            {
                "target": torch.eye(4)[:2],
                "noise_waveform": target_noise,
                "local_snr_db": torch.stack(snr_rows),
                "local_snr_mask": torch.stack(mask_rows),
            },
        )
        self.assertTrue(bool(torch.isfinite(losses["loss"])))
        losses["loss"].backward()
        self.assertIsNotNone(next(self.model.noise_extractor.parameters()).grad)
        self.assertIsNotNone(next(self.model.classifier.parameters()).grad)

    def test_stage_freezing(self) -> None:
        self.model.set_stage("extractor")
        self.assertTrue(all(p.requires_grad for p in self.model.noise_extractor.parameters()))
        self.assertTrue(all(not p.requires_grad for p in self.model.classifier.parameters()))
        self.model.set_stage("heads")
        self.assertTrue(all(not p.requires_grad for p in self.model.noise_extractor.parameters()))
        self.assertTrue(all(p.requires_grad for p in self.model.classifier.parameters()))
        self.model.set_stage("joint")
        self.assertTrue(all(p.requires_grad for p in self.model.parameters()))

    def test_three_stage_trainer_smoke(self) -> None:
        waveforms = torch.randn(2, 2_048)
        noises = 0.2 * torch.randn_like(waveforms)
        items = []
        for index, (waveform, noise) in enumerate(zip(waveforms, noises)):
            clean = waveform - noise
            local_snr, local_mask = local_snr_from_stems(clean, noise, frame_samples=1_024)
            target = torch.zeros(4)
            target[index] = 1.0
            items.append(
                {
                    "audio_name": f"sample-{index}",
                    "waveform": waveform,
                    "noise_waveform": noise,
                    "local_snr_db": local_snr,
                    "local_snr_mask": local_mask,
                    "target": target,
                    "target_snr_db": torch.tensor(float(index * 5)),
                }
            )
        loader = torch.utils.data.DataLoader(items, batch_size=2)
        config = TrainConfig(
            batch_size=2,
            model=self.model_config,
            audio_features=self.audio_config,
            multitask_loss=MultiTaskLossConfig(),
            stages=TrainingStagesConfig(
                extractor_epochs=1,
                heads_epochs=1,
                joint_epochs=1,
                oracle_noise_probability=1.0,
            ),
            snr_bands=[
                {"name": "all", "min_db": -30.0, "max_db": 30.0}
            ],
        )
        model = build_local_snr_model(self.audio_config, self.model_config)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        with tempfile.TemporaryDirectory() as directory:
            trainer = LocalSNRTrainer(
                model,
                optimizer,
                torch.device("cpu"),
                config,
                directory,
                ["A", "B", "C", "D"],
                "missing-config.json",
            )
            result = trainer.train(loader, loader, loader)
            self.assertTrue(Path(result["final_checkpoint"]).is_file())
            self.assertTrue((Path(directory) / "summary.json").is_file())


if __name__ == "__main__":
    unittest.main()

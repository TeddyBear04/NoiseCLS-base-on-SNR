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
    RegularizationConfig,
    TrainConfig,
    TrainingStagesConfig,
)
from dataset import (
    local_snr_from_stems,
    mix_stems_with_dynamic_snr,
    mix_stems_with_snr,
)
from models import build_local_snr_model, build_noise_extractor
from tasks import LocalSNRTrainer, split_parameter_groups
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
        # A miniature Demucs keeps these tests quick; the default configuration
        # builds a six-million-parameter extractor, which is far more capacity
        # than a forward/backward smoke test needs.
        self.model_config = ModelConfig(
            classes_num=4,
            embedding_dim=32,
            encoder_channels=8,
            extractor_channels=8,
            extractor_depth=2,
            mixture_branch_dropout=0.0,
            demucs_hidden=8,
            demucs_depth=3,
            demucs_lstm_hidden=16,
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


class EarlyStoppingTest(unittest.TestCase):
    """A stage must stop once its own score stalls, without ending the run."""

    def _trainer_config(self, **overrides) -> TrainConfig:
        return TrainConfig(
            batch_size=2,
            model=ModelConfig(
                classes_num=4,
                embedding_dim=16,
                encoder_channels=8,
                demucs_hidden=8,
                demucs_depth=3,
                demucs_lstm_hidden=16,
                mixture_branch_dropout=0.0,
            ),
            audio_features=AudioFeaturesConfig(
                sample_rate=8_000,
                clip_seconds=0.256,
                inference_hop_seconds=0.128,
                window_size=128,
                hop_size=32,
                mel_bins=32,
                fmax=4_000,
                local_snr_segment_seconds=0.128,
            ),
            stages=TrainingStagesConfig(
                extractor_epochs=5, heads_epochs=0, joint_epochs=0
            ),
            snr_bands=[{"name": "all", "min_db": -30.0, "max_db": 30.0}],
            **overrides,
        )

    def _run(self, config: TrainConfig) -> int:
        torch.manual_seed(17)
        waveform = torch.randn(2, 2_048)
        noise = 0.2 * torch.randn_like(waveform)
        items = []
        for index, (mixture, component) in enumerate(zip(waveform, noise)):
            local_snr, mask = local_snr_from_stems(
                mixture - component, component, frame_samples=1_024
            )
            target = torch.zeros(4)
            target[index] = 1.0
            items.append(
                {
                    "audio_name": f"sample-{index}",
                    "waveform": mixture,
                    "noise_waveform": component,
                    "local_snr_db": local_snr,
                    "local_snr_mask": mask,
                    "target": target,
                    "target_snr_db": torch.tensor(0.0),
                }
            )
        loader = torch.utils.data.DataLoader(items, batch_size=2)
        model = build_local_snr_model(config.audio_features, config.model)
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
            trainer.train(loader, loader, loader)
            history = (Path(directory) / "history.jsonl").read_text(encoding="utf-8")
        return len([line for line in history.splitlines() if line.strip()])

    def test_a_stalled_stage_stops_before_its_epoch_budget(self) -> None:
        # A delta this large means no epoch ever counts as an improvement.
        epochs = self._run(
            self._trainer_config(early_stopping=True, patience=1, delta=100.0)
        )
        self.assertLess(epochs, 5)

    def test_disabling_early_stopping_runs_every_epoch(self) -> None:
        epochs = self._run(
            self._trainer_config(early_stopping=False, patience=1, delta=100.0)
        )
        self.assertEqual(epochs, 5)


class NoiseExtractorLengthTest(unittest.TestCase):
    """Both extractors must return exactly as many samples as they were given.

    Demucs pads the input up to its ``valid_length`` before the strided
    encoder and trims the result afterwards. A trim that is off by one breaks
    ``estimated_speech = mixture - estimated_noise`` silently through
    broadcasting, so the awkward lengths around the 4-second clip matter more
    than the nominal one.
    """

    LENGTHS = (63_999, 64_000, 64_001, 65_536)

    def setUp(self) -> None:
        torch.manual_seed(11)
        self.audio_config = AudioFeaturesConfig(window_size=512, hop_size=160)

    def _model_config(self, extractor_type: str) -> ModelConfig:
        return ModelConfig(
            classes_num=4,
            extractor_type=extractor_type,
            extractor_channels=8,
            extractor_depth=2,
            demucs_hidden=8,
            demucs_depth=3,
            demucs_lstm_hidden=16,
        )

    def test_every_extractor_preserves_length(self) -> None:
        for extractor_type in ("mask", "demucs"):
            extractor = build_noise_extractor(
                self.audio_config, self._model_config(extractor_type)
            )
            for length in self.LENGTHS:
                with self.subTest(extractor=extractor_type, length=length):
                    mixture = 0.1 * torch.randn(2, length)
                    with torch.no_grad():
                        estimated_noise = extractor(mixture)
                    self.assertEqual(estimated_noise.shape, mixture.shape)
                    self.assertTrue(bool(torch.isfinite(estimated_noise).all()))

    def test_demucs_keeps_the_mixture_invariant_end_to_end(self) -> None:
        model = build_local_snr_model(self.audio_config, self._model_config("demucs"))
        for length in self.LENGTHS:
            with self.subTest(length=length):
                mixture = 0.1 * torch.randn(2, length)
                with torch.no_grad():
                    output = model(mixture)
                self.assertEqual(output["estimated_noise"].shape, mixture.shape)
                self.assertEqual(output["estimated_speech"].shape, mixture.shape)
                self.assertTrue(
                    torch.allclose(
                        mixture,
                        output["estimated_speech"] + output["estimated_noise"],
                        atol=1e-6,
                    )
                )

    def test_silent_input_does_not_produce_nan(self) -> None:
        """Demucs divides by the input standard deviation; silence must not blow up."""
        extractor = build_noise_extractor(self.audio_config, self._model_config("demucs"))
        with torch.no_grad():
            estimated_noise = extractor(torch.zeros(2, 64_000))
        self.assertTrue(bool(torch.isfinite(estimated_noise).all()))


class ParameterGroupTest(unittest.TestCase):
    """The extractor is exempt from weight decay; biases and norms always are."""

    def setUp(self) -> None:
        torch.manual_seed(13)
        self.model = build_local_snr_model(
            AudioFeaturesConfig(window_size=512, hop_size=160),
            ModelConfig(
                classes_num=4,
                extractor_type="demucs",
                demucs_hidden=8,
                demucs_depth=3,
                demucs_lstm_hidden=16,
            ),
        )
        self.regularization = RegularizationConfig(
            l2_lambda=1e-4, extractor_l2_lambda=0.0
        )

    def test_every_trainable_parameter_lands_in_exactly_one_group(self) -> None:
        groups = split_parameter_groups(self.model, self.regularization)
        seen = [id(p) for group in groups for p in group["params"]]
        expected = [id(p) for p in self.model.parameters() if p.requires_grad]
        self.assertEqual(len(seen), len(set(seen)))
        self.assertEqual(set(seen), set(expected))

    def test_extractor_weights_are_exempt_from_decay(self) -> None:
        groups = split_parameter_groups(self.model, self.regularization)
        extractor = {id(p) for p in self.model.noise_extractor.parameters()}
        for group in groups:
            if group["weight_decay"] > 0.0:
                for parameter in group["params"]:
                    self.assertNotIn(id(parameter), extractor)
                    self.assertGreater(parameter.ndim, 1)

    def test_the_rest_of_the_model_still_gets_decayed(self) -> None:
        groups = split_parameter_groups(self.model, self.regularization)
        decayed = {
            id(p)
            for group in groups
            if group["weight_decay"] == self.regularization.l2_lambda
            for p in group["params"]
        }
        classifier_weights = [
            p for p in self.model.classifier.parameters() if p.ndim > 1
        ]
        self.assertTrue(classifier_weights)
        for parameter in classifier_weights:
            self.assertIn(id(parameter), decayed)

    def test_a_nonzero_extractor_lambda_creates_its_own_group(self) -> None:
        groups = split_parameter_groups(
            self.model, RegularizationConfig(l2_lambda=1e-4, extractor_l2_lambda=1e-5)
        )
        decays = sorted(group["weight_decay"] for group in groups)
        self.assertEqual(decays, [0.0, 1e-5, 1e-4])


if __name__ == "__main__":
    unittest.main()

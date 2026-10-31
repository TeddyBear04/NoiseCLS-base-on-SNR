"""
Unit tests for Frontend and 6 Acoustic Features.

Verifies:
1. RMS energy normalization to exact target_rms = 0.07925.
2. STFT output shape: 257 frequency bins and 401 time frames for 4.0s (64,000 samples).
3. 6 Acoustic features (Log-RMS, Crest factor, Kurtosis, Flatness, f5, f6).
4. LogMel Spectrogram output shape: [B, 1, 64, 398] with instance normalization.
"""

import torch
from src.frontend import (
    TARGET_RMS,
    AcousticFeatureExtractor,
    LogMel,
    SpecAugment,
    rms_normalize,
)


def test_rms_normalization(sample_waveform):
    # Test with varying amplitudes
    wave_quiet = sample_waveform * 0.01
    wave_loud = sample_waveform * 10.0

    norm_quiet = rms_normalize(wave_quiet)
    norm_loud = rms_normalize(wave_loud)

    rms_q = norm_quiet.square().mean(dim=-1).sqrt()
    rms_l = norm_loud.square().mean(dim=-1).sqrt()

    # Both should be standardized exactly to target_rms = 0.07925
    assert torch.allclose(rms_q, torch.tensor([TARGET_RMS, TARGET_RMS]), atol=1e-4)
    assert torch.allclose(rms_l, torch.tensor([TARGET_RMS, TARGET_RMS]), atol=1e-4)


def test_stft_dimensions_and_frames(sample_waveform):
    extractor = AcousticFeatureExtractor(n_fft=512, hop_length=160)
    power = extractor.compute_stft_power(sample_waveform)

    # 512 / 2 + 1 = 257 frequency bins
    # 64000 / 160 + 1 = 401 time frames (with center=True)
    assert power.shape == (2, 257, 401)


def test_acoustic_6_features(sample_waveform):
    wave_norm = rms_normalize(sample_waveform)
    extractor = AcousticFeatureExtractor(n_fft=512, hop_length=160)
    features = extractor(wave_norm)

    # Output must be [B, 6]
    assert features.shape == (2, 6)
    assert not torch.isnan(features).any()
    assert not torch.isinf(features).any()

    # Physical validity checks:
    # f1: Log-RMS should be close to log(0.07925) ~= -2.535
    f1_log_rms = features[:, 0]
    expected_log_rms = torch.log(torch.tensor(TARGET_RMS))
    assert torch.allclose(f1_log_rms, expected_log_rms.expand(2), atol=1e-2)

    # f2: Crest factor (peak / rms) must be >= 1.0
    f2_crest = features[:, 1]
    assert (f2_crest >= 1.0).all()

    # f3: Kurtosis of power distribution must be strictly positive
    f3_kurt = features[:, 2]
    assert (f3_kurt > 0.0).all()

    # f4: Spectral Flatness (Wiener entropy) must be in (0, 1]
    f4_flatness = features[:, 3]
    assert (f4_flatness > 0.0).all()
    assert (f4_flatness <= 1.0 + 1e-4).all()


def test_logmel_and_specaugment(sample_waveform):
    wave_norm = rms_normalize(sample_waveform)
    logmel = LogMel(n_mels=64, num_frames=398)
    mel = logmel(wave_norm)

    # Output shape: [B, 1, 64, 398]
    assert mel.shape == (2, 1, 64, 398)

    # Instance normalization check: mean ~= 0, std ~= 1 across time & frequency
    mean = mel.mean(dim=(-2, -1))
    std = mel.std(dim=(-2, -1))
    assert torch.allclose(mean, torch.zeros_like(mean), atol=1e-3)
    assert torch.allclose(std, torch.ones_like(std), atol=1e-2)

    # SpecAugment check
    spec_aug = SpecAugment(freq_masks=2, time_masks=2)
    spec_aug.train()
    aug_mel = spec_aug(mel)
    assert aug_mel.shape == mel.shape

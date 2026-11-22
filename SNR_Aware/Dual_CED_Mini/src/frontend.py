"""
Acoustic Front-End and Feature Extraction for Dual-Expert CED:
- Log-Mel Spectrogram extraction (16 kHz, 64 mels, 398 frames, win_length 400, hop 160, HTK filterbank)
- Instance Z-Score Normalization matching CED-Tiny pretraining
- 2D SpecAugment (Frequency and Time Masking)
- Physical RMS Normalization (target: 0.07925)
- 6 Physical Acoustic Morphology Features for Soft MoE Routing

References:
    - CED: Dinkel et al., "CED: Consistent Ensemble Distillation for Audio Tagging", ICASSP 2024. arXiv:2308.11957.
    - SpecAugment: Park et al., "SpecAugment: A Simple Data Augmentation Method for ASR", Interspeech 2019. arXiv:1904.08779.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

RMS_EPS = 1e-8
TARGET_RMS = 0.07925


def rms_normalize(waveform: Tensor, target_rms: float = 0.07925) -> Tensor:
    """
    Standardize the energy level of the waveform to target_rms (0.07925).
    Formula: x_norm = x * (target_rms / (sqrt(mean(x^2)) + eps))
    """
    if waveform.ndim == 1:
        waveform = waveform.unsqueeze(0)
    rms = waveform.square().mean(dim=-1).sqrt()
    return waveform * (target_rms / (rms + RMS_EPS))[:, None]


def create_htk_mel_filterbank(
    sr: int = 16000,
    n_fft: int = 512,
    n_mels: int = 64,
    f_min: float = 20.0,
    f_max: float = 8000.0,
) -> Tensor:
    """Construct standard HTK Mel Filterbank matrix identical to torchaudio."""
    min_mel = 2595.0 * np.log10(1.0 + f_min / 700.0)
    max_mel = 2595.0 * np.log10(1.0 + f_max / 700.0)
    mel_pts = np.linspace(min_mel, max_mel, n_mels + 2)
    f_pts = 700.0 * (10.0 ** (mel_pts / 2595.0) - 1.0)
    fft_freqs = np.linspace(0.0, sr / 2.0, n_fft // 2 + 1)
    weights = np.zeros((n_mels, n_fft // 2 + 1), dtype=np.float32)
    for i in range(n_mels):
        left, center, right = f_pts[i], f_pts[i + 1], f_pts[i + 2]
        up = (fft_freqs - left) / (center - left)
        down = (right - fft_freqs) / (right - center)
        weights[i] = np.maximum(0.0, np.minimum(up, down))
    return torch.from_numpy(weights)


class LogMelFrontend(nn.Module):
    """
    Pure PyTorch Log-Mel Spectrogram Frontend matching the CED specification:
    - Sample rate: 16kHz
    - FFT: 512, Win length: 400, Hop length: 160
    - Mel bins: 64, Frequency range: 20Hz - 8000Hz
    - Instance normalization over (frequency, time)
    """

    def __init__(
        self,
        sample_rate: int = 16000,
        n_fft: int = 512,
        win_length: int = 400,
        hop_length: int = 160,
        n_mels: int = 64,
        f_min: float = 20.0,
        f_max: float = 8000.0,
        target_frames: int = 398,
        log_eps: float = 1e-6,
        norm_eps: float = 1e-5,
    ) -> None:
        super().__init__()
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.win_length = win_length
        self.hop_length = hop_length
        self.target_frames = target_frames
        self.log_eps = log_eps
        self.norm_eps = norm_eps
        # torch.stft frames use n_fft samples even when the analysis window is
        # shorter. Center the shorter window inside each n_fft frame with
        # symmetric external zero padding so that center=False yields the
        # paper's floor((L - win_length) / hop_length) + 1 frame alignment.
        self.pad = max(n_fft - win_length, 0)

        self.register_buffer("window", torch.hann_window(win_length))
        mel_fb = create_htk_mel_filterbank(sample_rate, n_fft, n_mels, f_min, f_max)
        self.register_buffer("mel_fb", mel_fb)

    def logmel(self, waveform: Tensor) -> Tensor:
        if waveform.ndim == 1:
            waveform = waveform.unsqueeze(0)
        if self.pad > 0:
            pad_left = self.pad // 2
            pad_right = self.pad - pad_left
            waveform = F.pad(waveform, (pad_left, pad_right))
        # STFT computation
        stft = torch.stft(
            waveform,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=self.window.to(waveform.device),
            center=False,
            return_complex=True,
        )
        power = stft.abs().square()  # [B, n_fft // 2 + 1, Frames]
        mel = torch.matmul(self.mel_fb.to(waveform.device), power)
        return (mel + self.log_eps).log()

    def _offsets(self, frames: int, batch: int, device: torch.device) -> Tensor:
        limit = frames - self.target_frames
        if self.training and limit > 0:
            return torch.randint(0, limit + 1, (batch,), device=device)
        return torch.full((batch,), max(limit, 0) // 2, dtype=torch.long, device=device)

    def _crop(self, x: Tensor, offsets: Tensor) -> Tensor:
        frames = x.shape[-1]
        if frames == self.target_frames:
            return x
        if frames < self.target_frames:
            return F.pad(x, (0, self.target_frames - frames))
        index = offsets[:, None] + torch.arange(self.target_frames, device=x.device)[None, :]
        return x.gather(2, index[:, None, :].expand(-1, x.shape[1], -1))

    def forward(self, waveform: Tensor) -> Tensor:
        x = self.logmel(waveform)
        mean = x.mean(dim=(-2, -1), keepdim=True)
        std = x.std(dim=(-2, -1), keepdim=True)
        x = (x - mean) / (std + self.norm_eps)
        cropped = self._crop(x, self._offsets(x.shape[-1], x.shape[0], x.device))
        return cropped.unsqueeze(1)  # [B, 1, 64, 398]


class SpecAugment(nn.Module):
    """
    Frequency and Time Masking SpecAugment on 2D Spectrogram [B, 1, F, T].
    """

    def __init__(self, freq_masks: int = 2, freq_width: int = 8, time_masks: int = 2, time_width: int = 24) -> None:
        super().__init__()
        self.freq_masks = freq_masks
        self.freq_width = freq_width
        self.time_masks = time_masks
        self.time_width = time_width

    @staticmethod
    def _zero_band(x: Tensor, sample: int, dim: int, width: int) -> None:
        size = x.shape[dim]
        w = int(torch.randint(0, width + 1, (1,)).item())
        if w == 0 or w > size:
            return
        start = int(torch.randint(0, size - w + 1, (1,)).item())
        if dim == 2:
            x[sample, :, start : start + w, :] = 0.0
        else:
            x[sample, :, :, start : start + w] = 0.0

    def forward(self, x: Tensor) -> Tensor:
        if not self.training or (self.freq_masks == 0 and self.time_masks == 0):
            return x
        x = x.clone()
        for sample in range(x.shape[0]):
            for _ in range(self.freq_masks):
                self._zero_band(x, sample, dim=2, width=self.freq_width)
            for _ in range(self.time_masks):
                self._zero_band(x, sample, dim=3, width=self.time_width)
        return x


class AcousticFeatureExtractor(nn.Module):
    """
    Extracts 6 intrinsic acoustic and spectral morphology features directly from waveform:
    1. Log-RMS energy
    2. Crest factor (peak amplitude to RMS ratio)
    3. Frame-wise energy kurtosis
    4. Spectral Flatness (Wiener entropy: ratio of geometric mean to arithmetic mean)
    5. Low-band to Speech-band energy ratio (<300 Hz vs 300-3400 Hz)
    6. High-band to Speech-band energy ratio (>3400 Hz vs 300-3400 Hz)
    """

    def __init__(self, sample_rate: int = 16000, frame_len: int = 512, hop_len: int = 160) -> None:
        super().__init__()
        self.sample_rate = sample_rate
        self.frame_len = frame_len
        self.hop_len = hop_len
        self.register_buffer("window", torch.hann_window(frame_len))

    def forward(self, wave: Tensor) -> Tensor:
        if wave.ndim == 1:
            wave = wave.unsqueeze(0)
        B, T = wave.shape
        eps = 1e-8

        # 1. Log-RMS energy
        rms = wave.square().mean(dim=-1).clamp(min=eps).sqrt()
        log_rms = torch.log(rms + eps).unsqueeze(-1)  # [B, 1]

        # 2. Crest factor (Peak / RMS)
        peak = wave.abs().max(dim=-1).values
        crest_factor = (peak / (rms + eps)).unsqueeze(-1)  # [B, 1]

        # 3. Temporal Kurtosis (Peakiness of frame energy)
        frames = wave.unfold(dimension=-1, size=self.frame_len, step=self.hop_len)
        frame_energy = frames.square().mean(dim=-1) + eps  # [B, num_frames]
        fe_mean = frame_energy.mean(dim=-1, keepdim=True)
        fe_std = frame_energy.std(dim=-1, keepdim=True) + eps
        kurtosis = (((frame_energy - fe_mean) / fe_std) ** 4).mean(dim=-1, keepdim=True)  # [B, 1]

        # STFT for spectral flatness and band energy ratios
        stft = torch.stft(
            wave,
            n_fft=self.frame_len,
            hop_length=self.hop_len,
            win_length=self.frame_len,
            window=self.window.to(wave.device),
            center=True,
            return_complex=True,
        )
        spec = stft.abs()  # [B, 257, num_frames]
        freq_bins = spec.shape[1]
        freqs = torch.linspace(0, self.sample_rate / 2, freq_bins, device=wave.device)

        # 4. Spectral Flatness (Wiener entropy: geo_mean / arith_mean)
        arith_mean = spec.mean(dim=1, keepdim=True) + eps
        log_spec = torch.log(spec + eps)
        geo_mean = torch.exp(log_spec.mean(dim=1, keepdim=True))
        flatness = (geo_mean / arith_mean).mean(dim=-1)  # [B, 1]

        # 5. Low-band to Speech-band ratio (<300 Hz vs 300-3400 Hz)
        # 6. High-band to Speech-band ratio (>3400 Hz vs 300-3400 Hz)
        low_mask = freqs < 300.0
        speech_mask = (freqs >= 300.0) & (freqs <= 3400.0)
        high_mask = freqs > 3400.0

        p_low = spec[:, low_mask, :].square().mean(dim=(1, 2)) + eps
        p_speech = spec[:, speech_mask, :].square().mean(dim=(1, 2)) + eps
        p_high = spec[:, high_mask, :].square().mean(dim=(1, 2)) + eps

        f5 = (10.0 * torch.log10(p_low / p_speech)).unsqueeze(-1)
        f6 = (10.0 * torch.log10(p_high / p_speech)).unsqueeze(-1)

        features = torch.cat([log_rms, crest_factor, kurtosis, flatness, f5, f6], dim=-1)  # [B, 6]
        return features

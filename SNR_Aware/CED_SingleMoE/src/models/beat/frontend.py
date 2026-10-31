from __future__ import annotations

import sys
import types
import torch
from torch import nn

# Safe initialization for torchaudio to bypass broken C++ shared library on Linux/Python 3.13
if "torchaudio._extension" not in sys.modules:
    mock_ext = types.ModuleType("torchaudio._extension")
    mock_ext._IS_TORCHAUDIO_EXT_AVAILABLE = False
    mock_ext._IS_ALIGN_AVAILABLE = False
    mock_ext.fail_if_no_align = lambda fn=None, *args, **kwargs: fn if fn is not None else (lambda f: f)
    mock_ext._check_cuda_version = lambda *args, **kwargs: None
    sys.modules["torchaudio._extension"] = mock_ext

import torchaudio.compliance.kaldi as ta_kaldi


class Fbank(nn.Module):
    def __init__(
        self,
        sample_rate=16000,
        num_mel_bins=128,
        frame_length=25,
        frame_shift=10,
        fbank_mean=15.41663,
        fbank_std=6.55582,
    ):
        super().__init__()
        self.sample_rate = sample_rate
        self.num_mel_bins = num_mel_bins
        self.frame_length = frame_length
        self.frame_shift = frame_shift
        self.mean = fbank_mean
        self.std = fbank_std

    @torch.no_grad()
    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        waveform = waveform * (2**15)
        feats = [
            ta_kaldi.fbank(
                w.unsqueeze(0),
                num_mel_bins=self.num_mel_bins,
                sample_frequency=self.sample_rate,
                frame_length=self.frame_length,
                frame_shift=self.frame_shift,
            )
            for w in waveform
        ]
        feats = torch.stack(feats)
        return (feats - self.mean) / self.std

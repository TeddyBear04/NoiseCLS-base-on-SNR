from __future__ import annotations
import sys
import types
import torch
from torch import nn
from . import kaldi as ta_kaldi

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


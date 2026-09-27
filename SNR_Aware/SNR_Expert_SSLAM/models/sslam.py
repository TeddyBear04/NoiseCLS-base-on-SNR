"""SSLAM backbone: loading, mel features, and the two heads we bolt on.

SSLAM (Alex, Atito, Mustafa, Awais, Jackson - ICLR 2025, "Enhancing Self-Supervised
Models with Audio Mixtures for Polyphonic Soundscapes") tops the AudioSet board at
0.502 mAP against BEATs iter3's 0.480, and it is trained on mixtures rather than
isolated clips, which is the setting this project lives in.

Verified on the sandbox before any of this was written:

    transformers 4.57.6 + timm 1.0.30  ->  EATModel, 90.0M params
    extract_features(mel[B, 1, T, 128]) -> tensor [B, N_patch + 1, 768]
    T=1024 -> 513 tokens,  T=512 -> 257,  T=400 -> 201

Three things fall out of that and shape everything here:

  * 4-second clips (about 400 frames) run natively. No padding to the 1024 frames
    the model card suggests, so we do not feed the encoder 60% silence.
  * It returns PATCH TOKENS, not a pooled vector. That is the point of switching:
    the BEATs pipeline threw this structure away at `sequence.mean(dim=1)` before
    the distillation loss ever saw it.
  * 768 dims, same as BEATs, so the head, FiLM, CRD and Projection code carries
    over unchanged.

transformers 5.x does NOT work: the hub's remote code predates it and raises
`AttributeError: 'EATModel' object has no attribute 'all_tied_weights_keys'`.
Pin `transformers<5`.
"""

from __future__ import annotations

import torch
import torchaudio
from torch import Tensor, nn

SSLAM_PRETRAIN = "ta012/SSLAM_pretrain"

# From the model card. AST and EAT, which SSLAM descends from, divide by twice the
# standard deviation; the card gives only the two constants, so `norm_divisor` in
# the config decides and 2.0 is the family's convention. If features look wrong,
# this is the first knob to try - it is documented, not guessed at silently.
FBANK_MEAN = -4.268
FBANK_STD = 4.569


def load_sslam(device: torch.device, model_id: str = SSLAM_PRETRAIN):
    """Frozen-by-default SSLAM encoder. Caller unfreezes what it wants to train."""
    from transformers import AutoModel

    model = AutoModel.from_pretrained(model_id, trust_remote_code=True)
    model.to(device).eval()
    for parameter in model.parameters():
        parameter.requires_grad = False
    return model


def waveform_to_mel(waveform: Tensor, sample_rate: int = 16_000,
                    num_mel_bins: int = 128, norm_divisor: float = 2.0) -> Tensor:
    """Kaldi-compatible log-mel, shaped and normalised the way SSLAM expects.

    waveform [B, samples] -> mel [B, 1, frames, num_mel_bins]

    `torchaudio.compliance.kaldi.fbank` takes one clip at a time, so this loops.
    At batch 32 and 4-second clips the loop is not where the time goes; the encoder
    forward is.
    """
    if waveform.dim() == 1:
        waveform = waveform.unsqueeze(0)
    banks = []
    for clip in waveform:
        bank = torchaudio.compliance.kaldi.fbank(
            clip.unsqueeze(0), htk_compat=True, sample_frequency=sample_rate,
            use_energy=False, window_type="hanning", num_mel_bins=num_mel_bins,
            dither=0.0, frame_shift=10,
        )
        banks.append(bank)
    mel = torch.stack(banks)
    mel = (mel - FBANK_MEAN) / (FBANK_STD * norm_divisor)
    return mel.unsqueeze(1)


class SSLAMEncoder(nn.Module):
    """SSLAM plus mel computation, returning both patch tokens and a pooled vector.

    Patch tokens are what the CRD term aligns. The pooled vector still exists
    because the classifier head and FiLM need one, and because keeping the pooled
    path identical to the BEATs project is what makes the two comparable.

    Token 0 is the CLS token. Pooling averages the patch tokens after it rather
    than taking CLS, matching how the BEATs pipeline pooled, so a difference
    between the two projects is a difference in backbone and loss, not in pooling.
    """

    def __init__(self, model, num_mel_bins: int = 128, norm_divisor: float = 2.0,
                 sample_rate: int = 16_000):
        super().__init__()
        self.model = model
        self.num_mel_bins = num_mel_bins
        self.norm_divisor = norm_divisor
        self.sample_rate = sample_rate

    def forward(self, waveform: Tensor) -> tuple[Tensor, Tensor]:
        mel = waveform_to_mel(waveform, self.sample_rate, self.num_mel_bins,
                              self.norm_divisor).to(waveform.device)
        tokens = self.model.extract_features(mel)
        patches = tokens[:, 1:, :]
        return patches, patches.mean(dim=1)


def trainable_block_names(model, blocks: int) -> tuple[str, ...]:
    """Prefixes of the last `blocks` transformer layers, for selective unfreezing.

    The module path differs between backbones, so find the deepest container that
    looks like a list of transformer blocks rather than assuming `encoder.layers`
    the way the BEATs project could.
    """
    candidates = [(name, module) for name, module in model.named_modules()
                  if isinstance(module, (nn.ModuleList, nn.Sequential))
                  and len(module) >= blocks]
    if not candidates:
        raise RuntimeError("No block list found in the SSLAM model to unfreeze.")
    name, module = max(candidates, key=lambda item: len(item[1]))
    first = len(module) - blocks
    return tuple(f"{name}.{index}." for index in range(first, len(module)))


def unfreeze_last_blocks(model, blocks: int) -> int:
    prefixes = trainable_block_names(model, blocks)
    count = 0
    for name, parameter in model.named_parameters():
        if name.startswith(prefixes):
            parameter.requires_grad = True
            count += parameter.numel()
    return count

"""EfficientAT MobileNetV3 backbone behind the same interface as `models.sslam.SSLAMEncoder`.

EfficientAT (Schmid, Koutini, Widmer - ICASSP 2023, "Efficient Large-Scale Audio
Tagging via Transformer-to-CNN Knowledge Distillation";
https://github.com/fschmid56/EfficientAT) distils an ensemble of PaSST transformers
into MobileNetV3 CNNs. mn30 = width 3.0: about 39M parameters, 48.2 mAP on
AudioSet. What is taken from its code, step by step:

  * resample   The checkpoints were trained at 32 kHz; our clips are 16 kHz, so
               they are resampled up. Nothing above 8 kHz exists in them, so the
               mel bins above ~8 kHz stay empty - unavoidable, worth knowing.
  * mel        `models/preprocess.AugmentMelSTFT` as its inference uses it: 128
               bins, window 800, hop 320, n_fft 1024, pre-emphasis, log, then
               (x + 4.5) / 5. fmax 15 kHz is what its default resolves to
               (sr / 2 - fmax_aug_range / 2). Its own fmin/fmax jitter and
               freq/time masking are switched off: this project's SpecAugment
               replaces them, on the normalised log-mel as for every backbone.
               4 s -> 401 frames.
  * features   The 17 conv stages (stem, 15 inverted residuals, 1x1 expansion to
               2880 channels). Overall stride 32 -> a 4 frequency x 13 time grid.
  * head       Its 'mlp' classifier: global average pool, Linear(2880, 3840),
               Hardswish, Dropout 0.2, Linear(3840, 527). `PretrainedHead` keeps
               the first three and cuts the last Linear to our 36 labels' rows.

Tokens are the 52 cells of the last feature map, time-major (t * 4 + f), and the
pooled vector is their mean - exactly the classifier's own AdaptiveAvgPool2d(1).

FiLM and the head. FiLM always LayerNorms the pooled vector before modulating it.
BEATs' and CED's pooled vectors already sit at LayerNorm scale, so FiLM starts as
(near) identity, as `expert_lib.FiLM` intends. This CNN's do not: over 256 train
mixtures each clip's 2880-d vector has mean -0.0007 and std 0.0317 (+- 0.0009
across clips), so LayerNorm multiplies it by ~31 and the pretrained head's
cross-entropy goes from 2.10 to 44.2 (measured on molab, 2026-10-06). With a
per-clip std that steady, LN(x) ~ (x - mean) / std exactly enough to fold back
into the head's first Linear: W' = std * W, b' = b + mean * W 1, so that
head(LN(x)) ~ head(x) at initialisation. Only when FiLM is on - without it the
head sees the raw vector it was trained on.
"""

from __future__ import annotations

import contextlib
import io

import torch
from torch import Tensor, nn

from expert_lib import roll_waveform, spec_augment
from models.ced_backbone import PretrainedHead, copied
from models.external import import_isolated
from models.sslam import audioset_indices

EFFICIENTAT_SAMPLE_RATE = 32_000


def load_efficientat(device: torch.device, backbone: dict):
    """(network, mel front end), the checkpoint loaded strictly, frozen, on `device`."""
    model_module, preprocess = import_isolated(
        backbone["repo_dir"], backbone["repo_url"], backbone["commit"],
        "models.mn.model", "models.preprocess")
    with contextlib.redirect_stdout(io.StringIO()):  # get_model prints the whole net
        net = model_module.get_model(num_classes=527, pretrained_name=None,
                                     width_mult=backbone["width_mult"], head_type="mlp")
    url = model_module.pretrained_models[backbone["variant"]]
    net.load_state_dict(torch.hub.load_state_dict_from_url(url, map_location="cpu"),
                        strict=True)
    mel = preprocess.AugmentMelSTFT(
        n_mels=128, sr=EFFICIENTAT_SAMPLE_RATE, win_length=800, hopsize=320, n_fft=1024,
        freqm=0, timem=0, fmin=0.0, fmax=15_000, fmin_aug_range=1, fmax_aug_range=1)
    net.to(device).eval()
    mel.to(device).eval()
    for parameter in net.parameters():
        parameter.requires_grad = False
    return net, mel


class EfficientATEncoder(nn.Module):
    """MobileNetV3 features with an optional train-time augmentation hook."""

    def __init__(self, net, mel, sample_rate: int = 16_000):
        super().__init__()
        self.net = net
        self.mel = mel
        self.sample_rate = sample_rate
        self.pooling = "mean_cells"
        self.freq_bands = 4
        self.grid: tuple[int, int] | None = None
        self.augment: dict = {}
        self.generator = torch.Generator().manual_seed(0)

    def blocks(self) -> nn.Sequential:
        """The 17 conv stages, bottom to top."""
        return self.net.features

    def forward(self, waveform: Tensor) -> tuple[Tensor, Tensor]:
        import torchaudio

        augment = self.training and bool(self.augment)
        if augment and self.augment.get("roll"):
            waveform = roll_waveform(waveform, self.generator)
        with torch.autocast(device_type=waveform.device.type, enabled=False):
            audio = torchaudio.functional.resample(waveform.float(), self.sample_rate,
                                                   EFFICIENTAT_SAMPLE_RATE)
            spec = self.mel(audio)                                 # [B, 128, frames]
            if augment and self.augment.get("spec_ratio", 0) > 0:
                spec = spec_augment(spec.transpose(1, 2), self.augment["spec_ratio"],
                                    self.generator).transpose(1, 2)
        x = self.net.features(spec.unsqueeze(1))                   # [B, C, F, T]
        batch, channels, freq, time = x.shape
        self.freq_bands = freq
        self.grid = (time, freq)
        tokens = x.permute(0, 3, 2, 1).reshape(batch, time * freq, channels)
        return tokens, x.mean(dim=(2, 3))


def build_efficientat(device: torch.device, backbone: dict, label_mids: list[str],
                      sample_rate: int, layernormed: bool) -> tuple[EfficientATEncoder,
                                                                     PretrainedHead]:
    net, mel = load_efficientat(device, backbone)
    classifier = net.classifier            # pool, flatten, Linear, Hardswish, Dropout, Linear
    first = copied(classifier[2])
    if layernormed:                        # FiLM's LayerNorm sits in front (see above)
        stats = backbone["pooled_stats"]
        with torch.no_grad():
            first.bias.add_(stats["mean"] * first.weight.sum(dim=1))
            first.weight.mul_(stats["std"])
        print(f"head input = LayerNorm(pooled): first layer rescaled by the pooled "
              f"vector's std {stats['std']} (mean {stats['mean']})", flush=True)
    head = PretrainedHead([first, nn.Hardswish(), nn.Dropout(classifier[4].p)],
                          classifier[5], audioset_indices(label_mids))
    net.classifier = nn.Identity()         # replaced by `head`; keeps the param count honest
    return EfficientATEncoder(net, mel, sample_rate), head

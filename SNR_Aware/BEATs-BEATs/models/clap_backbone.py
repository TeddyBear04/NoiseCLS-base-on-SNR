"""LAION-CLAP backbone behind the same interface as `models.sslam.SSLAMEncoder`.

CLAP (Wu, Chen, Zhang, Hui, Berg-Kirkpatrick, Dubnov - ICASSP 2023, "Large-scale
Contrastive Language-Audio Pretraining with Feature Fusion and Keyword-to-Caption
Augmentation"; https://github.com/LAION-AI/CLAP) trains an HTSAT audio encoder
against a RoBERTa text encoder, so audio and text land in one 512-d space. What
this project takes from the paper, section by section:

  * 2.3 / 4.1  48 kHz mono, 10-second input, STFT window 1024, hop 480, 64 mel
               bins -> a (1024 x 64) spectrogram. Our clips are 16 kHz, so they are
               resampled; nothing above 8 kHz exists in them, so the top of CLAP's
               50 Hz - 14 kHz mel range stays empty. Unavoidable, worth knowing.
  * 3.4        T <= 10 s: "we first repeat the input, then pad it with zero values".
               A 4 s clip becomes 4 + 4 s of audio and 2 s of zeros. Stretching it
               to 10 s instead would show the encoder a time scale it never saw.
  * 3.3        HTSAT: 4 groups of Swin blocks, penultimate output 768-d, then a
               2-layer MLP into the joint space.
  * Fig. 1     Supervised classification adds the class layer AFTER that MLP:
               Audio Encoder -> MLP -> Projection Layers. `CLAPHead` is exactly that.
  * 4.3        Zero-shot prompt "This a sound of label." (the README writes "This
               is a sound of"). `CLAPHead` starts from those text embeddings, so
               epoch 0 IS CLAP's zero-shot classifier - the zero-shot head as the
               initialisation for end-to-end fine-tuning, as WiSE-FT does (Wortsman
               et al., CVPR 2022), and the same LP-FT argument the AudioSet-row
               warm start of the other two backbones rests on.

Verified on molab (2026-10-04), `630k-audioset-best.pt`, laion_clap 1.1.6:
every key loads, layers [2, 2, 6, 2], 64 tokens x 768, logit scale 23.39;
zero-shot on 360 test clips: 0.553 on clean noise, 0.297 on the mixture.

Token layout. HTSAT folds the (1024 x 64) spectrogram into a 256 x 256 image (four
256-frame chunks stacked along the frequency axis) and ends on an 8 x 8 grid. Its
own `forward_features` unfolds that grid into 2 frequency halves x 32 time steps
(0.32 s each); this module returns the tokens in that order, time-major
(index t * 2 + f, like BEATs' t * 8 + f). The steps that fall entirely on the
zero padding are dropped from the patch tokens: there the tokens of every clip
look alike, so a patch-level CRD anchor would be asked to tell apart negatives
that carry no signal. The pooled vector still averages all 64 tokens, which is
exactly HTSAT's own `embedding`, so the pretrained MLP sees what it was trained on.

Install on molab WITHOUT dependencies - laion_clap 1.1.6 pins numpy < 2.0 and
would downgrade the environment torch was built against:

    uv pip install --no-deps laion-clap==1.1.6
    uv pip install torchlibrosa librosa ftfy braceexpand webdataset wget h5py \
        regex progressbar   (with numpy / torch / torchvision pinned to what is there)
"""

from __future__ import annotations

import torch
from torch import Tensor, nn

from expert_lib import roll_waveform, spec_augment

CLAP_SAMPLE_RATE = 48_000
CLAP_SECONDS = 10.0
TIME_STEPS = 32  # HTSAT's unfolded time axis: 8 columns x freq_ratio 4 chunks
FREQ_BANDS = 2   # ... and its frequency axis: mel bins 0-31 and 32-63


def load_clap(device: torch.device, repo: str, filename: str, amodel: str = "HTSAT-tiny"):
    """CLAP_Module with the checkpoint loaded, frozen, on `device`.

    The checkpoint is read here rather than through `CLAP_Module.load_ckpt`:
    that path calls `torch.load` with the new `weights_only=True` default, and the
    file is a full training checkpoint (epoch, optimizer). The one key that later
    transformers versions dropped, `text_branch.embeddings.position_ids`, is
    removed the same way the library's own `load_state_dict` does.
    """
    import laion_clap
    from huggingface_hub import hf_hub_download

    clap = laion_clap.CLAP_Module(enable_fusion=False, amodel=amodel, device=str(device))
    saved = torch.load(hf_hub_download(repo, filename), map_location="cpu",
                       weights_only=False)
    state = saved.get("state_dict", saved)
    state = {k[len("module."):] if k.startswith("module.") else k: v
             for k, v in state.items()}
    state.pop("text_branch.embeddings.position_ids", None)
    clap.model.load_state_dict(state)
    clap.eval()
    for parameter in clap.parameters():
        parameter.requires_grad = False
    return clap


def repeat_pad(waveform: Tensor, max_len: int) -> tuple[Tensor, int]:
    """CLAP's `repeatpad` (paper 3.4) for a batch of equal-length clips.

    Returns the padded batch and how many samples of it are signal, so the caller
    can tell padding tokens from audio tokens.
    """
    length = waveform.shape[-1]
    if length >= max_len:
        return waveform[..., :max_len], max_len
    repeats = max_len // length
    filled = waveform.repeat(1, repeats)
    return nn.functional.pad(filled, (0, max_len - filled.shape[-1])), filled.shape[-1]


def signal_steps(signal: int, max_len: int) -> int:
    """Time steps (of 32) that lie wholly on signal; the rest touch the zero pad."""
    return max(1, min(TIME_STEPS, (signal * TIME_STEPS) // max_len))


def unfold_tokens(tokens: Tensor, freq_ratio: int) -> Tensor:
    """HTSAT's 8 x 8 token grid -> [B, time, freq, C], as `forward_features` unfolds it.

    Grid rows are (time chunk, frequency half), because `reshape_wav2img` stacked
    the four 256-frame chunks along the frequency axis; columns are time within
    the chunk. Same reshapes as HTSAT's own code, then time moved first.
    """
    batch, count, channels = tokens.shape
    side = int(count ** 0.5)
    grid = tokens.transpose(1, 2).reshape(batch, channels, side, side)
    per_chunk = side // freq_ratio
    grid = grid.reshape(batch, channels, side // per_chunk, per_chunk, side)
    grid = grid.permute(0, 1, 3, 2, 4).reshape(batch, channels, per_chunk, -1)
    return grid.permute(0, 3, 2, 1)


class CLAPEncoder(nn.Module):
    """HTSAT run step by step, so SpecAugment can act on the log-mel.

    Same steps as `HTSAT_Swin_Transformer.forward` in the non-fusion branch, minus
    its own SpecAugment and mixup (this project's augmentation hook replaces them),
    and stopping at the token grid instead of the AudioSet head.
    """

    def __init__(self, clap, sample_rate: int = 16_000):
        super().__init__()
        self.htsat = clap.model.audio_branch
        self.sample_rate = sample_rate
        self.max_len = int(CLAP_SAMPLE_RATE * CLAP_SECONDS)
        self.pooling = "mean_tokens"
        self.freq_bands = FREQ_BANDS
        self.augment: dict = {}
        self.generator = torch.Generator().manual_seed(0)

    def blocks(self) -> nn.ModuleList:
        """The 12 Swin blocks, bottom to top, for unfreezing and layer-wise lr decay.

        A stage's patch merging (and the final LayerNorm) rides with that stage's
        last block, so "train the top k blocks" trains everything above them rather
        than leaving a frozen projection between two trained blocks.
        """
        entries = []
        stages = list(self.htsat.layers)
        for index, stage in enumerate(stages):
            blocks = list(stage.blocks)
            for position, block in enumerate(blocks):
                extras = []
                if position == len(blocks) - 1:
                    if stage.downsample is not None:
                        extras.append(stage.downsample)
                    if index == len(stages) - 1:
                        extras.append(self.htsat.norm)
                entries.append(nn.ModuleList([block, *extras]) if extras else block)
        return nn.ModuleList(entries)

    def forward(self, waveform: Tensor) -> tuple[Tensor, Tensor]:
        import torchaudio

        augment = self.training and bool(self.augment)
        if augment and self.augment.get("roll"):
            waveform = roll_waveform(waveform, self.generator)
        htsat = self.htsat
        # The front end runs in fp32: torchlibrosa computes the STFT as a conv1d,
        # and under bf16 autocast that conv would quantise the log-mel.
        with torch.autocast(device_type=waveform.device.type, enabled=False):
            audio = torchaudio.functional.resample(waveform.float(), self.sample_rate,
                                                   CLAP_SAMPLE_RATE)
            audio, signal = repeat_pad(audio, self.max_len)
            mel = htsat.logmel_extractor(htsat.spectrogram_extractor(audio))
            mel = htsat.bn0(mel.transpose(1, 3)).transpose(1, 3)  # [B, 1, frames, 64]
            if augment and self.augment.get("spec_ratio", 0) > 0:
                mel = spec_augment(mel.squeeze(1), self.augment["spec_ratio"],
                                   self.generator).unsqueeze(1)
        image = htsat.reshape_wav2img(mel)
        tokens = htsat.patch_embed(image)
        if htsat.ape:
            tokens = tokens + htsat.absolute_pos_embed
        tokens = htsat.pos_drop(tokens)
        for stage in htsat.layers:
            tokens, _ = stage(tokens)
        tokens = htsat.norm(tokens)                      # [B, 64, C] on an 8 x 8 grid

        ordered = unfold_tokens(tokens, htsat.freq_ratio)  # [B, 32 time, 2 freq, C]
        pooled = ordered.mean(dim=(1, 2))                  # == HTSAT's own `embedding`
        steps = signal_steps(signal, self.max_len)
        batch, _, bands, channels = ordered.shape
        patches = ordered[:, :steps].reshape(batch, steps * bands, channels)
        return patches, pooled


class CLAPHead(nn.Module):
    """Paper Fig. 1, supervised path: pretrained projection MLP, then the classes.

    `classes` starts as CLAP's zero-shot classifier: weight = scale * text
    embedding of each label prompt, bias 0, applied to the L2-normalised audio
    embedding - the same logits `CLAP` itself computes (scale = exp(logit_scale_a)).
    """

    def __init__(self, projection: nn.Module, text: Tensor | None, scale: float,
                 classes: int):
        super().__init__()
        self.projection = projection
        for parameter in self.projection.parameters():
            parameter.requires_grad = True  # finetuned with the class layer, at head_lr
        dim = projection[-1].out_features
        self.classes = nn.Linear(dim, classes)
        if text is not None:
            with torch.no_grad():
                self.classes.weight.copy_(scale * text)
                self.classes.bias.zero_()

    def forward(self, pooled: Tensor) -> Tensor:
        return self.classes(nn.functional.normalize(self.projection(pooled), dim=-1))


def build_clap(clap, labels: list[str], prompt: str, zero_shot: bool,
               sample_rate: int) -> tuple[CLAPEncoder, CLAPHead]:
    """Encoder plus head, both detached from the text branch, which is not kept."""
    text = None
    if zero_shot:
        with torch.no_grad():
            text = clap.get_text_embedding([prompt.format(label=label) for label in labels],
                                           use_tensor=True).float().cpu()
    scale = float(clap.model.logit_scale_a.exp())
    encoder = CLAPEncoder(clap, sample_rate)
    head = CLAPHead(clap.model.audio_projection, text, scale, len(labels))
    return encoder, head

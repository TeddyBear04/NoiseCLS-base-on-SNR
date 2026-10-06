"""CED backbone behind the same interface as `models.sslam.SSLAMEncoder`.

CED (Dinkel, Wang, Yan, Zhang, Wang - ICASSP 2024, "CED: Consistent Ensemble
Distillation for Audio Tagging"; https://github.com/RicherMans/CED) is a plain ViT
distilled from an ensemble of large AudioSet taggers. CED-Small: 12 blocks, 384-d,
6 heads, about 22M parameters, 49.6 mAP on AudioSet - against BEATs' ~90M. What is
taken from its code (`models/audiotransformer.py`), step by step:

  * FrontEnd   16 kHz - the rate of our clips, no resampling - n_fft 512, window
               512, hop 160, centred, 64 mel bins 0-8 kHz, AmplitudeToDB(top_db 120).
               4 s -> 401 frames.
  * init_bn    BatchNorm2d over the 64 mel bins. SpecAugment acts after it, on the
               normalised log-mel, so a masked band reads as average energy - the
               same convention as the BEATs fbank (see `expert_lib.spec_augment`).
  * patches    16 x 16, stride 16 -> 4 frequency x 25 time = 100 tokens (the 401st
               frame falls off the conv). Time position embeddings sliced to 25.
  * pooling    CED's own 'mean': the average of the normed tokens, then its
               `outputlayer` = LayerNorm + Linear(384, 527).

Token order. CED flattens 'b c f t -> b (f t) c' (frequency-major). This module
returns them time-major, index t * 4 + f, like BEATs' t * 8 + f, so the attention
summary and the patch-level CRD read one layout. The transformer itself does not
care: position embeddings are added before flattening and attention is
permutation-equivariant.

Head. `PretrainedHead` = CED's LayerNorm, then the Linear cut down to the rows of
our 36 labels (standard AudioSet order), so epoch 0 is CED's own tagger on our
labels - the same warm start the BEATs head gets.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn

from expert_lib import roll_waveform, spec_augment
from models.external import import_isolated
from models.sslam import audioset_indices


def load_ced(device: torch.device, backbone: dict):
    """CED with its AudioSet checkpoint loaded strictly, frozen, on `device`."""
    (module,) = import_isolated(backbone["repo_dir"], backbone["repo_url"],
                                backbone["commit"], "models.audiotransformer")
    model = getattr(module, backbone["variant"])(
        pretrained=False, drop_path_rate=backbone.get("drop_path", 0.0))
    state = torch.hub.load_state_dict_from_url(backbone["checkpoint_url"],
                                               map_location="cpu")
    state = state.get("model", state)
    model.load_state_dict(state, strict=True)
    model.to(device).eval()
    for parameter in model.parameters():
        parameter.requires_grad = False
    return model


class PretrainedHead(nn.Module):
    """A pretrained AudioSet head with its last Linear cut to our labels' rows."""

    def __init__(self, layers: list[nn.Module], last: nn.Linear, rows: Tensor):
        super().__init__()
        self.body = nn.Sequential(*layers)
        self.classes = nn.Linear(last.in_features, len(rows))
        with torch.no_grad():
            self.classes.weight.copy_(last.weight[rows])
            self.classes.bias.copy_(last.bias[rows])

    def forward(self, pooled: Tensor) -> Tensor:
        return self.classes(self.body(pooled))


def copied(module: nn.Module) -> nn.Module:
    """A trainable copy, so the head does not share frozen tensors with the encoder."""
    import copy
    clone = copy.deepcopy(module)
    for parameter in clone.parameters():
        parameter.requires_grad = True
    return clone


class CEDEncoder(nn.Module):
    """CED run step by step, so SpecAugment can act between init_bn and the patches."""

    def __init__(self, model):
        super().__init__()
        self.model = model
        self.pooling = "mean_tokens"
        self.freq_bands = model.patch_embed.grid_size[0]
        self.grid: tuple[int, int] | None = None
        self.augment: dict = {}
        self.generator = torch.Generator().manual_seed(0)

    def blocks(self) -> nn.ModuleList:
        """The 12 transformer blocks; the final LayerNorm rides with the top one, so
        training the top block also trains the norm that sits above it."""
        blocks = list(self.model.blocks)
        return nn.ModuleList(blocks[:-1] + [nn.ModuleList([blocks[-1], self.model.norm])])

    def forward(self, waveform: Tensor) -> tuple[Tensor, Tensor]:
        model = self.model
        augment = self.training and bool(self.augment)
        if augment and self.augment.get("roll"):
            waveform = roll_waveform(waveform, self.generator)
        with torch.autocast(device_type=waveform.device.type, enabled=False):
            spec = model.init_bn(model.front_end(waveform.float()).unsqueeze(1))
            if augment and self.augment.get("spec_ratio", 0) > 0:
                spec = spec_augment(spec.squeeze(1).transpose(1, 2), self.augment["spec_ratio"],
                                    self.generator).transpose(1, 2).unsqueeze(1)
        x = model.patch_embed(spec)                          # [B, C, F, T]
        batch, channels, freq, time = x.shape
        x = x + model.time_pos_embed[:, :, :, :time] + model.freq_pos_embed
        x = x.permute(0, 3, 2, 1).reshape(batch, time * freq, channels)  # t * F + f
        x = model.norm(model.blocks(model.pos_drop(x)))
        self.grid = (time, freq)
        return x, x.mean(dim=1)


def build_ced(device: torch.device, backbone: dict,
              label_mids: list[str]) -> tuple[CEDEncoder, PretrainedHead]:
    model = load_ced(device, backbone)
    norm, linear = model.outputlayer[0], model.outputlayer[1]
    head = PretrainedHead([copied(norm)], linear, audioset_indices(label_mids))
    model.outputlayer = nn.Identity()  # replaced by `head`; keeps the param count honest
    return CEDEncoder(model), head

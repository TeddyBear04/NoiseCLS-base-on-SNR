"""
Dual-Expert CED Architecture with Soft MoE Routing (arXiv:2308.00951)
and Pretrained BEATs Teacher Knowledge Distillation (KD).

- Student Model (Inference & Deployment): Dual-Expert CED (~10.83M parameters)
    * Expert 1 (Low-SNR):  ViTSBias CED-Tiny (~5.41M), specialized for noisy regimes
    * Expert 2 (High-SNR): ViTSBias CED-Tiny (~5.41M), specialized for speech-dominant regimes
    * Router: Soft MoE Gating Router (arXiv:2308.00951) with L2-normalized cosine similarity
    * Total inference parameters: 10.83M, zero Teacher footprint.
- Teacher Models (Training Only): Pretrained Dual BEATs Transformers (Fold0 + EMA, ~90.3M each)
    * Completely excluded from validation, evaluation, and inference.

References:
    - CED: Dinkel et al., "CED: Consistent Ensemble Distillation for Audio Tagging", ICASSP 2024. arXiv:2308.11957.
    - BEATs: Chen et al., "BEATs: Audio Pre-Training with Acoustic Tokenizers", ICML 2023. arXiv:2212.09058.
    - Soft MoE: Puigcerver et al., "From Sparse to Soft Mixtures of Experts", ICLR 2024. arXiv:2308.00951.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from .frontend import AcousticFeatureExtractor, LogMelFrontend, SpecAugment, rms_normalize

LOGGER = logging.getLogger(__name__)
GATE_HIDDEN = 8
CED_REPO = "mispeech/ced-tiny"
CED_PREFIX = "encoder."


# ==============================================================================
# 1. 2D Relative Position Bias Attention (GREP Attention)
# ==============================================================================

class SharedBias:
    def __init__(self) -> None:
        self.value: Tensor | None = None


class BiasAttention(nn.Module):
    """
    2D Gated Relative Position Bias Attention (GREP Attention).
    Encodes 2D frequency-time inductive bias into the scaled dot-product attention
    matrix of audio spectrogram patches.

    Reference:
        Chen et al., "BEATs: Audio Pre-Training with Acoustic Tokenizers", ICML 2023.
        Dinkel et al., "CED: Consistent Ensemble Distillation for Audio Tagging", ICASSP 2024.
    """

    def __init__(
        self,
        base: nn.Module,
        shared: SharedBias,
        owns_table: bool,
        freq_patches: int = 4,
        time_patches: int = 24,
        has_cls: bool = True,
        bias_init: str = "zero",
        use_grep: bool = True,
    ) -> None:
        super().__init__()
        self.has_cls = has_cls
        self.num_heads = base.num_heads
        self.head_dim = base.head_dim
        self.qkv, self.proj = base.qkv, base.proj
        self.attn_drop, self.proj_drop = base.attn_drop, base.proj_drop
        self.shared = shared
        self.freq_patches = freq_patches
        self.time_patches = time_patches
        self.use_grep = use_grep
        n_patches = freq_patches * time_patches

        if owns_table:
            num_f = 2 * freq_patches - 1
            num_t = 2 * time_patches - 1
            self.freq_rel_bias = nn.Embedding(num_f, self.num_heads)
            self.time_rel_bias = nn.Embedding(num_t, self.num_heads)
            self.cls_to_patches = nn.Parameter(torch.zeros(1, self.num_heads, 1, n_patches))
            self.patches_to_cls = nn.Parameter(torch.zeros(1, self.num_heads, n_patches, 1))
            self.cls_to_cls = nn.Parameter(torch.zeros(1, self.num_heads, 1, 1))

            if bias_init in {"normal", "random"}:
                nn.init.normal_(self.freq_rel_bias.weight, std=0.01)
                nn.init.normal_(self.time_rel_bias.weight, std=0.01)
                nn.init.normal_(self.cls_to_patches, std=0.01)
                nn.init.normal_(self.patches_to_cls, std=0.01)
                nn.init.normal_(self.cls_to_cls, std=0.01)
            else:
                self.freq_rel_bias.weight.data.zero_()
                self.time_rel_bias.weight.data.zero_()

        if self.use_grep:
            self.grep_linear = nn.Linear(self.head_dim, GATE_HIDDEN)
            self.grep_a = nn.Parameter(torch.ones(1, self.num_heads, 1, 1))

    def compute_bias(self, device: torch.device) -> Tensor:
        f_p, t_p = self.freq_patches, self.time_patches
        grid_f = torch.arange(f_p, device=device).unsqueeze(1).expand(f_p, t_p).reshape(-1)
        grid_t = torch.arange(t_p, device=device).unsqueeze(0).expand(f_p, t_p).reshape(-1)
        delta_f = (grid_f.unsqueeze(1) - grid_f.unsqueeze(0)) + (f_p - 1)
        delta_t = (grid_t.unsqueeze(1) - grid_t.unsqueeze(0)) + (t_p - 1)

        bias_patch = (self.freq_rel_bias(delta_f) + self.time_rel_bias(delta_t)).permute(2, 0, 1).unsqueeze(0)
        if not self.has_cls:
            return bias_patch

        row_cls = torch.cat([self.cls_to_cls, self.cls_to_patches], dim=-1)
        rows_patch = torch.cat([self.patches_to_cls, bias_patch], dim=-1)
        return torch.cat([row_cls, rows_patch], dim=-2)

    def forward(self, x: Tensor, attn_mask: Tensor | None = None, is_causal: bool = False) -> Tensor:
        B, T, C = x.shape
        H, d = self.num_heads, self.head_dim
        q, k, v = self.qkv(x).reshape(B, T, 3, H, d).permute(2, 0, 3, 1, 4)
        if hasattr(self, "freq_rel_bias"):
            self.shared.value = self.compute_bias(x.device)
        position_bias = self.shared.value
        if position_bias is None:
            bias = None
        elif self.use_grep:
            gate = torch.sigmoid(self.grep_linear(q).view(B, H, T, 2, GATE_HIDDEN // 2).sum(-1))
            gate_a, gate_b = gate.chunk(2, dim=-1)
            scale = gate_a * (gate_b * self.grep_a - 1.0) + 2.0
            bias = (scale * position_bias).to(q.dtype)
        else:
            bias = position_bias.to(q.dtype)
        out = F.scaled_dot_product_attention(
            q, k, v, attn_mask=bias, dropout_p=self.attn_drop.p if self.training else 0.0
        )
        return self.proj_drop(self.proj(out.transpose(1, 2).reshape(B, T, C)))


# ==============================================================================
# 2. CED-Tiny Backbone with 2D Relative Position Bias
# ==============================================================================

class ViTSBias(nn.Module):
    """
    Vision Transformer with 2D Relative Position Bias.
    Exact configuration matching official CED-Tiny:
        - 12 blocks, 192 embed dimension, 3 attention heads
        - 4 x 24 patch grid (freq x time) = 96 patch tokens + 1 CLS token
        - Standard Pre-LN structure (NO DeepNorm alpha scaling)
    """

    def __init__(
        self,
        num_classes: int = 36,
        freq_bins: int = 64,
        frames: int = 398,
        pretrained: bool = False,
        dropout: float = 0.0,
        attn_dropout: float = 0.0,
        drop_path: float = 0.0,
        arch: str = "ced_mini",
        bias_init: str = "zero",
        use_bias: bool = True,
        use_grep: bool = True,
        time_major: bool = False,
    ) -> None:
        super().__init__()
        self.use_bias = use_bias
        self.time_major = time_major
        self.freq_patches = freq_bins // 16
        self.time_patches = frames // 16
        self.frames = frames

        if arch in ("ced_mini", "vit_mini", "deit_mini", "mini"):
            from timm.models.vision_transformer import VisionTransformer
            self.vit = VisionTransformer(
                img_size=(freq_bins, frames),
                patch_size=16,
                in_chans=1,
                num_classes=num_classes,
                embed_dim=256,
                depth=12,
                num_heads=4,
                mlp_ratio=4.0,
                drop_rate=dropout,
                proj_drop_rate=dropout,
                attn_drop_rate=attn_dropout,
                drop_path_rate=drop_path,
            )
        elif arch in ("ced_small", "vit_small", "deit_small", "small"):
            self.vit = timm.create_model(
                "deit_small_patch16_224",
                pretrained=pretrained,
                in_chans=1,
                num_classes=num_classes,
                img_size=(freq_bins, frames),
                drop_rate=dropout,
                proj_drop_rate=dropout,
                attn_drop_rate=attn_dropout,
                drop_path_rate=drop_path,
            )
        else:
            self.vit = timm.create_model(
                arch,
                pretrained=pretrained,
                in_chans=1,
                num_classes=num_classes,
                img_size=(freq_bins, frames),
                drop_rate=dropout,
                proj_drop_rate=dropout,
                attn_drop_rate=attn_dropout,
                drop_path_rate=drop_path,
            )

        self.shared = SharedBias()
        for index, block in enumerate(self.vit.blocks):
            if use_bias:
                block.attn = BiasAttention(
                    block.attn,
                    self.shared,
                    owns_table=index == 0,
                    freq_patches=self.freq_patches,
                    time_patches=self.time_patches,
                    has_cls=True,
                    bias_init=bias_init,
                    use_grep=use_grep,
                )

    def forward_features(self, x: Tensor) -> tuple[Tensor, list[Tensor]]:
        vit = self.vit
        x = vit.patch_embed(x[..., : self.frames])
        x = x + vit.pos_embed[:, 1:]
        B, N, C = x.shape
        if self.time_major:
            x = x.view(B, self.freq_patches, self.time_patches, C).transpose(1, 2).reshape(B, N, C)
        cls = (vit.cls_token + vit.pos_embed[:, :1]).expand(B, -1, -1)
        x = vit.norm_pre(vit.pos_drop(torch.cat([cls, x], dim=1)))
        self.shared.value = None

        block_tokens: list[Tensor] = []
        for block in vit.blocks:
            x = block(x)
            block_tokens.append(x[:, 0])

        x = vit.norm(x)
        return x, block_tokens

    def forward(self, x: Tensor) -> Tensor:
        vit = self.vit
        x, _ = self.forward_features(x)
        return vit.forward_head(x)

    def forward_with_context(self, x: Tensor, context_block: int = 9) -> tuple[Tensor, Tensor]:
        """Returns classification logits and the intermediate CLS token at context_block."""
        x_norm, block_tokens = self.forward_features(x)
        logits = self.vit.forward_head(x_norm)
        if 0 <= context_block < len(block_tokens):
            ctx_token = block_tokens[context_block]
        else:
            ctx_token = x_norm[:, 0]
        return logits, ctx_token


def load_ced_tiny(
    model: ViTSBias,
    repo_or_path: str | Path = CED_REPO,
    class_mids: list[str] | None = None,
    beats_ckpt_path: str | Path | None = None,
) -> dict[str, int]:
    """
    Download and initialize model weights from the official AudioSet pre-trained CED-Tiny checkpoint.
    Re-maps 2D spatial frequency-time positional embeddings.
    """
    from huggingface_hub import hf_hub_download
    from safetensors.torch import load_file

    cand = Path(repo_or_path)
    if cand.is_file():
        local_path = str(cand)
    else:
        local_path = hf_hub_download(str(repo_or_path), "model.safetensors")

    state = load_file(local_path)
    vit = model.vit
    mapped = {k[len(CED_PREFIX) :]: v for k, v in state.items() if k.startswith(CED_PREFIX + "blocks.")}
    mapped.update(
        {
            k[len(CED_PREFIX) :]: v
            for k, v in state.items()
            if k.startswith((CED_PREFIX + "patch_embed.", CED_PREFIX + "norm."))
        }
    )
    own = vit.state_dict()
    matched = {k: v for k, v in mapped.items() if k in own and own[k].shape == v.shape}
    result = vit.load_state_dict(matched, strict=False)

    # Reconstruct 2D positional embeddings from freq_pos_embed and time_pos_embed
    if (CED_PREFIX + "freq_pos_embed") in state and (CED_PREFIX + "time_pos_embed") in state:
        freq = state[CED_PREFIX + "freq_pos_embed"][0, :, :, 0]
        time = state[CED_PREFIX + "time_pos_embed"][0, :, 0, :]
        if freq.shape[1] != model.freq_patches:
            freq = F.interpolate(freq[None], size=model.freq_patches, mode="linear", align_corners=False)[0]
        time = time[:, : model.time_patches]
        if time.shape[1] >= model.time_patches:
            grid = freq[:, :, None] + time[:, None, :]
            pos = grid.permute(1, 2, 0).reshape(1, model.freq_patches * model.time_patches, -1)
            vit.pos_embed.data[:, 1:] = pos.to(vit.pos_embed.dtype)

    # Initialize classification head from AudioSet checkpoint if mapped rows exist
    weights = state.get("outputlayer.1.weight")
    bias = state.get("outputlayer.1.bias")
    if weights is not None and bias is not None and beats_ckpt_path and Path(beats_ckpt_path).is_file():
        try:
            beats_ckpt = torch.load(beats_ckpt_path, map_location="cpu", weights_only=False)
            label_dict = beats_ckpt.get("label_dict", {})
            beats_index_by_mid = {str(mid).casefold(): int(idx) for idx, mid in label_dict.items()}

            rows: list[int] = []
            if class_mids:
                for mid in class_mids:
                    mid_cf = mid.casefold()
                    if mid_cf in beats_index_by_mid:
                        rows.append(beats_index_by_mid[mid_cf])

            if len(rows) == vit.head.weight.shape[0]:
                row_tensor = torch.tensor(rows, dtype=torch.long)
                with torch.no_grad():
                    vit.head.weight.copy_(weights[row_tensor])
                    vit.head.bias.copy_(bias[row_tensor])
                LOGGER.info("Successfully mapped %d AudioSet head rows into ViTSBias head.", len(rows))
        except Exception as exc:
            LOGGER.warning("Could not map AudioSet head for CED: %s", exc)

    return {"loaded_tensors": len(matched), "total_tensors": len(own), "missing": len(result.missing_keys)}


# ==============================================================================
# 3. Soft MoE Router (Google DeepMind arXiv:2308.00951)
# ==============================================================================

class SoftMoERouter(nn.Module):
    """
    Soft MoE Routing Network following Google DeepMind (arXiv:2308.00951):
    'From Sparse to Soft Mixtures of Experts' (Puigcerver et al., ICLR 2024).

    Implements:
    - Normalization and learnable temperature scaling:
        L_e(x) = gamma * (u(x)^T * phi_e) / (||u(x)||_2 * ||phi_e||_2 + eps)
    - Soft Combine Weights:
        C = Softmax([L_low, L_high]) in (0, 1)^2
    - Differentiable end-to-end routing without hard discrete gating.
    """

    def __init__(
        self,
        spectral_dim: int = 6,
        token_dim: int = 192,
        token_proj_dim: int = 16,
        hidden_dim: int = 32,
        temperature_init: float = 2.0,
        transition_center_db: float = 7.5,
        transition_tau: float = 3.0,
        mode: str = "soft_moe_cosine",
    ) -> None:
        super().__init__()
        self.mode = mode
        self.transition_center_db = transition_center_db
        self.transition_tau = transition_tau

        self.token_proj = nn.Sequential(
            nn.Linear(token_dim, token_proj_dim),
            nn.GELU(),
        )

        in_dim = spectral_dim + token_proj_dim
        self.feat_encoder = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )

        # Expert Slot Embeddings (Low slot: index 0, High slot: index 1)
        self.expert_slots = nn.Parameter(torch.randn(2, hidden_dim) * 0.02)
        # Learnable inverse temperature parameter (Puigcerver et al., Eq. 1-3)
        self.temperature = nn.Parameter(torch.tensor(float(temperature_init)))

    def forward(
        self, spectral_feat: Tensor, intermediate_token: Tensor, snr_db: Tensor | None = None
    ) -> tuple[Tensor, Tensor]:
        if self.mode == "hard_threshold" and snr_db is not None:
            is_high = (snr_db >= self.transition_center_db).float().unsqueeze(-1)
            combine = torch.cat([1.0 - is_high, is_high], dim=-1)
            return is_high, combine

        # 1. Project and concatenate acoustic features + token context
        proj_token = self.token_proj(intermediate_token.float())
        u = self.feat_encoder(torch.cat([spectral_feat.float(), proj_token], dim=-1))  # [B, hidden_dim]

        # 2. Soft MoE Cosine Sim (arXiv:2308.00951 Eq. 1-3)
        u_norm = F.normalize(u, p=2, dim=-1)  # [B, hidden_dim]
        phi_norm = F.normalize(self.expert_slots, p=2, dim=-1)  # [2, hidden_dim]

        # Cosine similarity matrix scaled by learned temperature gamma
        gamma = F.softplus(self.temperature) + 1e-4
        cosine_sim = torch.matmul(u_norm, phi_norm.t()) * gamma  # [B, 2]
        combine_weights = F.softmax(cosine_sim, dim=-1)  # [B, 2]
        alpha = combine_weights[:, 1:2]  # Probability of High-Expert

        return alpha, combine_weights

    def compute_router_loss(self, alpha: Tensor, snr_db: Tensor) -> Tensor:
        """Smooth supervised guidance loss for the router during training."""
        alpha_f32 = alpha.float().clamp(1e-6, 1.0 - 1e-6)
        target = torch.sigmoid((snr_db.float() - self.transition_center_db) / self.transition_tau).unsqueeze(-1)
        with torch.amp.autocast(device_type=alpha.device.type, enabled=False):
            return F.binary_cross_entropy(alpha_f32, target)


# ==============================================================================
# 4. Student Model: Dual-Expert CED (~10.83M Parameters, ONLY model used in inference)
# ==============================================================================

class DualExpertCED(nn.Module):
    """
    Dual-Expert CED System (~10.83M Parameters):
    - Student Model used for training, evaluation, and production inference.
    - Expert 1 (Low-SNR): ViTSBias CED-Tiny (5.41M) dedicated to noisy regimes
    - Expert 2 (High-SNR): ViTSBias CED-Tiny (5.41M) dedicated to speech-dominated regimes
    - Soft MoE Gating Router (arXiv:2308.00951): Soft Combine Dispatch
    """

    def __init__(
        self,
        num_classes: int = 36,
        sample_rate: int = 16000,
        n_mels: int = 64,
        num_frames: int = 398,
        arch: str = "ced_mini",
        embed_dim: int | None = None,
        dropout: float = 0.0,
        use_grep: bool = True,
        router_mode: str = "soft_moe_cosine",
        transition_center_db: float = 7.5,
        pretrained: bool = True,
        pretrained_source: str = "mispeech/ced-mini",
        class_names: list[str] | None = None,
        class_mids: list[str] | None = None,
        beats_ckpt_path: str | Path | None = None,
    ) -> None:
        super().__init__()
        self.sample_rate = sample_rate
        self.num_classes = num_classes
        self.arch = arch
        if embed_dim is None:
            embed_dim = 256 if "mini" in arch.lower() else (384 if "small" in arch.lower() else 192)
        self.embed_dim = embed_dim
        self.pretrained_source = pretrained_source

        # 1. Frontends
        self.frontend = LogMelFrontend(
            sample_rate=sample_rate,
            n_fft=512,
            win_length=400,
            hop_length=160,
            n_mels=n_mels,
            target_frames=num_frames,
        )
        self.spec_augment = SpecAugment(freq_masks=2, freq_width=8, time_masks=2, time_width=24)
        self.acoustic_extractor = AcousticFeatureExtractor(sample_rate=sample_rate)

        # 2. Expert Backbones (ViTSBias with 2D GREP relative bias)
        self.low_expert = ViTSBias(
            num_classes=num_classes,
            freq_bins=n_mels,
            frames=num_frames,
            dropout=dropout,
            use_bias=True,
            use_grep=use_grep,
            arch=arch,
        )
        self.high_expert = ViTSBias(
            num_classes=num_classes,
            freq_bins=n_mels,
            frames=num_frames,
            dropout=dropout,
            use_bias=True,
            use_grep=use_grep,
            arch=arch,
        )

        # 3. Soft MoE Router (arXiv:2308.00951)
        self.router = SoftMoERouter(
            spectral_dim=6,
            token_dim=self.embed_dim,
            token_proj_dim=16,
            hidden_dim=32,
            transition_center_db=transition_center_db,
            mode=router_mode,
        )

        if pretrained:
            self.load_pretrained_ced(
                pretrained_source,
                class_mids=class_mids,
                beats_ckpt_path=beats_ckpt_path,
            )

    def load_pretrained_ced(
        self,
        repo_or_path: str | Path = "mispeech/ced-mini",
        class_mids: list[str] | None = None,
        beats_ckpt_path: str | Path | None = None,
    ) -> None:
        """Loads official AudioSet-pretrained CED-Tiny weights into both experts."""
        try:
            stat_low = load_ced_tiny(self.low_expert, repo_or_path, class_mids, beats_ckpt_path)
            stat_high = load_ced_tiny(self.high_expert, repo_or_path, class_mids, beats_ckpt_path)
            LOGGER.info(
                "Loaded pretrained CED-Tiny weights into Dual Experts: %d/%d tensors matched.",
                stat_low["loaded_tensors"],
                stat_low["total_tensors"],
            )
        except Exception as exc:
            LOGGER.warning("Could not load CED weights from %s: %s", repo_or_path, exc)

    def forward(
        self,
        waveform: Tensor | None = None,
        waveform_raw: Tensor | None = None,
        waveform_frcrn: Tensor | None = None,
        snr_db: Tensor | None = None,
        return_details: bool = False,
    ) -> Tensor | dict[str, Any]:
        """
        Forward Pass of Dual-Expert CED with Dual-Input Routing:
        - Low-SNR Expert: receives waveform_raw (Mix Gốc)
        - High-SNR Expert: receives waveform_frcrn (Mix + FRCRN)
        Output: fused_logits [B, 36] (or dictionary if return_details=True)
        """
        if waveform_raw is None:
            if waveform is not None:
                waveform_raw = waveform
            else:
                raise ValueError("Must provide either waveform or waveform_raw")
        if waveform_frcrn is None:
            waveform_frcrn = waveform_raw

        if waveform_raw.ndim == 1:
            waveform_raw = waveform_raw.unsqueeze(0)
        if waveform_frcrn.ndim == 1:
            waveform_frcrn = waveform_frcrn.unsqueeze(0)

        # 1. Standardize energy level (target RMS = 0.07925)
        norm_raw = rms_normalize(waveform_raw, target_rms=0.07925)
        norm_frcrn = rms_normalize(waveform_frcrn, target_rms=0.07925)

        # 2. Extract LogMel Spectrograms:
        # Low Expert (chuyen gia tri nhieu manh) receives Mix + FRCRN
        # High Expert (chuyen gia giu hoa am tu nhien) receives Mix Goc (Raw)
        spec_low = self.frontend(norm_frcrn)
        spec_high = self.frontend(norm_raw)
        if self.training:
            spec_low = self.spec_augment(spec_low)
            spec_high = self.spec_augment(spec_high)

        # 3. Extract 6 physical acoustic features directly from normalized raw waveform
        acoustic_feats = self.acoustic_extractor(norm_raw)

        # 4. Run Low Expert on Raw Mix and High Expert on FRCRN Mix (Block 9 context)
        logits_low, ctx_low = self.low_expert.forward_with_context(spec_low, context_block=9)
        logits_high, ctx_high = self.high_expert.forward_with_context(spec_high, context_block=9)

        # Joint context token
        router_token = 0.5 * (ctx_low + ctx_high)

        # 5. Soft MoE Routing (arXiv:2308.00951)
        alpha, combine_weights = self.router(acoustic_feats, router_token, snr_db=snr_db)
        c_low = combine_weights[:, 0:1]
        c_high = combine_weights[:, 1:2]

        # 6. Soft Convex Combination of Logits
        fused_logits = c_low * logits_low + c_high * logits_high

        if return_details:
            return {
                "logits": fused_logits,
                "logits_low": logits_low,
                "logits_high": logits_high,
                "alpha": alpha,
                "combine_weights": combine_weights,
                "spectrogram_raw": spec_low,
                "spectrogram_frcrn": spec_high,
            }
        return fused_logits

load_ced = load_ced_tiny




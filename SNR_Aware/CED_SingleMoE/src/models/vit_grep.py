"""
Vision Transformer Backbone with 2D Gated Relative Position Bias (GREP Attention).

Implements:
1. Patch Projection: Conv2d 16x16, stride 16 (192 channels) -> 4 x 24 = 96 patches.
2. 2D Positional Embedding + CLS Token -> Sequence length 97.
3. Pre-LN Transformer Blocks:
   - Branch 1: LayerNorm1 -> BiasAttention -> Residual 1
   - Branch 2: LayerNorm2 -> MLP 192 -> 768 -> 192 (GELU) -> Residual 2
4. 2D Gated Relative Position Bias (GREP):
   - Shared relative distance bucket table (Block 0 owns, Blocks 1-11 reuse via SharedBias).
   - Dynamic Gating: Query Q passes through Linear(64 -> 8) + Sigmoid -> g_a, g_b.
   - Amplitude Modulation: S = g_a * (g_b * a - 1) + 2.
   - Modulated Bias: B_attn = S * P_rel added to QK^T / sqrt(64) before Softmax.

References:
    - CED: Dinkel et al., "CED: Consistent Ensemble Distillation for Audio Tagging",
      ICASSP 2024. arXiv:2308.11957.
    - BEATs: Chen et al., "BEATs: Audio Pre-Training with Acoustic Tokenizers",
      ICML 2023. arXiv:2212.09058.
    - DeiT: Touvron et al., "Training data-efficient image transformers & distillation
      through attention", ICML 2021. arXiv:2012.12877.
"""

from __future__ import annotations

import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

GATE_HIDDEN = 8
CED_REPO = "mispeech/ced-tiny"
CED_PREFIX = "encoder."


class SharedBias:
    """Shared container for relative position bias matrix computed once per forward pass."""
    def __init__(self) -> None:
        self.value: Tensor | None = None


class BiasAttention(nn.Module):
    """
    2D Gated Relative Position Bias Attention (GREP Attention).
    3 attention heads, head_dim = 64 (total embed_dim = 192).

    Mathematical Formulation:
        1. Query Q, Key K, Value V: [B, 3, 97, 64]
        2. Shared Static 2D Relative Bias P_rel:
           Constructed from 2D delta grid: delta_f in [-3, 3], delta_t in [-23, 23].
           Block 0 owns the embedding tables; subsequent blocks reuse the shared tensor.
        3. Dynamic GREP Gate:
           gate = sigmoid(Linear_64->8(Q)) -> split into g_a, g_b in R^[B, 3, 97, 4]
           scale S = g_a * (g_b * a - 1.0) + 2.0
        4. Modulated Bias Matrix:
           B_attn = S * P_rel
        5. Scaled Dot-Product Attention:
           Attention(Q, K, V) = Softmax((Q K^T) / sqrt(64) + B_attn) V
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

        # GREP dynamic gating linear layer (64 -> 8)
        self.grep_linear = nn.Linear(self.head_dim, GATE_HIDDEN)
        # Learnable gating scale parameter
        self.grep_a = nn.Parameter(torch.ones(1, self.num_heads, 1, 1))

    def compute_bias(self, device: torch.device) -> Tensor:
        """
        Compute static 2D relative position bias matrix P_rel of shape [1, 3, 97, 97].
        """
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
        # QKV projection: [B, T, 3, H, d] -> [3, B, H, T, d]
        q, k, v = self.qkv(x).reshape(B, T, 3, H, d).permute(2, 0, 3, 1, 4)

        if hasattr(self, "freq_rel_bias"):
            self.shared.value = self.compute_bias(x.device)
        position_bias = self.shared.value

        # Gated Relative Position Bias computation:
        # q: [B, H, T, d] -> grep_linear(q): [B, H, T, 8]
        gate = torch.sigmoid(self.grep_linear(q).view(B, H, T, 2, GATE_HIDDEN // 2).sum(-1))
        gate_a, gate_b = gate.chunk(2, dim=-1)  # Each: [B, H, T, 1] or [B, H, T, 2]
        scale = gate_a * (gate_b * self.grep_a - 1.0) + 2.0
        bias = (scale * position_bias).to(q.dtype) if position_bias is not None else None

        # Scaled Dot-Product Attention with modulated relative bias
        out = F.scaled_dot_product_attention(
            q, k, v, attn_mask=bias, dropout_p=self.attn_drop.p if self.training else 0.0
        )
        return self.proj_drop(self.proj(out.transpose(1, 2).reshape(B, T, C)))


class ViTSBias(nn.Module):
    """
    Vision Transformer Backbone with 2D Relative Position Bias matching CED-Tiny:
    - 12 Transformer blocks with Pre-LN structure
    - Embedding dimension: 192 (MLP dimension: 768, 295,872 parameters per MLP)
    - 3 Attention Heads (head_dim: 64)
    - Patch grid: 4 frequency patches x 24 time patches = 96 patch tokens (+1 CLS = 97 tokens)
    - Patch projection: Conv2d(1, 192, kernel_size=16, stride=16)
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
        arch: str = "deit_tiny_patch16_224",
        bias_init: str = "zero",
        use_bias: bool = True,
        time_major: bool = False,
        patch_stride: tuple[int, int] = (16, 16),
    ) -> None:
        super().__init__()
        self.use_bias = use_bias
        self.patch_stride = tuple(patch_stride)
        self.time_major = time_major
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
        # Patch 16 x 16 với bước (stride_f, stride_t): bước < 16 là patch chồng lấn (AST, Gong et al. 2021: 16 x 16
        # bước 10; Xiao et al., NeurIPS 2021: stem chồng lấn giúp ViT học ổn định hơn). Bước 16 = CED gốc (4 x 24).
        stride_f, stride_t = self.patch_stride
        self.freq_patches = (freq_bins - 16) // stride_f + 1  # bước 16: 4
        self.time_patches = (frames - 16) // stride_t + 1     # bước 16: 24, bước 8: 48
        self.frames = frames
        if self.patch_stride != (16, 16):
            patch_embed = self.vit.patch_embed
            patch_embed.proj.stride = self.patch_stride
            patch_embed.grid_size = (self.freq_patches, self.time_patches)
            patch_embed.num_patches = self.freq_patches * self.time_patches
            embed_dim = self.vit.pos_embed.shape[-1]
            self.vit.pos_embed = nn.Parameter(torch.zeros(1, 1 + patch_embed.num_patches, embed_dim))
            nn.init.trunc_normal_(self.vit.pos_embed, std=0.02)
        self.shared = SharedBias()

        # Replace standard attention with 2D GREP BiasAttention
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
                )

    def forward(self, x: Tensor) -> Tensor:
        vit = self.vit
        # Conv2d patch projection 16x16: [B, 1, 64, 384] -> [B, 192, 4, 24] -> [B, 96, 192]
        x = vit.patch_embed(x[..., : self.frames])
        x = x + vit.pos_embed[:, 1:]
        B, N, C = x.shape
        if self.time_major:
            x = x.view(B, self.freq_patches, self.time_patches, C).transpose(1, 2).reshape(B, N, C)
        # Prepend CLS token -> [B, 97, 192]
        cls = (vit.cls_token + vit.pos_embed[:, :1]).expand(B, -1, -1)
        x = vit.norm_pre(vit.pos_drop(torch.cat([cls, x], dim=1)))
        self.shared.value = None
        # Forward through Transformer blocks 0 to 11
        x = vit.blocks(x)
        x = vit.norm(x)
        return vit.forward_head(x)


def _sample_positions(table: Tensor, positions: Tensor) -> Tensor:
    """Nội suy tuyến tính bảng embedding [C, N] tại các vị trí thực (đơn vị: ô của lưới gốc)."""
    positions = positions.clamp(0, table.shape[1] - 1).to(table.dtype)
    low = positions.floor().long()
    high = (low + 1).clamp(max=table.shape[1] - 1)
    frac = positions - low.to(table.dtype)
    return table[:, low] * (1 - frac) + table[:, high] * frac


def load_ced_tiny(model: ViTSBias, repo: str = CED_REPO, path: str | None = None) -> dict[str, int]:
    """
    Initialize model weights from the official AudioSet pre-trained CED-Tiny checkpoint.
    """
    from huggingface_hub import hf_hub_download
    from safetensors.torch import load_file

    # Không nuốt lỗi tải: trọng số ngẫu nhiên âm thầm làm cả lần chạy vô nghĩa.
    local = path or hf_hub_download(repo, "model.safetensors")
    state = load_file(local)

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
    mapped = {k: v for k, v in mapped.items() if k in own and own[k].shape == v.shape}
    result = vit.load_state_dict(mapped, strict=False)

    if (CED_PREFIX + "freq_pos_embed") in state and (CED_PREFIX + "time_pos_embed") in state:
        freq = state[CED_PREFIX + "freq_pos_embed"][0, :, :, 0]  # [C, F_ced]
        time = state[CED_PREFIX + "time_pos_embed"][0, :, 0, :]  # [C, T_ced], lưới bước 16
        stride_f, stride_t = model.patch_stride
        if model.patch_stride == (16, 16):
            if freq.shape[1] != model.freq_patches:
                freq = F.interpolate(freq[None], size=model.freq_patches, mode="linear", align_corners=False)[0]
            time = time[:, : model.time_patches]
        else:
            # Patch chồng lấn: token thứ i bắt đầu ở i * stride, tức vị trí i * stride / 16 trên lưới CED
            freq = _sample_positions(freq, torch.arange(model.freq_patches) * stride_f / 16)
            time = _sample_positions(time, torch.arange(model.time_patches) * stride_t / 16)
        if time.shape[1] >= model.time_patches:
            grid = freq[:, :, None] + time[:, None, :]
            pos = grid.permute(1, 2, 0).reshape(1, model.freq_patches * model.time_patches, -1)
            vit.pos_embed.data[:, 1:] = pos.to(vit.pos_embed.dtype)

    loaded = sorted(mapped) + (["pos_embed"] if "pos_embed" in own and (CED_PREFIX + "freq_pos_embed") in state else [])
    return {"loaded_tensors": len(mapped), "total_tensors": len(own), "missing": len(result.missing_keys),
            "loaded_keys": loaded}

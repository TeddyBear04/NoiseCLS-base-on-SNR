"""
Single-Backbone CED-Tiny with Split-FFN MoE and Shared Expert (CEDSingleMoESplitFFN).

Architecture Flow:
    1. Waveform n_hat (16kHz, 4.0s = 64,000 samples)
       -> RMS Energy Normalization (target_rms = 0.07925)
    2. Parallel Frontends:
       -> AcousticFeatureExtractor: STFT (257 bins x 401 frames) -> 6 acoustic features f_spec in R^6
       -> LogMel Frontend: (64 Mel bins x 398 frames) -> Instance Normalization -> SpecAugment
    3. Patch Embedding & Early Blocks:
       -> Conv2d(1, 192, 16x16, stride 16) -> 4x24 = 96 patch tokens
       -> Prepend CLS token + 2D Positional Embeddings -> X_0 [B, 97, 192]
       -> Blocks 0 to 9 (Shared Blocks): Pre-LN with 2D GREP BiasAttention
    4. Self-Contained Spectral-Token Router:
       -> CLS token h_9 [B, 192] from Block 9
       -> Nén 192 -> 16 + Concat với f_spec -> R^22 -> MLP 22 -> 32 -> 2 -> Softmax -> alpha in (0, 1)
    5. Deep MoE Blocks 10 & 11 (SplitMoEFFN):
       -> FFN_shared (weight 0.5) + FFN_low (0.5 * (1 - alpha)) + FFN_high (0.5 * alpha)
       -> Identity warm-start initialization (exactly equal to baseline CED-Tiny at step 0)
    6. Dual Classifier Head:
       -> CLS token h_final [B, 192] after Block 11 + LayerNorm
       -> Head_low: Linear(192, 36)
       -> Head_high: Linear(192, 36)
       -> Final Logits: z = (1 - alpha) * z_low + alpha * z_high

References:
    - V-MoE: Riquelme et al., "Scaling Vision with Sparse Mixture of Experts",
      NeurIPS 2021. arXiv:2106.05974.
    - Soft MoE: Puigcerver et al., "From Sparse to Soft Mixtures of Experts",
      ICLR 2024. arXiv:2308.00951.
    - DeepSeekMoE: Dai et al., "DeepSeekMoE: Towards Ultimate Expert Specialization
      in Mixture-of-Experts Language Models", ACL 2024. arXiv:2401.06066.
    - CED: Dinkel et al., "CED: Consistent Ensemble Distillation for Audio Tagging",
      ICASSP 2024. arXiv:2308.11957.
"""

from __future__ import annotations

import copy
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from src.frontend import AcousticFeatureExtractor, LogMel, SpecAugment, rms_normalize
from src.models.vit_grep import BiasAttention, SharedBias, ViTSBias, load_ced_tiny
from src.router import SelfContainedSpectralTokenRouter


class SplitMoEFFN(nn.Module):
    """
    Split-FFN with 1 Shared Expert + 2 Routed Experts (Low-SNR & High-SNR).

    Mathematical Formulation:
        y = 0.5 * FFN_shared(x) + 0.5 * [(1 - alpha) * FFN_low(x) + alpha * FFN_high(x)]

    Total routing weight:
        0.5 + 0.5 * (1 - alpha + alpha) = 1.0 (Strictly normalized convex combination)

    Identity Warm-Start Initialization:
        When ffn_shared, ffn_low, and ffn_high are initialized with exact copies of base_mlp,
        at step 0:
            y = 0.5 * M(x) + 0.5 * [(1 - alpha) * M(x) + alpha * M(x)]
              = 0.5 * M(x) + 0.5 * M(x) = M(x)
        The output is MATHEMATICALLY IDENTICAL to the baseline CED-Tiny MLP for ANY alpha in (0, 1).
    """
    def __init__(self, base_mlp: nn.Module) -> None:
        super().__init__()
        self.ffn_shared = copy.deepcopy(base_mlp)
        self.ffn_low = copy.deepcopy(base_mlp)
        self.ffn_high = copy.deepcopy(base_mlp)

    def forward(self, x: Tensor, alpha: Tensor) -> Tensor:
        """
        Args:
            x: [B, Tokens, C] (e.g. [B, 97, 192])
            alpha: [B, 1] or [B, 1, 1] soft routing weight
        Returns:
            out: [B, Tokens, C]
        """
        if alpha.ndim == 2:
            alpha = alpha.unsqueeze(1)  # [B, 1, 1]

        out_shared = self.ffn_shared(x)
        out_low = self.ffn_low(x)
        out_high = self.ffn_high(x)

        # Normalized convex combination
        routed_out = (1.0 - alpha) * out_low + alpha * out_high
        return 0.5 * out_shared + 0.5 * routed_out


class CEDSingleMoESplitFFN(nn.Module):
    """
    Complete Single-Backbone CED-Tiny MoE Architecture (~6.62M parameters).

    - Blocks 0 to 9: 100% shared standard Transformer blocks with 2D GREP BiasAttention.
    - Blocks 10 & 11: Split-FFN containing 1 Shared Expert + 2 Routed Experts (Low & High).
    - Classifier: Dual-Head (Head_low and Head_high) blended via alpha.
    - Self-Contained Spectral-Token Router: Uses physical features from AcousticFeatureExtractor
      and intermediate CLS token representation from Block 9.
    """
    def __init__(
        self,
        num_classes: int = 36,
        freq_bins: int = 64,
        frames: int = 398,
        split_blocks: tuple[int, ...] = (10, 11),
        dropout: float = 0.0,
        load_pretrained: bool = False,
    ) -> None:
        super().__init__()
        self.split_blocks = split_blocks
        self.num_classes = num_classes

        # 1. Frontend Audio Processors
        self.frontend = LogMel(n_mels=freq_bins, num_frames=frames)
        self.spec_augment = SpecAugment()
        self.acoustic_extractor = AcousticFeatureExtractor()

        # 2. Backbone Base Model (Pre-LN ViT with 2D GREP)
        self.vit_base = ViTSBias(
            num_classes=num_classes,
            freq_bins=freq_bins,
            frames=frames,
            pretrained=False,
            dropout=dropout,
            use_bias=True,
            time_major=False,
        )
        if load_pretrained:
            load_ced_tiny(self.vit_base)

        # 3. Self-Contained Spectral-Token Router
        self.router = SelfContainedSpectralTokenRouter(
            spectral_dim=6,
            token_dim=192,
            token_proj_dim=16,
            hidden_dim=32,
        )

        # 4. Replace MLPs in deep blocks (10 & 11) with SplitMoEFFN
        self.moe_blocks: dict[int, SplitMoEFFN] = {}
        for block_idx in self.split_blocks:
            original_mlp = self.vit_base.vit.blocks[block_idx].mlp
            moe_mlp = SplitMoEFFN(original_mlp)
            self.vit_base.vit.blocks[block_idx].mlp = moe_mlp
            self.moe_blocks[block_idx] = moe_mlp

        # 5. Dual Classifier Heads (initialized from pretrained base head)
        original_head = self.vit_base.vit.head
        self.head_low = copy.deepcopy(original_head)
        self.head_high = copy.deepcopy(original_head)

    def forward(
        self,
        waveform: Tensor,
        return_dict: bool = False,
    ) -> dict[str, Tensor] | Tensor:
        """
        End-to-End Forward Pass:
            Waveform -> Features & Mel -> Blocks 0..9 -> Router alpha -> Blocks 10..11 -> Dual Head
        """
        # Step 1: RMS Energy Normalization (target_rms = 0.07925)
        wave_norm = rms_normalize(waveform)

        # Step 2: Extract 6 physical/spectral features from normalized waveform
        spectral_feat = self.acoustic_extractor(wave_norm)  # [B, 6]

        # Step 3: Compute Log-Mel Spectrogram & SpecAugment
        with torch.autocast(waveform.device.type, enabled=False):
            mel = self.frontend(wave_norm.float())  # [B, 1, 64, 398]
        mel_aug = self.spec_augment(mel)

        # Step 4: Patch Embedding & Pre-LN Initialization
        vit = self.vit_base.vit
        # Conv2d 16x16, stride 16 -> 4 x 24 = 96 patch tokens
        x = vit.patch_embed(mel_aug[..., : self.vit_base.frames])
        x = x + vit.pos_embed[:, 1:]
        B, N, C = x.shape
        cls = (vit.cls_token + vit.pos_embed[:, :1]).expand(B, -1, -1)
        x = vit.norm_pre(vit.pos_drop(torch.cat([cls, x], dim=1)))  # [B, 97, 192]

        self.vit_base.shared.value = None

        # Step 5: Execute Shared Blocks 0 through 9
        for i in range(10):
            x = vit.blocks[i](x)

        # Extract Block 9 CLS token representation for router
        block9_token = x[:, 0]  # [B, 192]

        # Step 6: Self-Contained Router predicts dynamic soft weight alpha in (0, 1)
        alpha, gates = self.router(spectral_feat, block9_token)

        # Step 7: Execute Deep MoE Blocks 10 & 11
        for i in self.split_blocks:
            block = vit.blocks[i]
            # Attention with 2D GREP
            x = x + block.drop_path1(block.ls1(block.attn(block.norm1(x))))
            # MoE Split-FFN modulated by alpha
            moe_mlp = block.mlp
            normed_x = block.norm2(x)
            mlp_out = moe_mlp(normed_x, alpha)
            x = x + block.drop_path2(block.ls2(mlp_out))

        # Step 8: Final LayerNorm and Dual Head Aggregation
        x = vit.norm(x)
        cls_final = x[:, 0]  # [B, 192]

        logits_low = self.head_low(cls_final)   # [B, 36]
        logits_high = self.head_high(cls_final) # [B, 36]

        # Continuous soft fusion: z = (1 - alpha) * z_low + alpha * z_high
        logits_fused = (1.0 - alpha) * logits_low + alpha * logits_high

        if return_dict:
            return {
                "logits": logits_fused,
                "logits_low": logits_low,
                "logits_high": logits_high,
                "alpha": alpha,
                "gates": gates,
                "spectral_feat": spectral_feat,
                "block9_token": block9_token,
            }
        return logits_fused

    def get_num_params(self) -> dict[str, int]:
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        router = sum(p.numel() for p in self.router.parameters())
        return {"total": total, "trainable": trainable, "router": router}

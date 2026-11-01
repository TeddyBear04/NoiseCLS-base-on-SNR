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
       -> Nén 192 -> 16 + Concat với f_spec -> R^22 -> MLP 22 -> 32 -> K -> Softmax -> gates g (K expert: low, [mid,] high)
    5. Deep MoE Blocks 10 & 11 (SplitMoEFFN):
       -> 0.5 * FFN_shared + 0.5 * sum_k g_k * FFN_k
       -> Identity warm-start initialization (exactly equal to baseline CED-Tiny at step 0)
    6. Một head mỗi expert:
       -> CLS token h_final [B, 192] after Block 11 + LayerNorm
       -> Head_k: Linear(192, 36) cho k in (low, [mid,] high)
       -> Final Logits: z = sum_k g_k * z_k

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
    Split-FFN: 1 shared expert + K routed expert theo SNR (ffn_low, [ffn_mid,] ffn_high).

        y = 0.5 * FFN_shared(x) + 0.5 * sum_k g_k * FFN_k(x),   sum_k g_k = 1

    Khởi tạo: mọi FFN là bản sao MLP gốc M, nên y = 0.5 M(x) + 0.5 M(x) = M(x) với mọi g, tức đúng bằng
    CED-Tiny gốc ở bước 0.
    """
    def __init__(self, base_mlp: nn.Module, expert_names: tuple[str, ...] = ("low", "high")) -> None:
        super().__init__()
        self.expert_names = expert_names
        self.ffn_shared = copy.deepcopy(base_mlp)
        for name in expert_names:
            setattr(self, f"ffn_{name}", copy.deepcopy(base_mlp))

    def forward(self, x: Tensor, gates: Tensor) -> Tensor:
        """x: [B, Tokens, C]; gates: [B, K] (tổng bằng 1)."""
        routed = sum(
            gates[:, k, None, None] * getattr(self, f"ffn_{name}")(x) for k, name in enumerate(self.expert_names)
        )
        return 0.5 * self.ffn_shared(x) + 0.5 * routed


class CEDSingleMoESplitFFN(nn.Module):
    """
    Complete Single-Backbone CED-Tiny MoE Architecture (~6.62M parameters).

    - Blocks 0 to 9: 100% shared standard Transformer blocks with 2D GREP BiasAttention.
    - Blocks 10 & 11: Split-FFN containing 1 Shared Expert + K Routed Experts (low, [mid,] high).
    - Classifier: một head mỗi expert, trộn theo gates.
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
        expert_names: tuple[str, ...] = ("low", "high"),
        router_input: str = "rms",
        patch_stride: tuple[int, int] = (16, 16),
    ) -> None:
        super().__init__()
        # "rms": như bản gốc, 6 đặc trưng tính trên n_hat đã chuẩn hoá RMS (log-RMS thành hằng số, mất mức năng lượng).
        # "raw": tính trên n_hat thô (RMS n_hat tương quan -0.83 với SNR), log crest/kurtosis, BatchNorm trước router.
        if router_input not in ("rms", "raw"):
            raise ValueError(f"router_input không hợp lệ: {router_input}")
        self.router_input = router_input
        self.split_blocks = split_blocks
        self.expert_names = tuple(expert_names)
        self.num_classes = num_classes

        # 1. Frontend Audio Processors
        self.frontend = LogMel(n_mels=freq_bins, num_frames=frames)
        self.spec_augment = SpecAugment()
        self.acoustic_extractor = AcousticFeatureExtractor(log_heavy_tails=router_input == "raw")

        # 2. Backbone Base Model (Pre-LN ViT with 2D GREP)
        self.vit_base = ViTSBias(
            num_classes=num_classes,
            freq_bins=freq_bins,
            frames=frames,
            pretrained=False,
            dropout=dropout,
            use_bias=True,
            time_major=False,
            patch_stride=tuple(patch_stride),
        )
        if load_pretrained:
            load_ced_tiny(self.vit_base)

        # 3. Self-Contained Spectral-Token Router
        self.router = SelfContainedSpectralTokenRouter(
            spectral_dim=6,
            token_dim=192,
            token_proj_dim=16,
            hidden_dim=32,
            num_experts=len(self.expert_names),
            feature_norm=router_input == "raw",
        )

        # 4. Replace MLPs in deep blocks (10 & 11) with SplitMoEFFN
        self.moe_blocks: dict[int, SplitMoEFFN] = {}
        for block_idx in self.split_blocks:
            original_mlp = self.vit_base.vit.blocks[block_idx].mlp
            moe_mlp = SplitMoEFFN(original_mlp, self.expert_names)
            self.vit_base.vit.blocks[block_idx].mlp = moe_mlp
            self.moe_blocks[block_idx] = moe_mlp

        # 5. Một head mỗi expert (khởi tạo từ head của mô hình nền)
        original_head = self.vit_base.vit.head
        for name in self.expert_names:
            setattr(self, f"head_{name}", copy.deepcopy(original_head))

    def forward(
        self,
        waveform: Tensor,
        return_dict: bool = False,
        oracle_gates: Tensor | None = None,
        oracle_mask: Tensor | None = None,
    ) -> dict[str, Tensor] | Tensor:
        """
        End-to-End Forward Pass:
            Waveform -> Features & Mel -> Blocks 0..9 -> Router gates -> Blocks 10..11 -> head mỗi expert
        """
        # Step 1: RMS Energy Normalization (target_rms = 0.07925)
        wave_norm = rms_normalize(waveform)

        # Step 2: 6 đặc trưng âm học cho router; "raw" giữ mức năng lượng của n_hat (gợi ý SNR mạnh nhất)
        with torch.autocast(waveform.device.type, enabled=False):
            source = waveform.float() if self.router_input == "raw" else wave_norm.float()
            spectral_feat = self.acoustic_extractor(source)  # [B, 6]

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

        # Step 6: Router nội tại -> gates [B, K]; alpha = gate của expert high
        alpha, gates = self.router(spectral_feat, block9_token)
        # Gate dùng cho expert: mẫu có oracle_mask lấy oracle_gates (khoảng SNR thật, chỉ khi train / chẩn đoán),
        # còn lại lấy gate của router. "gates" trả về luôn là của router (cho loss router).
        gates_used = gates
        if oracle_gates is not None:
            mask = torch.ones_like(gates[:, 0], dtype=torch.bool) if oracle_mask is None else oracle_mask
            gates_used = torch.where(mask[:, None], oracle_gates.to(gates.dtype), gates)

        # Step 7: Execute Deep MoE Blocks 10 & 11
        for i in self.split_blocks:
            block = vit.blocks[i]
            # Attention with 2D GREP
            x = x + block.drop_path1(block.ls1(block.attn(block.norm1(x))))
            # MoE Split-FFN trộn theo gates
            moe_mlp = block.mlp
            normed_x = block.norm2(x)
            mlp_out = moe_mlp(normed_x, gates_used)
            x = x + block.drop_path2(block.ls2(mlp_out))

        # Step 8: Final LayerNorm and Dual Head Aggregation
        x = vit.norm(x)
        cls_final = x[:, 0]  # [B, 192]

        # Một head mỗi expert, trộn theo gates: z = sum_k g_k * z_k
        logits_experts = torch.stack([getattr(self, f"head_{name}")(cls_final) for name in self.expert_names], dim=1)
        logits_fused = (gates_used.unsqueeze(-1) * logits_experts).sum(dim=1)  # [B, 36]

        if return_dict:
            return {
                "logits": logits_fused,
                "logits_experts": logits_experts,  # [B, K, 36]
                "alpha": alpha,
                "gates": gates,
                "gates_used": gates_used,
                "spectral_feat": spectral_feat,
                "block9_token": block9_token,
            }
        return logits_fused

    def get_num_params(self) -> dict[str, int]:
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        router = sum(p.numel() for p in self.router.parameters())
        return {"total": total, "trainable": trainable, "router": router}

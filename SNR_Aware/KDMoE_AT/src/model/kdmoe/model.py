from __future__ import annotations

from pathlib import Path
from typing import Any, NamedTuple

import torch
import torch.nn.functional as F
import yaml
from torch import Tensor, nn

from src.core import BaseModel

from .config import KDMoEConfig
from .experts import Experts
from .frontend import LogMel, SpecAugment
from .gate import MaskGate, NoiseHead
from .hybrid_block import HybridBlock
from .panns import BLOCK_CHANNELS, HintRegressor, PannsCnn10, PannsLogMel, PannsPool, attention_maps, panns_state
from .patch import PatchEmbed
from .pool_gru import FreqPoolGRU
from .router import Router
from .vit import ViTBlock


RMS_EPS = 1e-8
POWER_EPS = 1e-12


def frcrn_snr_db(mixture: Tensor, n_hat: Tensor) -> Tensor:
    """SNR (dB) mỗi clip từ năng lượng FRCRN: 10 log10(P(mixture - n_hat) / P(n_hat)), n_hat ở mức gốc (trước RMS)."""
    noise_power = n_hat.float().square().mean(dim=-1)
    speech_power = (mixture.float() - n_hat.float()).square().mean(dim=-1)
    return 10 * torch.log10((speech_power + POWER_EPS) / (noise_power + POWER_EPS))


class KDMoEOutput(NamedTuple):
    logits: Tensor
    router_logits: Tensor
    e: Tensor
    gate: Tensor
    noise_mel: Tensor | None = None
    noise_target: Tensor | None = None
    at_maps: Tensor | None = None  # Attention Transfer, (B, len(at_blocks), T, F)
    hint: Tensor | None = None  # FitNets, regressor(guided layer), (B, T, F, hint_dim)


class KDMoEModel(BaseModel):
    """Student: encoder (lai cascade ViT, hoặc PANNs CNN10) -> cổng mặt nạ -> router + expert.

    encoder "vit": khối lai x2 -> cổng -> ViT -> gộp tần số -> BiGRU.
    encoder "panns_cnn10": CNN10 pretrained -> cổng trên feature map -> pooling + fc1 của PANNs.
    """

    def __init__(self, config: KDMoEConfig, frontend_config: dict[str, Any] | None = None) -> None:
        super().__init__()
        self.config = config
        self.spec_augment = SpecAugment(
            freq_masks=config.specaug_freq_masks,
            freq_width=config.specaug_freq_width,
            time_masks=config.specaug_time_masks,
            time_width=config.specaug_time_width,
        )
        if config.encoder == "panns_cnn10":
            embed_dim = self._build_panns(config, frontend_config or {})
        else:
            embed_dim = self._build_vit(config, frontend_config or {})
        self.router = Router(embed_dim, config.router_hidden, config.num_groups) if config.route_source == "router" else None
        self.experts = Experts(
            embed_dim, config.expert_hidden, config.num_classes, config.num_groups, dropout=config.dropout
        )

    def _build_vit(self, config: KDMoEConfig, frontend_config: dict[str, Any]) -> int:
        self.frontend = LogMel(**{"num_frames": config.num_frames, **frontend_config})
        self.patch_embed = PatchEmbed(config.d, config.patch_size, config.freq_bins, config.num_frames)
        total_blocks = config.n_hybrid + config.n_vit

        def drop_path(index: int) -> float:
            return config.drop_path_max * index / max(total_blocks - 1, 1)

        self.hybrid = nn.ModuleList(
            HybridBlock(
                config.d,
                config.heads,
                config.hybrid_gru_hidden,
                config.ffn_ratio,
                config.dropout,
                drop_path(index),
                config.hybrid_branches,
            )
            for index in range(config.n_hybrid)
        )
        self.gate = MaskGate(config.d, config.gate_eps, config.gate_bias_init) if config.use_gate else None
        self.noise_head = NoiseHead(config.d, config.patch_size, config.freq_bins, config.num_frames)
        self.vit = nn.ModuleList(
            ViTBlock(config.d, config.heads, config.ffn_ratio, config.dropout, drop_path(config.n_hybrid + index))
            for index in range(config.n_vit)
        )
        self.final_ln = nn.LayerNorm(config.d)
        self.pool_gru = FreqPoolGRU(config.d, config.final_gru_hidden, freq_pool=config.freq_pool)
        return 4 * config.final_gru_hidden

    def _build_panns(self, config: KDMoEConfig, frontend_config: dict[str, Any]) -> int:
        self.frontend = PannsLogMel(frontend_config.get("sample_rate", 16000), config.num_frames)
        self.encoder = PannsCnn10()
        channels = PannsCnn10.out_channels
        self.gate = MaskGate(channels, config.gate_eps, config.gate_bias_init) if config.use_gate else None
        self.noise_head = NoiseHead(channels, config.patch_size, config.freq_bins, config.num_frames)
        self.post_fc = PannsPool(channels)
        if config.hint_block:
            self.hint_regressor = HintRegressor(BLOCK_CHANNELS[config.hint_block - 1], config.hint_dim)
        return channels

    def load_panns_pretrained(self, path: str | Path) -> int:
        """Nạp trọng số CNN10 AudioSet (bn0, conv_block1-4, fc1, melW); trả số tensor đã nạp."""
        state = panns_state(path)
        own = self.state_dict()
        bad = [key for key, value in state.items() if key not in own or own[key].shape != value.shape]
        if bad:
            raise ValueError(f"{path}: PANNs keys do not fit the model: {bad[:5]}")
        self.load_state_dict(state, strict=False)
        return len(state)

    @classmethod
    def from_yaml(cls, path: str | Path) -> KDMoEModel:
        with open(path, encoding="utf-8") as f:
            raw = yaml.safe_load(f)
        return cls(KDMoEConfig.from_dict(raw[KDMoEConfig.section]), raw.get("frontend"))

    def convert_checkpoint(self, checkpoint: dict[str, Any]) -> dict[str, Tensor]:
        return dict(checkpoint.get("model", checkpoint))

    def _gate(self, tokens: Tensor) -> tuple[Tensor, Tensor]:
        if self.gate is not None:
            return self.gate(tokens)
        return tokens, torch.ones_like(tokens)

    def _encode_vit(self, x: Tensor, with_noise: bool) -> tuple[Tensor, Tensor, Tensor | None, None, None]:
        h = self.patch_embed(x)
        for block in self.hybrid:
            h = block(h)
        h, gate = self._gate(h)
        noise_mel = self.noise_head(h) if with_noise else None
        for block in self.vit:
            h = block(h)
        embedding, _ = self.pool_gru(self.final_ln(h))
        return embedding, gate, noise_mel, None, None

    def _encode_panns(
        self, x: Tensor, with_noise: bool
    ) -> tuple[Tensor, Tensor, Tensor | None, Tensor | None, Tensor | None]:
        features, blocks = self.encoder(x, return_blocks=True)
        at_maps = None
        if self.config.at_blocks:
            at_maps = attention_maps([blocks[b - 1] for b in self.config.at_blocks], self.config.at_grid,
                                     self.config.at_time_fraction, self.config.at_freq_fraction)
        hint = None
        if self.config.hint_block:
            hint = self.hint_regressor(blocks[self.config.hint_block - 1], self.config.at_grid,
                                       self.config.at_time_fraction, self.config.at_freq_fraction)
        tokens = features.permute(0, 2, 3, 1)  # (B, mel', T', 512)
        tokens, gate = self._gate(tokens)
        noise_mel = self.noise_head(tokens) if with_noise else None
        embedding = self.post_fc(tokens.permute(0, 3, 1, 2))
        return embedding, gate, noise_mel, at_maps, hint

    def _route_by_frcrn_snr(self, snr_est: Tensor | None, expert_logits: Tensor) -> tuple[Tensor, Tensor]:
        """route_source frcrn_snr: one-hot expert theo khoảng SNR; router_logits 0 / -1e4 chỉ để log router_acc."""
        if snr_est is None:
            raise ValueError("route_source frcrn_snr needs snr_est per clip (frcrn_snr_db of mixture and n_hat)")
        edges = torch.tensor(self.config.snr_band_edges, device=snr_est.device)
        band = torch.bucketize(snr_est.float(), edges, right=True)  # [-5, 5) -> 0, [5, 15) -> 1, [15, 20] -> 2
        weights = F.one_hot(band, self.config.num_groups).to(expert_logits.dtype)
        logits = (weights.unsqueeze(-1) * expert_logits).sum(dim=1)
        return logits, (weights.float() - 1) * 1e4

    def forward(self, waveform: Tensor, noise: Tensor | None = None, snr_est: Tensor | None = None) -> KDMoEOutput:
        if self.config.rms_normalize:  # mỗi clip về cùng RMS; nhiễu sạch (đích L_front) nhân cùng hệ số
            rms = waveform.float().square().mean(dim=-1, keepdim=True).sqrt()
            scale = self.config.rms_normalize / (rms + RMS_EPS)
            waveform = waveform * scale
            noise = noise * scale if noise is not None else None
        x, noise_target = self.frontend(waveform, noise)
        x = self.spec_augment(x)
        encode = self._encode_panns if self.config.encoder == "panns_cnn10" else self._encode_vit
        embedding, gate, noise_mel, at_maps, hint = encode(x, noise_target is not None)
        expert_logits = self.experts(embedding)
        if self.router is None:
            logits, router_logits = self._route_by_frcrn_snr(snr_est, expert_logits)
            return KDMoEOutput(logits, router_logits, embedding, gate, noise_mel, noise_target, at_maps, hint)
        router_logits = self.router(embedding)
        if not self.training and self.config.inference_mode == "top1":
            index = router_logits.argmax(dim=-1)
            logits = expert_logits.gather(1, index[:, None, None].expand(-1, 1, expert_logits.shape[-1])).squeeze(1)
        else:
            weights = router_logits.softmax(dim=-1)
            logits = (weights.unsqueeze(-1) * expert_logits).sum(dim=1)
        return KDMoEOutput(
            logits=logits,
            router_logits=router_logits,
            e=embedding,
            gate=gate,
            noise_mel=noise_mel,
            noise_target=noise_target,
            at_maps=at_maps,
            hint=hint,
        )

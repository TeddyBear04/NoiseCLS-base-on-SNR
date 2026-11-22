from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

from src.core import BaseConfig

HYBRID_BRANCHES = ("both", "attn", "gru")
FREQ_POOL_MODES = ("mean", "attn")
INFERENCE_MODES = ("weighted", "top1")
ENCODERS = ("vit", "panns_cnn10")
ROUTE_BY = ("class", "snr")
ROUTE_SOURCES = ("router", "frcrn_snr")


@dataclass(frozen=True)
class KDMoEConfig(BaseConfig):
    section: ClassVar[str] = "model"

    d: int = 128
    heads: int = 4
    n_hybrid: int = 2
    n_vit: int = 2
    ffn_ratio: int = 4
    hybrid_gru_hidden: int = 64
    final_gru_hidden: int = 64
    expert_hidden: int = 128
    router_hidden: int = 64
    gate_eps: float = 0.05
    gate_bias_init: float = 2.0
    dropout: float = 0.1
    drop_path_max: float = 0.05
    patch_size: int = 16
    freq_bins: int = 128
    num_frames: int = 384
    num_classes: int = 36
    num_groups: int = 4
    use_gate: bool = True
    hybrid_branches: str = "both"
    freq_pool: str = "mean"
    inference_mode: str = "weighted"
    specaug_freq_masks: int = 2
    specaug_freq_width: int = 16
    specaug_time_masks: int = 2
    specaug_time_width: int = 24
    # encoder "panns_cnn10": PANNs CNN10 pretrained (AudioSet, 32 kHz) thay patch/khối lai/ViT/BiGRU
    encoder: str = "vit"
    panns_checkpoint: str = ""
    # route_by "snr": nhóm expert = khoảng SNR, cạnh [5, 15] -> [-5, 5), [5, 15), [15, 20]
    route_by: str = "class"
    snr_band_edges: tuple[float, ...] = (5.0, 15.0)
    # route_source "router": MLP router đoán nhóm từ embedding. "frcrn_snr" (cần route_by snr): không router học,
    # gán cứng 1 expert theo khoảng của SNR ước lượng từ năng lượng FRCRN 10 log10(P(x - n_hat) / P(n_hat)), tính
    # trước rms_normalize (RMS xoá mức n_hat so với mixture). Định tuyến cố định như Hash Layers (NeurIPS 2021),
    # expert theo miền như DEMix (NAACL 2022).
    route_source: str = "router"
    # Attention Transfer (panns_cnn10): bản đồ attention của các block CNN10 (đánh số từ 1), đưa về lưới patch
    # BEATs (24 thời gian x 8 tần số). Student phủ 4.0 s, 50-14000 Hz; BEATs phủ ~3.84 s (24 x 16 frame),
    # 20-8000 Hz -> chỉ lấy phần [0, at_time_fraction] thời gian và [0, at_freq_fraction] trục mel của student.
    at_blocks: tuple[int, ...] = ()
    at_grid: tuple[int, int] = (24, 8)
    at_time_fraction: float = 0.96
    at_freq_fraction: float = 0.845
    # FitNets (panns_cnn10): guided layer = block hint_block của CNN10 (0 = tắt), regressor 1x1 -> hint_dim kênh
    # trên cùng lưới at_grid / at_*_fraction; hint = token BEATs layer 6 (768)
    hint_block: int = 0
    hint_dim: int = 768
    # Chuẩn hoá RMS đầu vào student (n_hat sau FRCRN) về giá trị này trước front-end; 0 = tắt. Nhiễu sạch làm đích
    # L_front nhân cùng hệ số. 0.07925 = trung vị RMS clip nhiễu AudioSet (median_rms, như beat-reviewkd.pdf mục 6.4).
    rms_normalize: float = 0.0

    def __post_init__(self) -> None:
        if self.rms_normalize < 0:
            raise ValueError(f"rms_normalize must be >= 0 (0 = off), got {self.rms_normalize}")
        if self.encoder not in ENCODERS:
            raise ValueError(f"encoder must be one of {ENCODERS}, got {self.encoder!r}")
        if self.route_by not in ROUTE_BY:
            raise ValueError(f"route_by must be one of {ROUTE_BY}, got {self.route_by!r}")
        if self.route_source not in ROUTE_SOURCES:
            raise ValueError(f"route_source must be one of {ROUTE_SOURCES}, got {self.route_source!r}")
        if self.route_source == "frcrn_snr" and self.route_by != "snr":
            raise ValueError(f"route_source frcrn_snr needs route_by snr, got {self.route_by!r}")
        if self.route_by == "snr" and self.num_groups != len(self.snr_band_edges) + 1:
            raise ValueError(
                f"route_by snr needs num_groups = len(snr_band_edges) + 1, "
                f"got {self.num_groups} groups for edges {self.snr_band_edges}"
            )
        if self.hint_block and (self.encoder != "panns_cnn10" or not 1 <= self.hint_block <= 4):
            raise ValueError(f"hint_block needs encoder panns_cnn10 and a block in 1..4, got {self.hint_block}")
        if self.hybrid_branches not in HYBRID_BRANCHES:
            raise ValueError(f"hybrid_branches must be one of {HYBRID_BRANCHES}, got {self.hybrid_branches!r}")
        if self.freq_pool not in FREQ_POOL_MODES:
            raise ValueError(f"freq_pool must be one of {FREQ_POOL_MODES}, got {self.freq_pool!r}")
        if self.inference_mode not in INFERENCE_MODES:
            raise ValueError(f"inference_mode must be one of {INFERENCE_MODES}, got {self.inference_mode!r}")
        if self.freq_bins % self.patch_size or self.num_frames % self.patch_size:
            raise ValueError(
                f"freq_bins and num_frames must be divisible by patch_size {self.patch_size}, "
                f"got {self.freq_bins} and {self.num_frames}"
            )

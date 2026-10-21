"""EfficientAT mn10 (Schmid, Koutini, Widmer - ICASSP 2023, "Efficient Large-Scale Audio Tagging via
Transformer-to-CNN Knowledge Distillation"; github.com/fschmid56/EfficientAT) làm encoder cho KD-MoE.

mn10_as: MobileNetV3 width 1.0, 4.88M tham số, mAP AudioSet 47.1 (PANNs CNN10: 38.0, ~5M). Lấy từ code gốc:

  * mel      `models/preprocess.AugmentMelSTFT` như ex_esc50.py (fine-tune ESC-50): 32 kHz, 128 mel, cửa sổ 800,
             bước 320, n_fft 1024, pre-emphasis, log, (x + 4.5) / 5; fmin 0, fmax mặc định (sr/2 - 1000), khi train
             fmin/fmax lấy ngẫu nhiên trong +-10 Hz / 2000 Hz (augment của chính EfficientAT, thay SpecAugment:
             recipe ESC-50 đặt freqm = timem = 0). Waveform 16 kHz được resample lên 32 kHz.
  * features 17 stage conv (stride 32): (B, 1, 128, T) -> (B, 960, 4, T/32). Cắt giữa 384 frame (3.84 s) để
             lưới 4 x 12 dựng lại đúng 128 x 384 log-mel nhiễu ở NoiseHead (L_front, patch 32).
  * head     lớp đầu của head 'mlp' (Linear 960 -> 1280, Hardswish, Dropout) sau global average pooling làm
             embedding cho router + expert của KD-MoE; lớp cuối 527 lớp AudioSet bỏ.

Cổng mặt nạ, NoiseHead, router 3 khoảng SNR + 3 expert giữ nguyên KD-MoE. Repo được clone ở commit cố định
và import cô lập (`external_repo.import_isolated`): gói `models` của EfficientAT trùng tên với gói `models`
của teacher (teacher/models), nên không đưa thẳng repo vào sys.path.
"""

from __future__ import annotations

import contextlib
import io
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
import yaml
from torch import Tensor, nn

from external_repo import import_isolated
from src.model.kdmoe import KDMoEConfig, KDMoEModel
from src.model.kdmoe.gate import MaskGate, NoiseHead
from src.model.kdmoe.panns import HintRegressor, attention_maps

EFFICIENTAT_SAMPLE_RATE = 32_000
NUM_FRAMES = 384


def build_efficientat(student: dict[str, Any]) -> tuple[nn.Module, nn.Module, dict[str, str]]:
    """(network 527 lớp chưa nạp trọng số, AugmentMelSTFT theo ex_esc50.py, pretrained_models của repo)."""
    model_module, preprocess = import_isolated(
        student["repo_dir"], student["repo_url"], student["commit"], "models.mn.model", "models.preprocess")
    with contextlib.redirect_stdout(io.StringIO()):  # get_model in cả mạng ra stdout
        net = model_module.get_model(num_classes=527, pretrained_name=None, width_mult=student["width_mult"],
                                     head_type="mlp")
    mel = preprocess.AugmentMelSTFT(n_mels=128, sr=EFFICIENTAT_SAMPLE_RATE, win_length=800, hopsize=320, n_fft=1024,
                                    freqm=0, timem=0, fmin=0.0, fmax=None, fmin_aug_range=10, fmax_aug_range=2000)
    return net, mel, model_module.pretrained_models


class EfficientATFrontend(nn.Module):
    """Waveform 16 kHz -> (log-mel EfficientAT (B, 1, 128, 384), log-mel nhiễu cùng bộ lọc hoặc None).

    Mixture và nhiễu đi chung một lần gọi mel, nên khi train cùng nhận một fmin/fmax ngẫu nhiên.
    """

    def __init__(self, mel: nn.Module, sample_rate: int = 16_000, num_frames: int = NUM_FRAMES) -> None:
        super().__init__()
        self.mel = mel
        self.sample_rate = sample_rate
        self.num_frames = num_frames

    def _crop(self, x: Tensor) -> Tensor:
        start = max(x.shape[-1] - self.num_frames, 0) // 2
        return x[..., start : start + self.num_frames]

    def forward(self, waveform: Tensor, noise: Tensor | None = None) -> tuple[Tensor, Tensor | None]:
        import torchaudio

        batch = waveform.shape[0]
        audio = waveform if noise is None else torch.cat([waveform, noise], dim=0)
        with torch.autocast(device_type=waveform.device.type, enabled=False):
            audio = torchaudio.functional.resample(audio.float(), self.sample_rate, EFFICIENTAT_SAMPLE_RATE)
            spec = self._crop(self.mel(audio)).unsqueeze(1)                 # (B or 2B, 1, 128, 384)
        return spec[:batch], (spec[batch:] if noise is not None else None)


class MlpPool(nn.Module):
    """Feature map (B, C, F', T') -> embedding: global average pooling + lớp đầu của head 'mlp' pretrained."""

    def __init__(self, linear: nn.Linear, dropout: float) -> None:
        super().__init__()
        self.fc = linear
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor) -> Tensor:
        return self.dropout(F.hardswish(self.fc(x.mean(dim=(2, 3)))))


class EfficientATKDMoE(KDMoEModel):
    """KD-MoE với encoder EfficientAT (mục `student` trong yaml). Đi đường encode CNN của KD-MoE (encoder:
    panns_cnn10): feature map -> cổng -> pooling -> router + expert; front-end/encoder/pooling thay bằng EfficientAT."""

    def __init__(self, config: KDMoEConfig, frontend_config: dict[str, Any] | None = None,
                 student: dict[str, Any] | None = None) -> None:
        if config.encoder != "panns_cnn10":
            raise ValueError(f"EfficientATKDMoE uses the CNN encode path (encoder: panns_cnn10), got {config.encoder!r}")
        self._student = dict(student or {})
        super().__init__(config, frontend_config)

    def _build_panns(self, config: KDMoEConfig, frontend_config: dict[str, Any]) -> int:
        net, mel, self._pretrained_urls = build_efficientat(self._student)
        first, dropout = net.classifier[2], net.classifier[4].p
        channels = first.in_features
        self.frontend = EfficientATFrontend(mel, frontend_config.get("sample_rate", 16000), config.num_frames)
        self.encoder = net.features
        self.gate = MaskGate(channels, config.gate_eps, config.gate_bias_init) if config.use_gate else None
        self.noise_head = NoiseHead(channels, config.patch_size, config.freq_bins, config.num_frames)
        self.post_fc = MlpPool(first, dropout)
        # Tầng trung gian cho KD qua đặc trưng (chỉ số trong net.features, 0..16 với mn10): at_stages (AT),
        # hint_stage (FitNets), review_stages (ReviewKD). Lưới ra (C, F', T'); so với lưới BEATs qua
        # at_grid / at_time_fraction / at_freq_fraction như PANNs.
        self._at_stages = list(self._student.get("at_stages", []))
        self._hint_stage = self._student.get("hint_stage")
        self._review_stages = list(self._student.get("review_stages", []))
        with torch.no_grad():  # số kênh từng tầng
            x, self._stage_channels = torch.zeros(1, 1, config.freq_bins, config.num_frames), []
            for module in self.encoder:
                x = module(x)
                self._stage_channels.append(x.shape[1])
        if self._hint_stage is not None:
            self.hint_regressor = HintRegressor(self._stage_channels[self._hint_stage], config.hint_dim)
        self._stages: list[Tensor] = []
        return first.out_features

    def _encode_panns(self, x: Tensor, with_noise: bool) -> tuple:
        taps = set(self._at_stages) | set(self._review_stages) | ({self._hint_stage} - {None})
        tapped: dict[int, Tensor] = {}
        for index, module in enumerate(self.encoder):
            x = module(x)
            if index in taps:
                tapped[index] = x                                    # (B, C, F', T')
        features = x
        cfg = self.config
        at_maps = hint = None
        if self._at_stages:  # attention_maps / HintRegressor nhận bố cục (B, C, T', F') như block CNN10
            at_maps = attention_maps([tapped[i].transpose(2, 3) for i in self._at_stages], cfg.at_grid,
                                     cfg.at_time_fraction, cfg.at_freq_fraction)
        if self._hint_stage is not None:
            hint = self.hint_regressor(tapped[self._hint_stage].transpose(2, 3), cfg.at_grid, cfg.at_time_fraction,
                                       cfg.at_freq_fraction)
        self._stages = [tapped[i] for i in self._review_stages]
        tokens, gate = self._gate(features.permute(0, 2, 3, 1))     # (B, F', T', C)
        noise_mel = self.noise_head(tokens) if with_noise else None
        return self.post_fc(tokens.permute(0, 3, 1, 2)), gate, noise_mel, at_maps, hint

    def load_efficientat_pretrained(self) -> int:
        """Trọng số AudioSet của biến thể (pretrained_models của repo): features -> encoder, classifier.2 -> post_fc.fc."""
        url = self._pretrained_urls[self._student["variant"]]
        source = torch.hub.load_state_dict_from_url(url, map_location="cpu")
        state: dict[str, Tensor] = {}
        for key, value in source.items():
            if key.startswith("features."):
                state["encoder." + key.removeprefix("features.")] = value
            elif key.startswith("classifier.2."):
                state["post_fc.fc." + key.removeprefix("classifier.2.")] = value
        own = self.state_dict()
        bad = [key for key, value in state.items() if key not in own or own[key].shape != value.shape]
        missing = [key for key in own if key.startswith("encoder.") and key not in state]
        if bad or missing:
            raise ValueError(f"{url}: EfficientAT weights do not fit: bad {bad[:3]}, missing {missing[:3]}")
        self.load_state_dict(state, strict=False)
        return len(state)

    @classmethod
    def from_yaml(cls, path: str | Path) -> EfficientATKDMoE:
        with open(path, encoding="utf-8") as f:
            raw = yaml.safe_load(f)
        return cls(KDMoEConfig.from_dict(raw[KDMoEConfig.section]), raw.get("frontend"), raw["student"])


class EfficientATReviewKDMoE(EfficientATKDMoE):
    """EfficientAT KD-MoE + ReviewModule (reviewkd.py) trên các tầng `student.review_stages`."""

    def __init__(self, config: KDMoEConfig, frontend_config: dict[str, Any] | None = None,
                 student: dict[str, Any] | None = None) -> None:
        from reviewkd import ReviewModule

        super().__init__(config, frontend_config, student)
        if not self._review_stages:
            raise ValueError("EfficientATReviewKDMoE needs student.review_stages")
        self.review = ReviewModule(tuple(self._stage_channels[i] for i in self._review_stages),
                                   freq_fraction=config.at_freq_fraction, time_fraction=config.at_time_fraction)

    def forward(self, waveform: Tensor, noise: Tensor | None = None, snr_est: Tensor | None = None):
        from reviewkd import ReviewKDOutput

        output = super().forward(waveform, noise, snr_est)
        review = self.review(self._stages) if self.training else None
        self._stages = []
        return ReviewKDOutput(**output._asdict(), review=review)

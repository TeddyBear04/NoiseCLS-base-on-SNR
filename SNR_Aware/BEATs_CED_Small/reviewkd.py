"""ReviewKD (Chen, Liu, Zhao, Jia - CVPR 2021, "Distilling Knowledge via Knowledge Review") cho KD-MoE.

Teacher: 5 fold BEATs cross-fit (fine-tune 36 lớp, `teacher_crossfit.py crossfit`), chạy trên n_hat của
chính mẫu đó bằng fold KHÔNG thấy mẫu (out-of-fold); token layer 3, 6, 9, 12 được xuất một lần
(`teacher_crossfit.py features`) thành .npy (rows, 24, 8, 768) fp16 + .ids.json.

Student: KD-MoE PANNs CNN10 pretrained AudioSet (như các run PANNs khác) trên n_hat; 4 block CNN10
(64, 128, 256, 512 kênh, trước dropout) so với layer 3, 6, 9, 12. Block CNN10 có lưới (T', mel') =
(200, 32) (100, 16) (50, 8) (25, 4), không trùng lưới patch BEATs (24 thời gian x 8 tần số, phủ 3.84 s đầu
và 20-8000 Hz). Mỗi bản đồ được cắt về phần [0, 0.96] thời gian x [0, 0.845] trục mel (phần <= 8 kHz của
64 mel 50-14000 Hz), như Attention Transfer / FitNets của thư mục này, rồi đưa về lưới 8 x 24 (`to_teacher_grid`).

ABF và HCL theo code gốc (github.com/dvlab-research/ReviewKD, CIFAR-100/model/reviewkd.py). Điều chỉnh cho
teacher ViT một độ phân giải (beat-reviewkd.pdf): mọi stage so trên lưới 8 x 24, bản đồ trộn thu về lưới đó;
residual cho stage nông hơn là bản đồ 8 x 24 đó, phóng lên bằng nearest như code gốc.

Loss = loss KD-MoE (CE + KD logit + route_ce + cons + front) + review_weight * min(1, epoch / warmup) * sum HCL
(code gốc: kd_loss_weight 1.0, kd_warm_up 20 trên 240 epoch).
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, NamedTuple

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor, nn

from scripts.load_dataset import KDMoEMixDataset
from src.loss import KDMoELossConfig
from src.loss.kdmoe import KDMoELoss
from src.model.kdmoe import KDMoEConfig, KDMoEModel
from src.model.kdmoe.panns import BLOCK_CHANNELS
from src.train import KDMoETrainer
from src.train.base import StepOutput

TEACHER_DIM, GRID = 768, (8, 24)  # BEATs base; lưới (F, T)


def to_teacher_grid(x: Tensor, grid: tuple[int, int], freq_fraction: float, time_fraction: float) -> Tensor:
    """(B, C, F', T') -> (B, C, F, T): phần [0, freq_fraction] x [0, time_fraction] đưa về lưới teacher.

    Khi bản đồ lớn hơn lưới, gộp trung bình (area) trước về cỡ vừa đủ phủ phần cắt để tránh răng cưa, rồi
    lấy mẫu song tuyến tại tâm ô (border như Attention Transfer: điểm ở mép không bị kéo về 0).
    """
    rows, cols = grid
    f, t = x.shape[-2:]
    mid = (min(f, math.ceil(rows / freq_fraction)), min(t, math.ceil(cols / time_fraction)))
    if mid != (f, t):
        x = F.adaptive_avg_pool2d(x, mid)
    fy = (torch.arange(rows, device=x.device) + 0.5) / rows * freq_fraction * 2 - 1
    tx = (torch.arange(cols, device=x.device) + 0.5) / cols * time_fraction * 2 - 1
    mesh_f, mesh_t = torch.meshgrid(fy, tx, indexing="ij")
    sample = torch.stack((mesh_t, mesh_f), dim=-1).to(x.dtype)  # grid_sample: (x = thời gian, y = tần số)
    return F.grid_sample(x, sample.expand(x.shape[0], -1, -1, -1), align_corners=False, padding_mode="border")


# ---------------------------------------------------------------------- ReviewKD


class ABF(nn.Module):
    """Attention-based fusion của code gốc: stage student -> độ rộng teacher, trộn với residual tầng sâu hơn."""

    def __init__(self, in_channel: int, mid_channel: int, out_channel: int, fuse: bool) -> None:
        super().__init__()
        self.conv1 = nn.Sequential(nn.Conv2d(in_channel, mid_channel, 1, bias=False), nn.BatchNorm2d(mid_channel))
        self.conv2 = nn.Sequential(nn.Conv2d(mid_channel, out_channel, 3, padding=1, bias=False),
                                   nn.BatchNorm2d(out_channel))
        self.att_conv = nn.Sequential(nn.Conv2d(mid_channel * 2, 2, 1), nn.Sigmoid()) if fuse else None
        nn.init.kaiming_uniform_(self.conv1[0].weight, a=1)
        nn.init.kaiming_uniform_(self.conv2[0].weight, a=1)

    def forward(self, x: Tensor, residual: Tensor | None, resize) -> tuple[Tensor, Tensor]:
        n, _, h, w = x.shape
        x = self.conv1(x)
        if self.att_conv is not None:
            residual = F.interpolate(residual, (h, w), mode="nearest")
            z = self.att_conv(torch.cat([x, residual], dim=1))
            x = x * z[:, 0].view(n, 1, h, w) + residual * z[:, 1].view(n, 1, h, w)
        x = resize(x)
        return self.conv2(x), x


class ReviewModule(nn.Module):
    """Block student (nông -> sâu, (B, C, F', T')) -> bản đồ (B, 768, 8, 24) mỗi block, dựng từ sâu lên nông."""

    def __init__(self, in_channels: tuple[int, ...] = BLOCK_CHANNELS, out_channel: int = TEACHER_DIM,
                 grid: tuple[int, int] = GRID, freq_fraction: float = 0.845, time_fraction: float = 0.96) -> None:
        super().__init__()
        mid = min(512, in_channels[-1])
        last = len(in_channels) - 1
        self.abfs = nn.ModuleList([ABF(c, mid, out_channel, i < last) for i, c in enumerate(in_channels)][::-1])
        self.grid, self.freq_fraction, self.time_fraction = tuple(grid), freq_fraction, time_fraction

    def _resize(self, x: Tensor) -> Tensor:
        return to_teacher_grid(x, self.grid, self.freq_fraction, self.time_fraction)

    def forward(self, stages: list[Tensor]) -> list[Tensor]:
        results, residual = [], None
        for feature, abf in zip(stages[::-1], self.abfs, strict=True):
            out, residual = abf(feature, residual, self._resize)
            results.insert(0, out)
        return results


def hcl(student: list[Tensor], teacher: list[Tensor]) -> Tensor:
    """Hierarchical context loss, hàm `hcl` của code gốc: MSE đầy đủ + pool 4, 2, 1 trọng số 1/2, 1/4, 1/8,
    chia tổng trọng số; cộng qua các cặp."""
    total = torch.zeros((), device=student[0].device)
    for fs, ft in zip(student, teacher, strict=True):
        fs, ft = fs.float(), ft.float()
        loss = F.mse_loss(fs, ft)
        cnt = tot = 1.0
        for level in (4, 2, 1):
            if level >= fs.shape[2]:
                continue
            cnt /= 2.0
            loss = loss + F.mse_loss(F.adaptive_avg_pool2d(fs, level), F.adaptive_avg_pool2d(ft, level)) * cnt
            tot += cnt
        total = total + loss / tot
    return total


def tokens_to_grid(tokens: Tensor) -> Tensor:
    """Token BEATs theo lưới (B, T, F, C) (chỉ số token t * 8 + f) -> (B, C, F, T), bố cục của CNN."""
    return tokens.permute(0, 3, 2, 1)


# ---------------------------------------------------------------------- KD-MoE wiring


class ReviewKDOutput(NamedTuple):
    logits: Tensor
    router_logits: Tensor
    e: Tensor
    gate: Tensor
    noise_mel: Tensor | None = None
    noise_target: Tensor | None = None
    at_maps: Tensor | None = None
    hint: Tensor | None = None
    review: list[Tensor] | None = None  # 4 x (B, 768, 8, 24), chỉ khi train


class ReviewKDMoE(KDMoEModel):
    """KD-MoE PANNs CNN10 (encoder: panns_cnn10) + ReviewModule trên 4 block CNN10."""

    def __init__(self, config: KDMoEConfig, frontend_config: dict[str, Any] | None = None) -> None:
        super().__init__(config, frontend_config)
        if config.encoder != "panns_cnn10":
            raise ValueError(f"ReviewKDMoE needs encoder panns_cnn10, got {config.encoder!r}")
        self.review = ReviewModule(freq_fraction=config.at_freq_fraction, time_fraction=config.at_time_fraction)
        self._blocks: list[Tensor] = []
        self._review: list[Tensor] | None = None
        self.encoder.register_forward_hook(self._keep_blocks)

    def _keep_blocks(self, module: nn.Module, inputs: Any, output: Any) -> None:
        self._blocks = output[1] if isinstance(output, tuple) else []

    def _encode_panns(self, x: Tensor, with_noise: bool) -> tuple:
        encoded = super()._encode_panns(x, with_noise)
        # block CNN10 (B, C, T', mel') -> (B, C, F', T'), bố cục lưới teacher
        self._review = self.review([b.transpose(2, 3) for b in self._blocks]) if self.training else None
        self._blocks = []
        return encoded

    def forward(self, waveform: Tensor, noise: Tensor | None = None, snr_est: Tensor | None = None) -> ReviewKDOutput:
        output = super().forward(waveform, noise, snr_est)
        return ReviewKDOutput(**output._asdict(), review=self._review)


class ReviewKDLoss(KDMoELoss):
    """Loss KD-MoE + weight * scale * sum HCL; `scale` = min(1, epoch / warmup) do trainer đặt mỗi epoch."""

    def __init__(self, config: KDMoELossConfig, membership: Tensor, weight: float) -> None:
        super().__init__(config, membership)
        self.review_weight = weight
        self.review_scale = 1.0
        self.last_review: float | None = None

    def forward(self, output: Any, batch: dict[str, Any]) -> dict[str, Tensor]:
        terms = super().forward(output, batch)
        self.last_review = None
        review = getattr(output, "review", None)
        if self.review_weight and review is not None and "teacher_review" in batch:
            teacher = batch["teacher_review"]  # (B, 4, 24, 8, 768)
            terms["review"] = hcl(review, [tokens_to_grid(teacher[:, i]) for i in range(teacher.shape[1])])
            terms["total"] = terms["total"] + self.review_weight * self.review_scale * terms["review"]
            self.last_review = terms["review"].item()
        return terms


class ReviewKDTrainer(KDMoETrainer):
    def __init__(self, *args: Any, review_weight: float, warmup_epochs: float, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.criterion = ReviewKDLoss(self.criterion.config, self.criterion.membership, review_weight)
        self.warmup_epochs = warmup_epochs

    def begin_epoch(self) -> None:
        super().begin_epoch()
        self.criterion.review_scale = min(1.0, self._current_epoch / self.warmup_epochs) if self.warmup_epochs else 1.0

    def training_step(self, batch: dict[str, Any]) -> StepOutput:
        step = super().training_step(batch)
        if self.criterion.last_review is not None:
            self._sums["review"] += self.criterion.last_review * step.size
        return step

    def epoch_summary(self) -> dict[str, float]:
        summary = super().epoch_summary()
        if "review" in self._sums:
            summary["review"] = self._sums["review"] / max(self._seen, 1)
            summary["review_scale"] = self.criterion.review_scale
        return summary


class ReviewKDMixDataset(KDMoEMixDataset):
    """KDMoEMixDataset + đặc trưng teacher nhiều layer: item["teacher_review"] (L, 24, 8, 768) fp16."""

    def __init__(self, *args: Any, review_features: list[str | Path], **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._review_paths = [Path(p) for p in review_features]
        self._review_arrays: list[np.ndarray] | None = None
        ids = json.loads(self._review_paths[0].with_suffix(".ids.json").read_text(encoding="utf-8"))
        for path in self._review_paths[1:]:
            if json.loads(path.with_suffix(".ids.json").read_text(encoding="utf-8")) != ids:
                raise ValueError(f"{path}: mix ids differ from {self._review_paths[0]}")
        self._review_row = {sample_id: i for i, sample_id in enumerate(ids)}
        missing = [row["sample_id"] for row in self.rows if row["sample_id"] not in self._review_row]
        if missing:
            raise KeyError(f"{len(missing)} rows have no teacher features, e.g. {missing[0]}")

    def __getitem__(self, index: int) -> dict[str, Any]:
        item = super().__getitem__(index)
        if self._review_arrays is None:  # mở memmap trong từng worker
            self._review_arrays = [np.load(p, mmap_mode="r") for p in self._review_paths]
        row = self._review_row[item["sample_id"]]
        item["teacher_review"] = torch.from_numpy(np.stack([np.asarray(a[row]) for a in self._review_arrays]))
        return item

"""KD theo quan hệ (không phụ thuộc số kênh) cho student EfficientAT mn10 của KD-MoE.

Ba loss, mỗi run chỉ bật một, cộng vào loss KD-MoE hiện có (CE + KD logit + route_ce + front + 0.1 cons):

  spatial smooth_l1  (R1) Pair-wise distillation (Liu et al., CVPR 2019, "Structured Knowledge Distillation for
                     Semantic Segmentation", Eq. 2): nút = vị trí trên lưới, a_ij = cosine giữa đặc trưng nút
                     i, j; khớp ma trận a_ij của student với teacher. Ở đây mỗi clip 192 nút (lưới BEATs 24 x 8),
                     bỏ đường chéo (luôn = 1), SmoothL1 (|a_s - a_t| <= 2, hệ số beta 1 -> gần 0.5 (a_s - a_t)^2
                     như bình phương của paper). Ma trận G = F F^T theo tinh thần Similarity-Preserving KD
                     (Tung & Mori, ICCV 2019) nhưng quan hệ giữa vị trí trong một clip, không giữa các mẫu.
  spatial cka        (R2) 1 - CKA tuyến tính (Kornblith et al., ICML 2019) giữa đặc trưng 192 vị trí của
                     student (C_s kênh) và teacher (768): CKA = <K_s, K_t>_F / (|K_s|_F |K_t|_F), K = X_c X_c^T,
                     X_c trừ trung bình theo vị trí. Bất biến với phép quay trực giao và co giãn đều, nên khớp
                     được hai biểu diễn khác số chiều không cần projector (Dasgupta & Cohn, ICLR 2025, dùng
                     1 - CKA để khớp hidden state của student hẹp hơn teacher).
  sample             (R3) Quan hệ giữa các mẫu trong batch (Similarity-Preserving KD, Tung & Mori, ICCV 2019;
                     RKD, Park et al., CVPR 2019): R = cos(z_i, z_j) của embedding student (1280) và embedding
                     BEATs gộp trung bình (768), SmoothL1 như RKD. Bỏ đường chéo, cặp trùng sample_id và cặp
                     cùng noise_ytid (hai mix của cùng clip nhiễu, đã có L_cons kéo lại gần).

Teacher: 5 fold BEATs cross-fit nghe nhiễu sạch, mỗi mẫu train lấy fold KHÔNG thấy nó
(`teacher_crossfit.py features`): token layer 6 (rows, 24, 8, 768) cho spatial, embedding gộp (rows, 768) cho
sample. Chỉ train; teacher detach. Student: tầng `student.spatial_stage` của mn10 (12: stride 16, 112 kênh,
lưới 8 x 24), phần <= 8 kHz (at_freq_fraction) đưa về lưới teacher bằng `reviewkd.to_teacher_grid`.

Trainer đo thêm chuẩn gradient trên encoder do CE và do loss phụ (đã nhân trọng số) mỗi `grad_probe_every`
bước -> grad_ratio = |grad aux| / |grad CE|, dùng để chọn trọng số (mục tiêu 10-30 %).
"""

from __future__ import annotations

import math

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, ClassVar, NamedTuple

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor, nn

from efficientat import EfficientATKDMoE
from featurekd import GlobalCosineLoss, MGDLoss, ViTKDLoss, qkv_relation_loss, teacher_to_student_layout
from reviewkd import to_teacher_grid
from scripts.load_dataset import KDMoEMixDataset
from src.core import BaseConfig
from src.loss import KDMoELossConfig
from src.loss.kdmoe import KDMoELoss
from src.model.kdmoe import KDMoEConfig
from src.train import KDMoETrainer
from src.train.base import StepOutput

SPATIAL_TYPES = ("smooth_l1", "cka")


@dataclass(frozen=True)
class RelationKDConfig(BaseConfig):
    """Các khoá relation KD trong mục `loss` của yaml student (BaseConfig.from_dict bỏ qua khoá khác)."""

    section: ClassVar[str] = "loss"

    spatial_kd_weight: float = 0.0
    spatial_kd_type: str = "smooth_l1"
    spatial_exclude_diagonal: bool = True
    # R4: trọng số cặp vị trí theo năng lượng nhiễu sạch (mặt nạ attention không gian của FGD)
    spatial_mask: bool = False
    spatial_mask_temperature: float = 0.5
    spatial_mask_floor: float = 0.1
    relation_kd_weight: float = 0.0
    # R6: CRD dạng InfoNCE với bank teacher out-of-fold
    crd_weight: float = 0.0
    crd_dim: int = 256
    crd_temperature: float = 0.07
    crd_negatives: int = 1024
    # crd_level "patch": CRD theo từng token như SNR_Expert_SSLAM (mỗi token student là một anchor, dương = token teacher
    # cùng vùng thời gian-tần số, âm khác nhãn trong batch; CRDLoss chuẩn hoá Z của RepDistiller). crd_proj_dim = chiều g_s / g_t.
    crd_level: str = "pooled"
    crd_proj_dim: int = 128
    crd_n_data: int = 30240  # số hàng train: nền của Pn = 1 / n_data và của Z
    # feature-based KD cho student ViT (featurekd.py): fkd_method mgd | vitkd | qkv | gfd. fkd_weight = alpha (MGD, L_lr của ViTKD) hoặc
    # trọng số loss Q/K/V; fkd_gen_weight = beta (L_gen của ViTKD); relation head của MiniLMv2 = fkd_relation_heads.
    fkd_method: str = ""
    fkd_weight: float = 0.0
    fkd_spkd_weight: float = 0.0  # mgd_spkd: trọng số SPKD (fkd_weight là alpha của MGD)
    fkd_gen_weight: float = 0.0
    fkd_mask_ratio: float = 0.5
    fkd_relation_heads: int = 8
    fkd_norm: str = ""  # MGD: "" thô | "pkd" chuẩn hoá theo kênh (PKD, NeurIPS 2022) | "ln" layer norm target (data2vec, ICML 2022)
    # Cân bằng gradient (theo GradNorm, Chen et al., ICML 2018, dạng đóng giản hoá): > 0 thì loss phụ nhân hệ số aux_scale
    # cập nhật mỗi aux_balance_every bước để |grad_encoder(aux)| = aux_target_ratio * |grad_encoder(CE)|; 0 = tắt (trọng số cố định).
    aux_target_ratio: float = 0.0
    aux_balance_every: int = 10
    aux_balance_ema: float = 0.9  # trung bình trượt của log(aux_scale) giữa các lần đo

    def __post_init__(self) -> None:
        if self.crd_level not in ("pooled", "patch"):
            raise ValueError(f"crd_level must be 'pooled' or 'patch', got {self.crd_level!r}")
        if self.spatial_kd_type not in SPATIAL_TYPES:
            raise ValueError(f"spatial_kd_type must be one of {SPATIAL_TYPES}, got {self.spatial_kd_type!r}")
        if self.fkd_method not in ("", "mgd", "vitkd", "qkv", "spkd", "mgd_spkd", "gfd"):
            raise ValueError(f"fkd_method must be mgd, vitkd, qkv, spkd, mgd_spkd or gfd, got {self.fkd_method!r}")
        if sum(bool(w) for w in (self.spatial_kd_weight, self.relation_kd_weight, self.crd_weight, self.fkd_weight)) > 1:
            raise ValueError("one relation loss per run: set only one of spatial_kd_weight, relation_kd_weight, crd_weight")
        if self.spatial_mask and self.spatial_kd_type != "smooth_l1":
            raise ValueError("spatial_mask weights position pairs, so it needs spatial_kd_type smooth_l1")


# ---------------------------------------------------------------------- losses


def _off_diagonal(n: int, device: torch.device) -> Tensor:
    return ~torch.eye(n, dtype=torch.bool, device=device)


def spatial_similarity_loss(student: Tensor, teacher: Tensor, exclude_diagonal: bool = True,
                            position_weight: Tensor | None = None) -> Tensor:
    """SmoothL1 giữa ma trận cosine vị trí - vị trí; student (B, N, C_s), teacher (B, N, C_t).

    position_weight (B, N), đã detach, trung bình 1: cặp (i, j) nặng w_i * w_j, chuẩn hoá về trung bình 1."""
    if student.shape[:2] != teacher.shape[:2]:
        raise ValueError(f"positions differ: student {tuple(student.shape)} vs teacher {tuple(teacher.shape)}")
    s = F.normalize(student.float(), dim=-1)
    t = F.normalize(teacher.detach().float(), dim=-1)
    g_s, g_t = s @ s.transpose(1, 2), t @ t.transpose(1, 2)           # (B, N, N)
    keep = _off_diagonal(g_s.shape[-1], g_s.device) if exclude_diagonal else torch.ones_like(g_s[0], dtype=torch.bool)
    error = F.smooth_l1_loss(g_s, g_t, reduction="none")[:, keep]       # (B, cặp)
    if position_weight is None:
        return error.mean()
    w = position_weight.detach().float()
    pair = (w[:, :, None] * w[:, None, :])[:, keep]
    return (error * pair / pair.mean(dim=1, keepdim=True)).mean()


def noise_energy_mask(noise_logmel: Tensor, grid: tuple[int, int], freq_fraction: float, time_fraction: float,
                      temperature: float = 0.5, floor: float = 0.1) -> Tensor:
    """Mặt nạ vị trí (B, T * F) từ log-mel nhiễu sạch (B, 1, mel, frame): G = log-mel trung bình mỗi ô lưới teacher,
    A = N softmax(G / T) như mặt nạ attention không gian của FGD (Yang et al., CVPR 2022, T = 0.5), chặn dưới `floor`
    rồi chuẩn hoá trung bình 1. Không gradient."""
    with torch.no_grad():
        cells = to_teacher_grid(noise_logmel.float(), grid, freq_fraction, time_fraction)   # (B, 1, F, T)
        g = cells.permute(0, 3, 2, 1).flatten(1)                                        # (B, T * F), chỉ số t * F + f
        mask = g.shape[1] * (g / temperature).softmax(dim=1)
        mask = mask.clamp_min(floor)
        return mask / mask.mean(dim=1, keepdim=True)


def linear_cka(student: Tensor, teacher: Tensor, eps: float = 1e-8) -> Tensor:
    """CKA tuyến tính theo từng mẫu giữa (B, N, C_s) và (B, N, C_t), N vị trí là "mẫu" của CKA -> (B,)."""
    x = student.float() - student.float().mean(dim=1, keepdim=True)
    y = teacher.detach().float() - teacher.detach().float().mean(dim=1, keepdim=True)
    k, l = x @ x.transpose(1, 2), y @ y.transpose(1, 2)               # Gram (B, N, N)
    hsic = (k * l).sum(dim=(1, 2))
    return hsic / (k.flatten(1).norm(dim=1) * l.flatten(1).norm(dim=1) + eps)


def spatial_cka_loss(student: Tensor, teacher: Tensor) -> Tensor:
    if student.shape[:2] != teacher.shape[:2]:
        raise ValueError(f"positions differ: student {tuple(student.shape)} vs teacher {tuple(teacher.shape)}")
    return (1.0 - linear_cka(student, teacher)).mean()


def _keys(values: list[str] | Tensor, device: torch.device) -> Tensor:
    if isinstance(values, Tensor):
        return values.to(device)
    lookup = {value: index for index, value in enumerate(sorted(set(values)))}
    return torch.tensor([lookup[value] for value in values], device=device)


def relation_mask(sample_ids: list[str], noise_ytids: list[str], device: torch.device) -> Tensor:
    """Cặp (i, j) hợp lệ: khác vị trí, khác sample_id, khác noise_ytid."""
    ids, ytids = _keys(sample_ids, device), _keys(noise_ytids, device)
    return (_off_diagonal(len(ids), device) & (ids[:, None] != ids[None, :])
            & (ytids[:, None] != ytids[None, :]))


def sample_relation_loss(student: Tensor, teacher: Tensor, mask: Tensor,
                         labels: Tensor | None = None) -> tuple[Tensor, dict[str, float]]:
    """SmoothL1 giữa ma trận cosine mẫu - mẫu trên các cặp `mask`; student (B, D_s), teacher (B, D_t)."""
    s = F.normalize(student.float(), dim=-1)
    t = F.normalize(teacher.detach().float(), dim=-1)
    r_s, r_t = s @ s.T, t @ t.T
    pairs = int(mask.sum())
    stats: dict[str, float] = {"rel_pairs": float(pairs)}
    if not pairs:
        return r_s.new_zeros(()), stats
    error = F.smooth_l1_loss(r_s, r_t, reduction="none")
    if labels is not None:
        same = labels[:, None] == labels[None, :]
        for name, part in (("rel_same", mask & same), ("rel_diff", mask & ~same)):
            if part.any():
                stats[name] = error[part].mean().item()
    return error[mask].mean(), stats


class CRDHead(nn.Module):
    """Contrastive Representation Distillation (Tian, Krishnan, Isola, ICLR 2020) dạng InfoNCE với bank teacher cố định.

    embed_s (D_s -> dim) và embed_t (D_t -> dim) đều học như RepDistiller, chuẩn hoá L2; dương = embedding BEATs
    out-of-fold của chính mẫu, âm = `negatives` embedding teacher của mẫu KHÁC lớp lấy từ bank (cũng out-of-fold);
    loss = CE([s·t+, s·t-_1..k] / tau, 0).
    """

    def __init__(self, student_dim: int, teacher_dim: int, dim: int, temperature: float, negatives: int) -> None:
        super().__init__()
        self.embed_s = nn.Linear(student_dim, dim)
        self.embed_t = nn.Linear(teacher_dim, dim)
        self.temperature, self.negatives = temperature, negatives
        self.register_buffer("bank", torch.zeros(0, teacher_dim), persistent=False)
        self.register_buffer("bank_labels", torch.zeros(0, dtype=torch.long), persistent=False)

    def set_bank(self, features: Tensor, labels: Tensor) -> None:
        self.bank = features.float().to(self.embed_t.weight.device)
        self.bank_labels = labels.long().to(self.embed_t.weight.device)

    def sample_negatives(self, labels: Tensor, labels2: Tensor | None = None) -> Tensor:
        """(B, k) chỉ số bank khác lớp: lấy đều rồi bốc lại các ô trùng lớp (xác suất ~1/36 mỗi lần).
        labels2 (mixup): loại cả lớp của clip được trộn vào."""
        index = torch.randint(0, len(self.bank), (labels.shape[0], self.negatives), device=labels.device)
        for _ in range(10):
            clash = self.bank_labels[index] == labels[:, None]
            if labels2 is not None:
                clash = clash | (self.bank_labels[index] == labels2[:, None])
            if not clash.any():
                break
            index[clash] = torch.randint(0, len(self.bank), (int(clash.sum()),), device=labels.device)
        return index

    def forward(self, student: Tensor, teacher: Tensor, labels: Tensor, labels2: Tensor | None = None) -> Tensor:
        """labels2 (mixup): `teacher` đã là embedding teacher trộn tuyến tính (trước chuẩn hoá L2) của hai clip."""
        if not len(self.bank):
            raise RuntimeError("CRDHead.set_bank must run before training")
        s = F.normalize(self.embed_s(student.float()), dim=-1)                          # (B, dim)
        positive = F.normalize(self.embed_t(teacher.detach().float()), dim=-1)          # (B, dim)
        negative = F.normalize(self.embed_t(self.bank[self.sample_negatives(labels, labels2)]), dim=-1)  # (B, k, dim)
        logits = torch.cat([(s * positive).sum(-1, keepdim=True), torch.einsum("bd,bkd->bk", s, negative)], dim=1)
        return F.cross_entropy(logits / self.temperature, torch.zeros_like(labels))


class PatchCRD(nn.Module):
    """CRD theo patch (Tian, Krishnan, Isola, ICLR 2020), port từ SNR_Expert_SSLAM (expert_lib.CRDLoss, Projection,
    align_patch_grid, sample_negatives mode different_label): g_s học, g_t ngẫu nhiên đóng băng; token teacher (T x F) được
    average-pool về lưới student cho mỗi ô cùng vùng thời gian-tần số; mỗi token student là một anchor, 1024 âm lấy từ các
    token teacher khác nhãn trong batch. loss = -(log D1 + sum log D0) / (số anchor) với Z cố định sau batch đầu (RepDistiller)."""

    def __init__(self, student_dim: int, teacher_dim: int, student_grid: tuple[int, int], proj_dim: int,
                 temperature: float, negatives: int, n_data: int, seed: int = 0, eps: float = 1e-7) -> None:
        super().__init__()
        self.g_s = nn.Linear(student_dim, proj_dim)
        generator = torch.Generator().manual_seed(seed)
        self.register_buffer("g_t_weight", torch.randn(proj_dim, teacher_dim, generator=generator) / teacher_dim ** 0.5,
                             persistent=False)
        self.register_buffer("g_t_bias", torch.zeros(proj_dim), persistent=False)
        self.student_grid = tuple(student_grid)
        self.temperature, self.negatives, self.n_data, self.eps = temperature, negatives, n_data, eps
        self.register_buffer("Z", torch.tensor(-1.0), persistent=False)

    def align(self, teacher: Tensor) -> Tensor:
        """(B, T_t, F_t, C) -> (B, T_s * F_s, C): pool token teacher về lưới student, chỉ số t * F + f."""
        batch, _, _, channels = teacher.shape
        grid = F.adaptive_avg_pool2d(teacher.permute(0, 3, 1, 2).float(), self.student_grid)
        return grid.permute(0, 2, 3, 1).reshape(batch, -1, channels)

    def sample_negatives(self, labels: Tensor) -> Tensor:
        index = torch.randint(0, len(labels), (len(labels), self.negatives), device=labels.device)
        for _ in range(10):
            clash = labels[index] == labels[:, None]
            if not clash.any():
                break
            index[clash] = torch.randint(0, len(labels), (int(clash.sum()),), device=labels.device)
        return index

    def forward(self, student: Tensor, teacher: Tensor, labels: Tensor) -> Tensor:
        batch, count, _ = student.shape
        z_s = F.normalize(self.g_s(student.reshape(batch * count, -1).float()), dim=-1)
        t = self.align(teacher.detach()).reshape(batch * count, -1)
        z_t = F.normalize(F.linear(t, self.g_t_weight, self.g_t_bias), dim=-1)
        z_neg = z_t[self.sample_negatives(labels.repeat_interleave(count))]            # (N, m, D)
        pos = torch.exp((z_s * z_t).sum(-1, keepdim=True) / self.temperature)
        neg = torch.exp(torch.bmm(z_neg, z_s.unsqueeze(-1)).squeeze(-1) / self.temperature)
        if self.Z.item() < 0:
            with torch.no_grad():
                self.Z.fill_(torch.cat([pos, neg], dim=1).mean().item() * self.n_data)
        p_pos, p_neg = pos / self.Z, neg / self.Z
        m, pn = neg.shape[1], 1.0 / float(self.n_data)
        log_d1 = torch.log(p_pos / (p_pos + m * pn + self.eps))
        log_d0 = torch.log((m * pn) / (p_neg + m * pn + self.eps))
        return -(log_d1.sum() + log_d0.sum()) / (batch * count)  # chia theo số anchor như expert_lib.CRDLoss


# ---------------------------------------------------------------------- student


class RelationKDOutput(NamedTuple):
    logits: Tensor
    router_logits: Tensor
    e: Tensor
    gate: Tensor
    noise_mel: Tensor | None = None
    noise_target: Tensor | None = None
    at_maps: Tensor | None = None
    hint: Tensor | None = None
    spatial: Tensor | None = None  # (B, T * F, C_s) trên lưới teacher, chỉ khi train
    extra: Any = None  # feature-based KD của student ViT (ced.py): {"last", "shallow", "qkv"}, chỉ khi train


class EfficientATRelationKDMoE(EfficientATKDMoE):
    """EfficientAT KD-MoE + đặc trưng tầng `student.spatial_stage` trên lưới teacher (không thêm tham số)."""

    def __init__(self, config: KDMoEConfig, frontend_config: dict[str, Any] | None = None,
                 student: dict[str, Any] | None = None) -> None:
        super().__init__(config, frontend_config, student)
        time, freq = self._student.get("spatial_grid", (24, 8))
        self._spatial_grid = (int(freq), int(time))                    # to_teacher_grid nhận (F, T)

    def forward(self, waveform: Tensor, noise: Tensor | None = None, snr_est: Tensor | None = None):
        output = super().forward(waveform, noise, snr_est)
        spatial = None
        tokens, self._tokens = self._tokens, None
        if self.training and self._student.get("patch_crd"):   # CRD theo patch: token sau cổng, chỉ số t * F' + f
            spatial = tokens.permute(0, 2, 1, 3).flatten(1, 2)
        elif self.training and self._spatial is not None:
            cfg = self.config
            grid = to_teacher_grid(self._spatial, self._spatial_grid, cfg.at_freq_fraction, cfg.at_time_fraction)
            spatial = grid.permute(0, 3, 2, 1).flatten(1, 2)            # (B, C, F, T) -> (B, T * F, C), chỉ số t * F + f
        self._spatial = None
        return RelationKDOutput(**output._asdict(), spatial=spatial)


# ---------------------------------------------------------------------- loss + trainer


class RelationKDLoss(KDMoELoss):
    """Loss KD-MoE + spatial_kd_weight * spatial (R1/R2, R4 có mặt nạ), relation_kd_weight * relation (R3)
    hoặc crd_weight * crd (R6)."""

    def __init__(self, config: KDMoELossConfig, membership: Tensor, relation: RelationKDConfig,
                 grid: tuple[int, int] = (8, 24), freq_fraction: float = 0.81, time_fraction: float = 1.0,
                 student_dim: int = 1280, patch_grid: tuple[int, int] | None = None,
                 patch_dim: int | None = None, num_shallow: int = 2) -> None:
        super().__init__(config, membership)
        self.relation = relation
        self.grid, self.freq_fraction, self.time_fraction = grid, freq_fraction, time_fraction  # cho mặt nạ R4
        pooled = relation.crd_weight and relation.crd_level == "pooled"
        self.crd = (CRDHead(student_dim, 768, relation.crd_dim, relation.crd_temperature, relation.crd_negatives)
                    if pooled else None)
        patch = relation.crd_weight and relation.crd_level == "patch"
        self.patch_crd = (PatchCRD(patch_dim or student_dim, 768, patch_grid, relation.crd_proj_dim, relation.crd_temperature,
                                   relation.crd_negatives, relation.crd_n_data) if patch else None)
        self.patch_grid = patch_grid
        self.fkd = None
        if relation.fkd_method in ("mgd", "mgd_spkd"):
            self.fkd = MGDLoss(patch_dim or student_dim, relation.fkd_mask_ratio, relation.fkd_norm)
        elif relation.fkd_method == "vitkd":
            self.fkd = ViTKDLoss(patch_dim or student_dim, num_shallow, relation.fkd_mask_ratio)
        elif relation.fkd_method == "gfd":  # một projector mỗi tap: block nông (student.fkd.taps) + block cuối
            self.fkd = GlobalCosineLoss(patch_dim or student_dim, num_shallow + 1)
        self.last_terms: dict[str, Tensor] = {}
        self.last_stats: dict[str, float] = {}
        # aux_target_ratio > 0: trainer đặt aux_scale; 0 trước lần hiệu chỉnh đầu nên loss phụ chưa tác động bước đầu
        self.aux_scale = 0.0 if relation.aux_target_ratio > 0 else 1.0

    def forward(self, output: Any, batch: dict[str, Any]) -> dict[str, Tensor]:
        terms = super().forward(output, batch)
        self.last_stats = {}
        cfg = self.relation
        if cfg.spatial_kd_weight and getattr(output, "spatial", None) is not None and "teacher_hint" in batch:
            teacher = batch["teacher_hint"].flatten(1, 2)                # (B, 24, 8, 768) -> (B, 192, 768)
            if cfg.spatial_kd_type == "cka":
                terms["spatial"] = spatial_cka_loss(output.spatial, teacher)
            else:
                weight = None
                if cfg.spatial_mask:
                    if output.noise_target is None:
                        raise ValueError("spatial_mask needs the clean-noise log-mel (pass noise to the model)")
                    weight = noise_energy_mask(output.noise_target, self.grid, self.freq_fraction, self.time_fraction,
                                               cfg.spatial_mask_temperature, cfg.spatial_mask_floor)
                    self.last_stats["mask_max"] = weight.max(dim=1).values.mean().item()
                    self.last_stats["mask_floor_share"] = (weight <= cfg.spatial_mask_floor * 1.5).float().mean().item()
                terms["spatial"] = spatial_similarity_loss(output.spatial, teacher, cfg.spatial_exclude_diagonal,
                                                           weight)
            terms["aux"] = cfg.spatial_kd_weight * terms["spatial"]
        mix = batch.get("mix")
        if mix is not None and (cfg.spatial_kd_weight or cfg.relation_kd_weight or cfg.fkd_method
                                or self.patch_crd is not None):
            raise ValueError("mixup chỉ hỗ trợ CRD mức clip (pooled) trong nhóm relation KD")
        if self.crd is not None and "teacher_pooled" in batch and mix is not None:
            lam, perm = mix
            teacher = lam * batch["teacher_pooled"] + (1.0 - lam) * batch["teacher_pooled"][perm]
            terms["crd"] = self.crd(output.e, teacher, batch["label"], batch["label"][perm])
        elif self.crd is not None and "teacher_pooled" in batch:
            terms["crd"] = self.crd(output.e, batch["teacher_pooled"], batch["label"])
            terms["aux"] = cfg.crd_weight * terms["crd"]
        if self.patch_crd is not None and getattr(output, "spatial", None) is not None and "teacher_hint" in batch:
            terms["crd"] = self.patch_crd(output.spatial, batch["teacher_hint"], batch["label"])
            terms["aux"] = cfg.crd_weight * terms["crd"]
        extra = getattr(output, "extra", None)
        if cfg.fkd_method and extra is not None:
            terms.update(self._feature_kd(extra, batch))
        if cfg.relation_kd_weight and "teacher_pooled" in batch:
            device = output.e.device
            mask = relation_mask(batch["sample_id"], batch["noise_ytid"], device)
            terms["relation"], self.last_stats = sample_relation_loss(output.e, batch["teacher_pooled"], mask,
                                                                      batch["label"])
            terms["aux"] = cfg.relation_kd_weight * terms["relation"]
        if "aux" in terms:
            if self.relation.aux_target_ratio > 0:
                terms["aux_raw"] = terms["aux"]
                terms["aux"] = self.aux_scale * terms["aux"]
            terms["total"] = terms["total"] + terms["aux"]
        self.last_terms = terms
        return terms


def _teacher_grid(hint: Tensor, grid: tuple[int, int]) -> Tensor:
    """(B, 24, 8, C) token BEATs thô -> (B, C, F', T') pool về lưới student (T', F')."""
    pooled = F.adaptive_avg_pool2d(hint.float().permute(0, 3, 1, 2), tuple(grid))      # (B, C, T', F')
    return pooled.permute(0, 1, 3, 2)


def _feature_kd(self, extra: dict, batch: dict) -> dict[str, Tensor]:
    cfg, terms = self.relation, {}
    if cfg.fkd_method == "mgd":
        terms["mgd"] = self.fkd(extra["last"], _teacher_grid(batch["teacher_hint"], self.patch_grid))
        terms["aux"] = cfg.fkd_weight * terms["mgd"]
    elif cfg.fkd_method == "vitkd":
        review = batch["teacher_review"]                                       # (B, L, T', F', C) đã pool về lưới student
        shallow = [teacher_to_student_layout(review[:, i]) for i in range(review.shape[1])]
        lr, gen = self.fkd(extra["shallow"], extra["last"], shallow, _teacher_grid(batch["teacher_hint"], self.patch_grid))
        terms["vitkd_lr"], terms["vitkd_gen"] = lr, gen
        terms["aux"] = cfg.fkd_weight * lr + cfg.fkd_gen_weight * gen
    elif cfg.fkd_method == "mgd_spkd":  # KD logit đi kèm ở loss chính; ở đây cộng MGD (alpha) và SPKD (lambda) cùng block 12
        grid_t = _teacher_grid(batch["teacher_hint"], self.patch_grid)
        terms["mgd"] = self.fkd(extra["last"], grid_t)
        student = extra["last"].permute(0, 3, 2, 1).flatten(1, 2)
        terms["spkd"] = spatial_similarity_loss(student, batch["teacher_hint"].float().flatten(1, 2), True, None)
        terms["aux"] = cfg.fkd_weight * terms["mgd"] + cfg.fkd_spkd_weight * terms["spkd"]
    elif cfg.fkd_method == "spkd":
        # SPKD / pair-wise distillation (Liu et al., CVPR 2019): nút = token, a_ij = cosine; student block 6 <-> token BEATs L6
        # đã pool về lưới student (cùng 100 nút, t-major), SmoothL1 bỏ đường chéo như R1 của mn10
        student = extra["last"].permute(0, 3, 2, 1).flatten(1, 2)              # (B, C, F', T') -> (B, T' * F', C)
        teacher = batch["teacher_hint"].float().flatten(1, 2)                  # (B, T', F', 768) -> (B, T' * F', 768)
        terms["spkd"] = spatial_similarity_loss(student, teacher, True, None)
        terms["aux"] = cfg.fkd_weight * terms["spkd"]
    elif cfg.fkd_method == "gfd":
        # GFD-Cosine: tap student theo block tăng dần (shallow đã sort + last = block lớn nhất) <-> paths.teacher_review
        # (mỗi lớp teacher một file teacher_hints_l{l}_g1x1.npy, cùng thứ tự tăng dần): (B, L, 1, 1, 768) -> (B, L, 768)
        teacher = batch["teacher_review"].float().flatten(2)
        terms["gfd"] = self.fkd(list(extra["shallow"]) + [extra["last"]], teacher)
        terms["aux"] = cfg.fkd_weight * terms["gfd"]
    elif cfg.fkd_method == "qkv":
        review = batch["teacher_review"]                                       # (B, 3, T', F', 768): Q, K, V
        teacher = tuple(review[:, i].float().flatten(1, 2) for i in range(3))   # t-major (B, T' * F', 768)
        terms["qkv"] = qkv_relation_loss(extra["qkv"], teacher, cfg.fkd_relation_heads)
        terms["aux"] = cfg.fkd_weight * terms["qkv"]
    return terms


RelationKDLoss._feature_kd = _feature_kd


def balanced_aux_scale(scale: float, grad_ce: float, grad_raw: float, ratio: float, ema: float,
                       max_step: float = 10.0) -> float:
    """Hệ số mới cho loss phụ để |grad aux| = ratio * |grad CE| (GradNorm, Chen et al., ICML 2018, dạng đóng).
    Lần đầu (scale <= 0) đặt thẳng; sau đó mục tiêu kẹp trong [scale / max_step, scale * max_step] rồi trung bình trượt
    trong không gian log. Gradient loss phụ bằng 0 hoặc số đo không hữu hạn thì giữ nguyên."""
    if not (math.isfinite(grad_ce) and math.isfinite(grad_raw)) or grad_raw <= 0.0:
        return scale
    target = ratio * grad_ce / grad_raw
    if scale <= 0.0:
        return target
    target = min(max(target, scale / max_step), scale * max_step)
    return math.exp(ema * math.log(scale) + (1.0 - ema) * math.log(target))


class RelationKDTrainer(KDMoETrainer):
    """KDMoETrainer + log loss quan hệ, tỉ lệ gradient aux / CE trên encoder, dừng sớm cho pilot."""

    LOGGED = ("spatial", "relation", "crd", "mgd", "vitkd_lr", "vitkd_gen", "qkv", "spkd", "gfd", "rel_same", "rel_diff", "rel_pairs", "mask_max", "mask_floor_share",
              "grad_ce", "grad_aux", "grad_ratio", "aux_scale", "balance_ratio")

    def __init__(self, *args: Any, relation: RelationKDConfig, grad_probe_every: int = 0,
                 stop_after_epochs: int = 0, teacher_bank: tuple[Tensor, Tensor] | None = None, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        cfg = self.model.config
        grid = getattr(self.model, "_spatial_grid", (8, 24))
        self.criterion = RelationKDLoss(self.criterion.config, self.criterion.membership, relation, grid,
                                        cfg.at_freq_fraction, cfg.at_time_fraction, self.model.embed_dim,
                                        getattr(self.model, "_patch_grid", None),
                                        getattr(self.model, "_patch_dim", self.model.embed_dim),
                                        getattr(self.model, "_num_shallow", 2))
        if self.criterion.patch_crd is not None:
            self.criterion.patch_crd.to(self.device)
        if self.criterion.fkd is not None:
            self.criterion.fkd.to(self.device)
        if self.criterion.crd is not None:
            if teacher_bank is None:
                raise ValueError("crd_weight needs teacher_bank (teacher_pooled of every train row)")
            self.criterion.crd.to(self.device)
            self.criterion.crd.set_bank(*teacher_bank)
        self.grad_probe_every = grad_probe_every
        self.stop_after_epochs = stop_after_epochs
        self._step = 0

    def configure_optimizer(self, steps_per_epoch: int) -> None:
        """Như KDMoETrainer; R6 thêm embed_s/embed_t của CRD (chỉ dùng khi train, không nằm trong checkpoint student)
        vào AdamW cùng lr, rồi dựng lại lịch cosine để nó phủ cả nhóm tham số mới."""
        crd_module = next((m for m in (self.criterion.crd, self.criterion.patch_crd, self.criterion.fkd) if m is not None), None)
        if crd_module is None:
            super().configure_optimizer(steps_per_epoch)
            return
        from src.train.beat_trainer import cosine_with_warmup

        resume, self._resume_state = self._resume_state, None  # resume.pt có cả nhóm CRD: nạp sau khi thêm nhóm
        super().configure_optimizer(steps_per_epoch)
        self.optimizer.add_param_group({"params": list(crd_module.parameters()),  # module mới: lr của head
                                        "lr": self.config.lr * float(getattr(self.model, "head_lr_scale", 1.0)),
                                        "weight_decay": self.config.weight_decay})
        self.scheduler = cosine_with_warmup(self.optimizer, self.config.epochs * steps_per_epoch,
                                            self.config.warmup_ratio, self.config.min_lr / self.config.lr)
        self._resume_state = resume
        self._apply_resume_state()

    def begin_epoch(self) -> None:
        super().begin_epoch()
        self._counts: dict[str, int] = {}

    def _add(self, key: str, value: float, weight: int = 1) -> None:
        self._sums[key] += value * weight
        self._counts[key] = self._counts.get(key, 0) + weight

    def _probe(self, terms: dict[str, Tensor]) -> None:
        params = [p for p in self.model.encoder.parameters() if p.requires_grad]
        norms = {}
        for key, name in (("ce", "grad_ce"), ("aux", "grad_aux")):
            grads = torch.autograd.grad(terms[key], params, retain_graph=True, allow_unused=True)
            norms[name] = torch.sqrt(sum(g.float().pow(2).sum() for g in grads if g is not None)).item()
        for name, value in norms.items():
            self._add(name, value)
        self._add("grad_ratio", norms["grad_aux"] / max(norms["grad_ce"], 1e-12))

    def _rebalance(self, terms: dict[str, Tensor]) -> None:
        """Đo |grad CE| và |grad aux chưa nhân hệ số| trên encoder rồi cập nhật criterion.aux_scale (bước sau mới dùng)."""
        relation = self.criterion.relation
        params = [p for p in self.model.encoder.parameters() if p.requires_grad]
        norms = []
        for key in ("ce", "aux_raw"):
            grads = torch.autograd.grad(terms[key], params, retain_graph=True, allow_unused=True)
            norms.append(torch.sqrt(sum(g.float().pow(2).sum() for g in grads if g is not None)).item())
        used = self.criterion.aux_scale
        self._add("balance_ratio", used * norms[1] / max(norms[0], 1e-12))   # tỉ lệ thật của bước này
        self.criterion.aux_scale = balanced_aux_scale(used, norms[0], norms[1], relation.aux_target_ratio,
                                                      relation.aux_balance_ema)
        self._add("aux_scale", self.criterion.aux_scale)

    def training_step(self, batch: dict[str, Any]) -> StepOutput:
        step = super().training_step(batch)
        terms, self.criterion.last_terms = self.criterion.last_terms, {}
        for key in ("spatial", "relation", "crd", "mgd", "vitkd_lr", "vitkd_gen", "qkv", "spkd", "gfd"):
            if key in terms:
                self._add(key, terms[key].item(), step.size)
        for key, value in self.criterion.last_stats.items():
            self._add(key, value)
        if self.grad_probe_every and "aux" in terms and self._step % self.grad_probe_every == 0:
            self._probe(terms)
        every = self.criterion.relation.aux_balance_every
        if self.stage == 2 and "aux_raw" in terms and every and self._step % every == 0:
            self._rebalance(terms)
        self._step += 1
        return step

    def epoch_summary(self) -> dict[str, float]:
        summary = super().epoch_summary()
        for key in self.LOGGED:
            if self._counts.get(key):
                summary[key] = self._sums[key] / self._counts[key]
        probe = {key: summary[key] for key in ("grad_ce", "grad_aux", "grad_ratio") if key in summary}
        if probe:  # dòng log epoch in 3 chữ số thập phân, tỉ lệ nhỏ bị làm tròn về 0
            self.logger.info("grad probe epoch %d: %s", self._current_epoch,
                             " ".join(f"{key} {value:.4e}" for key, value in probe.items()))
        return summary

    def should_stop(self, epoch: int) -> bool:
        return self.stage == 2 and bool(self.stop_after_epochs) and epoch >= self.stop_after_epochs


# ---------------------------------------------------------------------- data + wiring


class RelationKDMixDataset(KDMoEMixDataset):
    """KDMoEMixDataset + embedding teacher gộp: item["teacher_pooled"] (768,) (R3). Token layer 6 cho R1/R2 đi
    qua `teacher_hints` có sẵn (item["teacher_hint"] (24, 8, 768))."""

    def __init__(self, *args: Any, teacher_pooled: str | Path, **kwargs: Any) -> None:
        import json

        super().__init__(*args, **kwargs)
        path = Path(teacher_pooled)
        ids = json.loads(path.with_suffix(".ids.json").read_text(encoding="utf-8"))
        row = {sample_id: i for i, sample_id in enumerate(ids)}
        missing = [r["sample_id"] for r in self.rows if r["sample_id"] not in row]
        if missing:
            raise KeyError(f"{len(missing)} rows have no pooled teacher embedding, e.g. {missing[0]}")
        pooled = np.load(path)                                         # (rows, 768) fp16, ~46 MB
        self._pooled = {r["sample_id"]: pooled[row[r["sample_id"]]] for r in self.rows}

    def __getitem__(self, index: int) -> dict[str, Any]:
        item = super().__getitem__(index)
        item["teacher_pooled"] = torch.from_numpy(np.asarray(self._pooled[item["sample_id"]], dtype=np.float32))
        return item


def relation_settings(student_config: Path, section: dict[str, Any]) -> RelationKDConfig:
    """RelationKDConfig từ yaml; mục `relationkd` của JSON có thể ghi đè trọng số (pilot chọn trọng số)."""
    relation = RelationKDConfig.from_yaml(student_config)
    keys = ("spatial_kd_weight", "relation_kd_weight", "crd_weight", "crd_level", "crd_n_data", "fkd_method", "fkd_weight", "fkd_spkd_weight",
            "fkd_gen_weight", "fkd_mask_ratio", "fkd_relation_heads", "fkd_norm", "crd_temperature",
            "aux_target_ratio", "aux_balance_every", "aux_balance_ema")
    overrides = {k: section[k] for k in keys if k in section}
    return replace(relation, **overrides) if overrides else relation


def teacher_bank(dataset: Any) -> tuple[Tensor, Tensor]:
    """Bank CRD (R6): embedding BEATs gộp out-of-fold + nhãn của mọi hàng train trong RelationKDMixDataset."""
    while not isinstance(dataset, RelationKDMixDataset):  # Subset của smoke
        dataset = dataset.dataset
    features = np.stack([dataset._pooled[row["sample_id"]] for row in dataset.rows]).astype(np.float32)
    return torch.from_numpy(features), torch.tensor(dataset.labels, dtype=torch.long)

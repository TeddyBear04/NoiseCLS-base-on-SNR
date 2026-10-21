from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, ClassVar

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor, nn

from src.core import BaseConfig

from .base import DistillationLoss, Loss, SupervisedLoss
from .cross_entropy import CrossEntropyLoss

MEMBERSHIP_FILE = Path(__file__).resolve().parents[2] / "config" / "membership.npy"
PROB_EPS = 1e-8


@dataclass(frozen=True)
class KDMoELossConfig(BaseConfig):
    section: ClassVar[str] = "loss"

    label_smoothing: float = 0.1
    kd_temperature: float = 2.0
    kd_weight: float = 1.0
    # "router": KL router vs teacher gom 36 -> nhóm (membership); "logits": KD Hinton 2015 trên 36 logit
    kd_target: str = "router"
    kd_weight_by_snr: bool = False
    route_ce_weight: float = 0.0
    front_weight: float = 1.0
    front_lam_over: float = 2.0
    cons_weight: float = 0.1
    # Attention Transfer (Zagoruyko & Komodakis, ICLR 2017): beta/2 * sum_layers mean((Q_s/|Q_s| - Q_t/|Q_t|)^2),
    # beta = 1e3 như code gốc (szagoruyko/attention-transfer); 0 = tắt
    at_weight: float = 0.0
    # "l2": AT gốc ở trên. "pearson": sum_layers mean_batch(1 - Pearson(Q_s, Q_t)), khoảng cách Pearson của DIST
    # (Huang et al., NeurIPS 2022): bất biến với co giãn và tịnh tiến từng bản đồ, nên chỉ khớp hình dạng. Bản đồ
    # BEATs = hằng số lớn + dao động nhỏ; "l2" chủ yếu phạt độ tương phản và ép student làm phẳng bản đồ của nó.
    # Trọng số: total += at_weight * at (DIST phân loại beta = 2)
    at_loss: str = "l2"
    # FitNets (Romero et al., ICLR 2015) hint: MSE(regressor(guided), hint teacher); huấn luyện chung với CE + KD,
    # beta = 100 như HintLoss của RepDistiller (Tian et al., CRD, ICLR 2020); 0 = tắt
    hint_weight: float = 0.0

    def __post_init__(self) -> None:
        if self.kd_target not in ("router", "logits"):
            raise ValueError(f"kd_target must be 'router' or 'logits', got {self.kd_target!r}")
        if self.at_loss not in ("l2", "pearson"):
            raise ValueError(f"at_loss must be 'l2' or 'pearson', got {self.at_loss!r}")


def load_membership(path: Path = MEMBERSHIP_FILE) -> Tensor:
    if not path.is_file():
        raise FileNotFoundError(f"membership matrix not found, run scripts.build_group_map first: {path}")
    membership = torch.from_numpy(np.load(path)).float()
    if membership.ndim != 2:
        raise ValueError(f"membership must be 2D [classes, groups], got {tuple(membership.shape)}")
    return membership


def teacher_group_probs(teacher_logits: Tensor, membership: Tensor, temperature: float) -> Tensor:
    probs = (teacher_logits.detach().float() / temperature).softmax(dim=-1)
    return probs @ membership.to(probs.device).float()


def router_kl(
    router_logits: Tensor,
    teacher_logits: Tensor,
    membership: Tensor,
    temperature: float,
    weights: Tensor | None = None,
) -> Tensor:
    p_t = teacher_group_probs(teacher_logits, membership, temperature)
    log_p_s = (router_logits.float() / temperature).log_softmax(dim=-1)
    kl = (p_t * (p_t.clamp_min(PROB_EPS).log() - log_p_s)).sum(dim=-1)
    if weights is not None:
        kl = kl * weights.to(kl.device).float()
    return (temperature * temperature) * kl


def attention_transfer(student_maps: Tensor, teacher_maps: Tensor) -> Tensor:
    """Tổng theo layer của mean((Q_s/||Q_s||_2 - Q_t/||Q_t||_2)^2); maps (B, L, T, F)."""
    if student_maps.shape != teacher_maps.shape:
        raise ValueError(f"attention maps differ: student {tuple(student_maps.shape)} vs teacher {tuple(teacher_maps.shape)}")
    s = F.normalize(student_maps.float().flatten(2), dim=-1)
    t = F.normalize(teacher_maps.detach().float().flatten(2), dim=-1)
    return (s - t).pow(2).mean(dim=(0, 2)).sum()


def attention_pearson(student_maps: Tensor, teacher_maps: Tensor) -> Tensor:
    """Tổng theo layer của mean_batch(1 - Pearson(Q_s, Q_t)); maps (B, L, T, F)."""
    if student_maps.shape != teacher_maps.shape:
        raise ValueError(f"attention maps differ: student {tuple(student_maps.shape)} vs teacher {tuple(teacher_maps.shape)}")
    s = student_maps.float().flatten(2)
    t = teacher_maps.detach().float().flatten(2)
    s = F.normalize(s - s.mean(dim=-1, keepdim=True), dim=-1)
    t = F.normalize(t - t.mean(dim=-1, keepdim=True), dim=-1)
    return (1.0 - (s * t).sum(dim=-1)).mean(dim=0).sum()


def logits_kl(
    student_logits: Tensor, teacher_logits: Tensor, temperature: float, weights: Tensor | None = None
) -> Tensor:
    """T^2 * KL(softmax(t/T) || softmax(s/T)) theo từng mẫu (Hinton, Vinyals & Dean, 2015)."""
    p_t = (teacher_logits.detach().float() / temperature).softmax(dim=-1)
    log_p_s = (student_logits.float() / temperature).log_softmax(dim=-1)
    kl = (p_t * (p_t.clamp_min(PROB_EPS).log() - log_p_s)).sum(dim=-1)
    if weights is not None:
        kl = kl * weights.to(kl.device).float()
    return (temperature * temperature) * kl


class RouterKDLoss(DistillationLoss):
    """KL nhiệt độ giữa phân phối nhóm của router và của teacher, gom từ logit 36 lớp."""

    name = "router_kd"

    def __init__(self, membership: Tensor, temperature: float = 2.0) -> None:
        super().__init__()
        if membership.ndim != 2:
            raise ValueError(f"membership must be 2D [classes, groups], got {tuple(membership.shape)}")
        self.register_buffer("membership", membership.detach().clone().float())
        self.temperature = temperature

    def forward(self, student: Tensor, teacher: Tensor, weights: Tensor | None = None) -> Tensor:
        kl = router_kl(student, teacher, self.membership, self.temperature, weights)
        return kl.mean()


class FrontLoss(SupervisedLoss):
    """L1 bất đối xứng trên log-mel nhiễu: phạt phần ước lượng thừa (rò tiếng nói) nặng hơn."""

    name = "front"

    def __init__(self, lam_over: float = 2.0) -> None:
        super().__init__()
        self.lam_over = lam_over

    def forward(self, prediction: Tensor, target: Tensor) -> Tensor:
        diff = prediction.float() - target.float()
        weight = 1.0 + (self.lam_over - 1.0) * (diff > 0).float()
        return (weight * diff.abs()).mean()


class ConsistencyLoss(Loss):
    """1 - cos trên các cặp mix cùng clip nhiễu trong batch, gom theo pair_id."""

    name = "cons"

    def forward(self, embedding: Tensor, pair_ids: Sequence[str] | Tensor) -> Tensor:
        normalized = F.normalize(embedding.float(), dim=-1)
        if isinstance(pair_ids, Tensor):
            keys = pair_ids
        else:
            lookup = {value: index for index, value in enumerate(sorted(set(pair_ids)))}
            keys = torch.tensor([lookup[value] for value in pair_ids])
        losses = []
        for key in keys.unique():
            members = (keys == key).nonzero(as_tuple=True)[0]
            if members.numel() < 2:
                continue
            group = normalized[members]
            similarity = group @ group.T
            rows, cols = torch.triu_indices(members.numel(), members.numel(), offset=1)
            losses.append((1.0 - similarity[rows, cols]).mean())
        if not losses:
            return normalized.new_zeros(())
        return torch.stack(losses).mean()


class KDMoELoss(nn.Module):
    """Tổng loss: CE + KL router + L_front + L_cons (+ CE nhãn nhóm), mọi số hạng ở fp32."""

    name = "kdmoe"

    def __init__(self, config: KDMoELossConfig, membership: Tensor) -> None:
        super().__init__()
        self.config = config
        if membership.ndim != 2:
            raise ValueError(f"membership must be 2D [classes, groups], got {tuple(membership.shape)}")
        self.register_buffer("membership", membership.detach().clone().float())
        self.ce = CrossEntropyLoss(config.label_smoothing)
        self.kd = RouterKDLoss(membership, config.kd_temperature)
        self.front = FrontLoss(config.front_lam_over)
        self.cons = ConsistencyLoss()
        self.snr_accuracy: Mapping[float, float] | None = None

    def _snr_weights(self, snr_db: Tensor) -> Tensor | None:
        if not self.config.kd_weight_by_snr or not self.snr_accuracy:
            return None
        best = max(self.snr_accuracy.values())
        if best <= 0.0:
            return None
        values = torch.as_tensor(
            [self.snr_accuracy.get(round(float(snr), 1), best) for snr in snr_db.tolist()], dtype=torch.float32
        )
        return values / best

    def kd_term(self, output: Any, teacher_logits: Tensor, weights: Tensor | None = None) -> Tensor:
        if self.config.kd_target == "logits":
            return logits_kl(output.logits, teacher_logits, self.config.kd_temperature, weights).mean()
        return self.kd(output.router_logits, teacher_logits, weights)

    def forward(self, output: Any, batch: Mapping[str, Any]) -> dict[str, Tensor]:
        terms: dict[str, Tensor] = {}
        labels = batch["label"]
        terms["ce"] = self.ce(output.logits, labels)
        if self.config.kd_weight:
            weights = self._snr_weights(batch["snr_db"]) if "snr_db" in batch else None
            terms["kd"] = self.kd_term(output, batch["teacher_logits36"], weights)
        if output.noise_mel is not None and output.noise_target is not None:
            terms["front"] = self.front(output.noise_mel, output.noise_target)
        terms["cons"] = self.cons(output.e, batch["noise_ytid"])
        if self.config.at_weight and getattr(output, "at_maps", None) is not None and "teacher_at" in batch:
            distance = attention_pearson if self.config.at_loss == "pearson" else attention_transfer
            terms["at"] = distance(output.at_maps, batch["teacher_at"])
        if self.config.hint_weight and getattr(output, "hint", None) is not None and "teacher_hint" in batch:
            terms["hint"] = F.mse_loss(output.hint.float(), batch["teacher_hint"].float())
        if self.config.route_ce_weight:
            terms["route_ce"] = F.cross_entropy(output.router_logits.float(), batch["group"])
        total = terms["ce"] + self.config.cons_weight * terms["cons"]
        if "kd" in terms:
            total = total + self.config.kd_weight * terms["kd"]
        if "front" in terms:
            total = total + self.config.front_weight * terms["front"]
        if "route_ce" in terms:
            total = total + self.config.route_ce_weight * terms["route_ce"]
        if "at" in terms:
            scale = 1.0 if self.config.at_loss == "pearson" else 0.5
            total = total + scale * self.config.at_weight * terms["at"]
        if "hint" in terms:
            total = total + self.config.hint_weight * terms["hint"]
        terms["total"] = total
        return terms

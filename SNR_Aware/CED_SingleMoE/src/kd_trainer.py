"""
Knowledge Distillation Engine for Single-Backbone MoE (CEDSingleMoESplitFFN).
KD từ teacher BEATs: một teacher (5 fold nhiễu sạch, logit out-of-fold) hoặc hai teacher Fold0/EMA như bản gốc.

References:
    - KD: Hinton et al., "Distilling the Knowledge in a Neural Network", NIPS 2014 Workshop.
    - Consistent Distillation: Beyer et al., "Knowledge Distillation: A Good Teacher is Patient and Consistent", CVPR 2022.
    - CED: Dinkel et al., "CED: Consistent Ensemble Distillation for Audio Tagging", ICASSP 2024.
    - Soft MoE: Puigcerver et al., "From Sparse to Soft Mixtures of Experts", ICLR 2024.
    - DeepSeekMoE: Dai et al., "DeepSeekMoE: Towards Ultimate Expert Specialization", ACL 2024.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import DataLoader

from src.metrics import compute_snr_breakdown
from src.router import router_loss, snr_band_index, snr_band_targets


def apply_mixup_with_teachers(
    waveform: Tensor,
    target: Tensor,
    snr_db: Tensor,
    teacher_logits: Tensor,
    band: Tensor,
    route_band: Tensor,
    alpha: float = 0.4,
    prob: float = 0.8,
    temp: float = 2.0,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
    """Mixup nhất quán trên waveform, nhãn, SNR, khoảng SNR one-hot [B, K] và phân phối mềm của teacher từng
    expert [B, K, C] (Beyer et al., CVPR 2022)."""
    p_teacher = F.softmax(teacher_logits / temp, dim=-1)
    if np.random.rand() > prob or alpha <= 0.0:
        return waveform, target, snr_db, p_teacher, band, route_band

    perm = torch.randperm(waveform.shape[0], device=waveform.device)
    lam = float(np.random.beta(alpha, alpha))

    def mix(x: Tensor) -> Tensor:
        return lam * x + (1.0 - lam) * x[perm]

    return mix(waveform), mix(target), mix(snr_db), mix(p_teacher), mix(band), mix(route_band)


class KDTrainer:
    """
    Knowledge Distillation Trainer (một teacher mỗi expert SNR) for Single-Backbone MoE.
    Integrates fused-logit KD, SNR-weighted expert distillation,
    dynamic router supervision, and cosine annealing schedule.
    """
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        device: torch.device,
        epochs: int = 15,
        lr_backbone: float = 4.0e-5,
        lr_experts: float = 1.2e-4,
        lr_router: float = 2.0e-4,
        weight_decay: float = 0.05,
        warmup_epochs: int = 2,
        mixup_alpha: float = 0.4,
        mixup_prob: float = 0.8,
        label_smoothing: float = 0.05,
        lambda_kd: float = 0.65,
        lambda_route: float = 0.20,
        kd_temperature: float = 2.0,
        kd_fused_weight: float = 0.65,
        kd_expert_weight: float = 0.35,
        snr_edges_db: tuple[float, ...] = (15.0,),
        snr_tau: float = 2.5,
        band_labels: str = "soft",
        route_source: str = "router",
        oracle_prob_start: float = 0.0,
        oracle_prob_end: float = 0.0,
        grad_clip: float = 1.0,
        num_classes: int = 36,
        amp_dtype: str = "bfloat16",
        out_dir: str | Path = "checkpoints",
    ) -> None:
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.epochs = epochs
        self.mixup_alpha = mixup_alpha
        self.mixup_prob = mixup_prob
        self.label_smoothing = label_smoothing
        self.lambda_kd = lambda_kd
        self.lambda_route = lambda_route
        self.temp = kd_temperature
        self.kd_fused_weight = kd_fused_weight
        self.kd_expert_weight = kd_expert_weight
        self.snr_edges_db = tuple(snr_edges_db)
        self.snr_tau = snr_tau
        self.band_labels = band_labels
        if route_source not in ("router", "frcrn_snr"):
            raise ValueError(f"routing.source không hợp lệ: {route_source}")
        self.route_source = route_source
        self.oracle_prob_start = oracle_prob_start
        self.oracle_prob_end = oracle_prob_end
        self.grad_clip = grad_clip
        self.num_classes = num_classes
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)

        self.use_amp = device.type == "cuda"
        self.amp_type = torch.bfloat16 if amp_dtype == "bfloat16" else torch.float16

        # Parameter groups with differential learning rates
        param_groups = self._build_param_groups(lr_backbone, lr_experts, lr_router, weight_decay)
        self.optimizer = torch.optim.AdamW(param_groups)

        total_steps = epochs * len(train_loader) if len(train_loader) > 0 else 1
        warmup_steps = warmup_epochs * len(train_loader) if len(train_loader) > 0 else 1

        def lr_lambda(step: int) -> float:
            if step < warmup_steps:
                return float(step + 1) / float(max(1, warmup_steps))
            progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
            return 0.5 * (1.0 + math.cos(math.pi * progress))

        self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda)
        self.best_macro_acc = 0.0
        self.best_top1_acc = 0.0
        self.best_results: dict[str, float] = {}

    def _build_param_groups(
        self, lr_backbone: float, lr_experts: float, lr_router: float, weight_decay: float
    ) -> list[dict[str, Any]]:
        backbone_params, expert_params, router_params = [], [], []

        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            if "router" in name or "acoustic_extractor" in name:
                router_params.append(param)
            elif "moe" in name or "head_" in name or "ffn_" in name:
                expert_params.append(param)
            else:
                backbone_params.append(param)

        groups = []
        if backbone_params:
            groups.append({"params": backbone_params, "lr": lr_backbone, "weight_decay": weight_decay})
        if expert_params:
            groups.append({"params": expert_params, "lr": lr_experts, "weight_decay": weight_decay})
        if router_params:
            groups.append({"params": router_params, "lr": lr_router, "weight_decay": weight_decay})

        if not groups:
            groups.append({"params": list(self.model.parameters()), "lr": lr_backbone, "weight_decay": weight_decay})
        return groups

    def frcrn_gates(self, batch: dict[str, Any], num_experts: int) -> Tensor:
        """One-hot khoảng SNR của SNR ước lượng từ FRCRN [B, K]."""
        snr_est = batch["snr_est"].to(self.device, non_blocking=True)
        return F.one_hot(snr_band_index(snr_est, self.snr_edges_db), num_experts).float()

    def oracle_probability(self, epoch: int) -> float:
        """Xác suất dùng gate oracle ở epoch này: tuyến tính từ oracle_prob_start (epoch đầu) tới oracle_prob_end (epoch cuối)."""
        frac = epoch / max(1, self.epochs - 1)
        return self.oracle_prob_start + (self.oracle_prob_end - self.oracle_prob_start) * frac

    def train_epoch(self, epoch: int) -> dict[str, float]:
        self.model.train()
        total_loss, total_ce, total_kd, total_route = 0.0, 0.0, 0.0, 0.0
        num_batches = len(self.train_loader)
        if num_batches == 0:
            return {"loss": 0.0, "loss_ce": 0.0, "loss_kd": 0.0, "loss_route": 0.0}

        for batch_idx, batch in enumerate(self.train_loader):
            wave = batch["waveform"].to(self.device, non_blocking=True)
            target = batch["target"].to(self.device, non_blocking=True)
            snr_db = batch["snr_db"].to(self.device, non_blocking=True)
            teacher_log = batch["teacher_logits"].to(self.device, non_blocking=True)  # [B, K, C]
            num_experts = teacher_log.shape[1]
            band = F.one_hot(snr_band_index(snr_db, self.snr_edges_db), num_experts).float()
            route_band = self.frcrn_gates(batch, num_experts) if self.route_source == "frcrn_snr" else band

            wave_m, target_m, snr_m, p_teacher, band_m, route_band_m = apply_mixup_with_teachers(
                wave, target, snr_db, teacher_log, band, route_band,
                alpha=self.mixup_alpha, prob=self.mixup_prob, temp=self.temp,
            )

            # Label smoothing regularizer
            if self.label_smoothing > 0.0:
                target_smooth = target_m * (1.0 - self.label_smoothing) + self.label_smoothing / self.num_classes
            else:
                target_smooth = target_m

            # Nhãn khoảng SNR [B, K]: trọng số expert trong KD, nhãn router và gate oracle.
            # "hard": one-hot khoảng thật (trộn theo mixup); "soft": sigmoid quanh ngưỡng như bản gốc.
            if self.band_labels == "hard":
                w_band = band_m
            else:
                w_band = snr_band_targets(snr_m, self.snr_edges_db, self.snr_tau)
            # Scheduled sampling (Bengio et al., NeurIPS 2015) cho gate: mẫu dùng gate oracle với xác suất
            # eps giảm tuyến tính qua các epoch, còn lại dùng gate của router như lúc suy luận.
            eps = self.oracle_probability(epoch)
            oracle_mask = torch.rand(wave_m.shape[0], device=self.device) < eps if eps > 0.0 else None
            oracle_gates = w_band
            if self.route_source == "frcrn_snr":
                # Cách của FRCRN_KD_MoE: expert route one-hot theo khoảng của SNR ước lượng từ FRCRN, cả train lẫn
                # suy luận; router học không quyết định gì.
                oracle_gates, oracle_mask = route_band_m, torch.ones_like(snr_m, dtype=torch.bool)
            # Đích KD của logit trộn: teacher của mỗi expert theo trọng số khoảng SNR
            p_fused = (w_band.unsqueeze(-1) * p_teacher).sum(dim=1)

            self.optimizer.zero_grad()
            with torch.autocast(self.device.type, dtype=self.amp_type, enabled=self.use_amp):
                out = self.model(wave_m, return_dict=True,
                                 oracle_gates=oracle_gates if oracle_mask is not None else None, oracle_mask=oracle_mask)
                logits_fused = out["logits"]

                # 1. Supervised Task Cross-Entropy
                loss_ce = -(target_smooth * F.log_softmax(logits_fused, dim=-1)).sum(dim=-1).mean()

                # 2. KD trên logit đã trộn (T^2 scaling)
                log_p_fused_kd = F.log_softmax(logits_fused / self.temp, dim=-1)
                loss_kd_fused = -(p_fused * log_p_fused_kd).sum(dim=-1).mean() * (self.temp ** 2)

                # 3. Expert KD: expert k học teacher của nó, trọng số theo khoảng SNR
                log_p_experts = F.log_softmax(out["logits_experts"] / self.temp, dim=-1)  # [B, K, C]
                loss_per_expert = -(p_teacher * log_p_experts).sum(dim=-1) * (self.temp ** 2)  # [B, K]
                loss_kd_experts = (w_band * loss_per_expert).sum(dim=-1).mean()

                loss_kd = self.kd_fused_weight * loss_kd_fused + self.kd_expert_weight * loss_kd_experts

                # 4. Router: CE giữa gates của router và nhãn khoảng SNR (soft, K = 2: đúng BCE(alpha, w) của bản gốc)
                with torch.autocast(self.device.type, enabled=False):
                    loss_route = router_loss(out["gates"], w_band)

                # Total Multi-Task Objective
                loss = (1.0 - self.lambda_kd) * loss_ce + self.lambda_kd * loss_kd + self.lambda_route * loss_route.to(loss_ce.dtype)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.grad_clip)
            self.optimizer.step()
            self.scheduler.step()

            total_loss += float(loss.item())
            total_ce += float(loss_ce.item())
            total_kd += float(loss_kd.item())
            total_route += float(loss_route.item())

        return {
            "loss": total_loss / num_batches,
            "loss_ce": total_ce / num_batches,
            "loss_kd": total_kd / num_batches,
            "loss_route": total_route / num_batches,
        }

    @torch.no_grad()
    def evaluate(self) -> dict[str, Any]:
        self.model.eval()
        all_preds, all_targets, all_snrs, all_gates = [], [], [], []

        for batch in self.val_loader:
            wave = batch["waveform"].to(self.device, non_blocking=True)
            target_idx = batch["target_idx"].numpy()
            snr_val = batch["snr_db"].numpy()

            route = None
            if self.route_source == "frcrn_snr":
                route = self.frcrn_gates(batch, len(self.model.expert_names))
            with torch.autocast(self.device.type, dtype=self.amp_type, enabled=self.use_amp):
                out = self.model(wave, return_dict=True, oracle_gates=route)
                logits = out["logits"]
                gates = out["gates_used"].float()

            probs = F.softmax(logits, dim=-1).cpu().numpy()
            all_preds.append(probs)
            all_targets.append(target_idx)
            all_snrs.append(snr_val)
            all_gates.append(gates.cpu().numpy())

        if not all_preds:
            return {}

        preds = np.concatenate(all_preds, axis=0)
        targets = np.concatenate(all_targets, axis=0)
        snrs = np.concatenate(all_snrs, axis=0)
        gates = np.concatenate(all_gates, axis=0)

        results = compute_snr_breakdown(preds, targets, snrs)
        results["mean_gates"] = gates.mean(axis=0).tolist()
        band = snr_band_index(torch.from_numpy(snrs), self.snr_edges_db).numpy()
        results["router_band_acc"] = float((gates.argmax(axis=1) == band).mean())
        return results

    def save_checkpoint(self, path: Path | str, is_best: bool = False) -> None:
        path = Path(path)
        payload = {
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "best_top1_acc": self.best_top1_acc,
            "best_results": self.best_results,
        }
        torch.save(payload, path)
        if is_best:
            torch.save(payload, self.out_dir / "best_model.pt")

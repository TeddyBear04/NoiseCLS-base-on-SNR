"""
Knowledge Distillation Engine for Single-Backbone MoE (CEDSingleMoESplitFFN).
KD từ teacher BEATs 5 fold fine-tune trên nhiễu sạch (logit out-of-fold); student nghe n_hat.

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


def apply_mixup_with_teacher(
    waveform: Tensor,
    target: Tensor,
    snr_db: Tensor,
    teacher_logits: Tensor,
    alpha: float = 0.4,
    prob: float = 0.8,
    temp: float = 2.0,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Mixup nhất quán trên waveform, nhãn, SNR và phân phối mềm của teacher (Beyer et al., CVPR 2022)."""
    p_teacher = F.softmax(teacher_logits / temp, dim=-1)
    if np.random.rand() > prob or alpha <= 0.0:
        return waveform, target, snr_db, p_teacher

    perm = torch.randperm(waveform.shape[0], device=waveform.device)
    lam = float(np.random.beta(alpha, alpha))
    return (
        lam * waveform + (1.0 - lam) * waveform[perm],
        lam * target + (1.0 - lam) * target[perm],
        lam * snr_db + (1.0 - lam) * snr_db[perm],
        lam * p_teacher + (1.0 - lam) * p_teacher[perm],
    )


class KDTrainer:
    """
    Knowledge Distillation Trainer (teacher BEATs 5 fold nghe nhiễu sạch) for Single-Backbone MoE.
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
        snr_center_db: float = 15.0,
        snr_tau: float = 2.5,
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
        self.snr_center_db = snr_center_db
        self.snr_tau = snr_tau
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
            elif (
                "moe" in name
                or "head_low" in name
                or "head_high" in name
                or "ffn_low" in name
                or "ffn_high" in name
                or "ffn_shared" in name
            ):
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
            teacher_log = batch["teacher_logits"].to(self.device, non_blocking=True)

            wave_m, target_m, snr_m, p_teacher = apply_mixup_with_teacher(
                wave, target, snr_db, teacher_log,
                alpha=self.mixup_alpha, prob=self.mixup_prob, temp=self.temp,
            )

            # Label smoothing regularizer
            if self.label_smoothing > 0.0:
                target_smooth = target_m * (1.0 - self.label_smoothing) + self.label_smoothing / self.num_classes
            else:
                target_smooth = target_m

            # Trọng số SNR liên tục w_snr: SNR thấp -> 0 (expert low), SNR cao -> 1 (expert high); cũng là nhãn router
            w_snr = torch.sigmoid((snr_m - self.snr_center_db) / self.snr_tau).unsqueeze(-1)

            self.optimizer.zero_grad()
            with torch.autocast(self.device.type, dtype=self.amp_type, enabled=self.use_amp):
                out = self.model(wave_m, return_dict=True)

                logits_fused = out["logits"]
                logits_low = out["logits_low"]
                logits_high = out["logits_high"]
                alpha = out["alpha"]

                # 1. Supervised Task Cross-Entropy
                log_probs_fused = F.log_softmax(logits_fused, dim=-1)
                loss_ce = -(target_smooth * log_probs_fused).sum(dim=-1).mean()

                # 2. KD từ teacher nhiễu sạch trên logit đã trộn (T^2 scaling)
                log_p_fused_kd = F.log_softmax(logits_fused / self.temp, dim=-1)
                loss_kd_fused = -(p_teacher * log_p_fused_kd).sum(dim=-1).mean() * (self.temp ** 2)

                # 3. Expert KD: cả hai expert học cùng teacher, trọng số theo SNR
                log_p_low_kd = F.log_softmax(logits_low / self.temp, dim=-1)
                log_p_high_kd = F.log_softmax(logits_high / self.temp, dim=-1)
                loss_low_kd = -(p_teacher * log_p_low_kd).sum(dim=-1) * (self.temp ** 2)
                loss_high_kd = -(p_teacher * log_p_high_kd).sum(dim=-1) * (self.temp ** 2)
                loss_kd_experts = ((1.0 - w_snr.squeeze(-1)) * loss_low_kd + w_snr.squeeze(-1) * loss_high_kd).mean()

                loss_kd = self.kd_fused_weight * loss_kd_fused + self.kd_expert_weight * loss_kd_experts

                # 4. Router Supervision Loss
                with torch.autocast(self.device.type, enabled=False):
                    loss_route = F.binary_cross_entropy(alpha.float().clamp(1e-6, 1.0 - 1e-6), w_snr.float())

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
        all_preds, all_targets, all_snrs, all_alphas = [], [], [], []

        for batch in self.val_loader:
            wave = batch["waveform"].to(self.device, non_blocking=True)
            target_idx = batch["target_idx"].numpy()
            snr_val = batch["snr_db"].numpy()

            with torch.autocast(self.device.type, dtype=self.amp_type, enabled=self.use_amp):
                out = self.model(wave, return_dict=True)
                logits = out["logits"]
                alpha = out["alpha"]

            probs = F.softmax(logits, dim=-1).cpu().numpy()
            all_preds.append(probs)
            all_targets.append(target_idx)
            all_snrs.append(snr_val)
            all_alphas.append(alpha.cpu().numpy())

        if not all_preds:
            return {}

        preds = np.concatenate(all_preds, axis=0)
        targets = np.concatenate(all_targets, axis=0)
        snrs = np.concatenate(all_snrs, axis=0)
        alphas = np.concatenate(all_alphas, axis=0)

        results = compute_snr_breakdown(preds, targets, snrs)
        results["mean_alpha"] = float(np.mean(alphas))
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

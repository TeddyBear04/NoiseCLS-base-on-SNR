"""Suy luận student CEDSingleMoESplitFFN: chỉ n_hat vào, không cần nhãn SNR.

TTA 3 hướng dịch thời gian (0, +50 ms, -50 ms = +-800 mẫu ở 16 kHz), lấy trung bình logit,
softmax với nhiệt độ hiệu chuẩn T_student = 1.40. `ensemble_with_teachers` là phép trộn của bản gốc
với hai teacher Fold0/EMA (cũng nghe n_hat), trọng số teacher theo alpha = gate của expert high.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor

from src.models.split_moe import CEDSingleMoESplitFFN

SHIFTS = (0, 800, -800)


def shift_wave(wave: Tensor, shift: int) -> Tensor:
    if shift > 0:
        return F.pad(wave[:, shift:], (0, shift), mode="reflect")
    if shift < 0:
        return F.pad(wave[:, :shift], (-shift, 0), mode="reflect")
    return wave


class StudentPipeline:
    def __init__(self, model: CEDSingleMoESplitFFN, device: torch.device, temperature: float = 1.40) -> None:
        self.device = device
        self.model = model.to(device).eval()
        self.temperature = temperature

    @torch.no_grad()
    def predict_tta(self, wave: Tensor) -> tuple[np.ndarray, np.ndarray]:
        """Xác suất [B, 36] và alpha [B, 1] trung bình qua 3 lần dịch."""
        wave = wave.to(self.device)
        logits, alphas = [], []
        for shift in SHIFTS:
            with torch.autocast(self.device.type, dtype=torch.bfloat16, enabled=self.device.type == "cuda"):
                out = self.model(shift_wave(wave, shift), return_dict=True)
            logits.append(out["logits"].float())
            alphas.append(out["alpha"].float())
        mean_logits = torch.stack(logits).mean(dim=0)
        probs = F.softmax(mean_logits / self.temperature, dim=-1).cpu().numpy()
        return probs, torch.stack(alphas).mean(dim=0).cpu().numpy()


def ensemble_with_teachers(
    p_student: np.ndarray,
    alpha: np.ndarray,
    t0_logits: np.ndarray,
    tema_logits: np.ndarray,
    teacher_temp: float = 1.20,
    student_weight: float = 0.25,
    t0_weights: tuple[float, float] = (0.45, 0.30),
    tema_weights: tuple[float, float] = (0.30, 0.45),
) -> np.ndarray:
    """P = w_s P_student + w_t0(alpha) P_t0 + w_tema(alpha) P_tema, trọng số chuẩn hoá về tổng 1 (bản gốc).

    alpha -> 0: Fold0 0.45, EMA 0.30; alpha -> 1: Fold0 0.30, EMA 0.45; student luôn 0.25."""
    p_t0 = F.softmax(torch.from_numpy(t0_logits).float() / teacher_temp, dim=-1).numpy()
    p_tema = F.softmax(torch.from_numpy(tema_logits).float() / teacher_temp, dim=-1).numpy()
    w_t0 = (1.0 - alpha) * t0_weights[0] + alpha * t0_weights[1]
    w_tema = (1.0 - alpha) * tema_weights[0] + alpha * tema_weights[1]
    total = student_weight + w_t0 + w_tema
    return (student_weight * p_student + w_t0 * p_t0 + w_tema * p_tema) / total


def load_pipeline(student_ckpt: Path | str, device: torch.device, **model_kwargs) -> StudentPipeline:
    student = CEDSingleMoESplitFFN(load_pretrained=False, **model_kwargs)
    ckpt = torch.load(student_ckpt, map_location="cpu", weights_only=False)
    student.load_state_dict(ckpt.get("model_state_dict", ckpt))
    return StudentPipeline(student, device)

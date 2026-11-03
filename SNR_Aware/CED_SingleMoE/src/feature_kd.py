"""Global Cosine Feature Distillation (GFD-Cosine): căn hướng vector gộp toàn clip của student theo teacher BEATs.

    teacher: token lớp l của BEATs (nghe nhiễu sạch) -> mean theo token -> t_l [B, 768] (xuất sẵn, fold out-of-fold)
    student: token patch (bỏ CLS) sau block b của CED -> mean theo token -> s_b [B, 192]
             (b = 12: sau LayerNorm cuối, tức đầu ra encoder trước head; b < 12: residual stream sau block b)
    projector mỗi cặp (b, l): Linear(192, 768) -> LayerNorm(768), chỉ dùng lúc train
    L_feat = mean_{cặp} mean_i (1 - cos(proj(s_b,i), t_l,i))

Gộp toàn cục nên không cần số token / lưới patch hai bên khớp nhau (BEATs 128 mel, CED 64 mel, bước patch khác).

Cơ sở: PEFD (Chen et al., "Improved Feature Distillation via Projector Ensemble", NeurIPS 2022): đặc trưng lớp cuối
trước classifier, projector chỉ đặt ở student và bỏ khi suy luận, loss trên vector đã chuẩn hoá L2
(||s/|s| - t/|t|||^2 = 2 - 2 cos, tức 2 lần loss ở đây), L = CE + 25 L_DA (~ lambda 50 theo thang 1 - cos).
Khác PEFD: projector Linear + LayerNorm (PEFD: Linear + ReLU, ensemble 3), cộng thêm KD logit, tuỳ chọn nhiều lớp.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class GlobalCosineFeatureKD(nn.Module):
    def __init__(self, student_dim: int = 192, teacher_dim: int = 768, num_taps: int = 1) -> None:
        super().__init__()
        self.projectors = nn.ModuleList(
            nn.Sequential(nn.Linear(student_dim, teacher_dim), nn.LayerNorm(teacher_dim)) for _ in range(num_taps)
        )

    def forward(self, students: list[Tensor], teacher: Tensor, mask: Tensor | None = None) -> Tensor:
        """students: num_taps tensor [B, Ds]; teacher [B, num_taps, Dt]; mask [B] (1 = có đặc trưng) -> loss vô hướng."""
        losses = []
        for k, (student, projector) in enumerate(zip(students, self.projectors)):
            z = F.normalize(projector(student.float()), dim=-1)
            t = F.normalize(teacher[:, k].float(), dim=-1)
            dist = 1.0 - (z * t).sum(dim=-1)  # [B]
            if mask is None:
                losses.append(dist.mean())
            else:
                m = mask.float()
                losses.append((dist * m).sum() / m.sum().clamp_min(1.0))
        return torch.stack(losses).mean()

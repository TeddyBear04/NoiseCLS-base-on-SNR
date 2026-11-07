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


class MaskedGenerativeFeatureKD(nn.Module):
    """MGD (Yang et al., "Masked Generative Distillation", ECCV 2022, arXiv 2205.01529), chép từ FRCRN_KD_MoE featurekd.MGDLoss
    (norm "" = MGD gốc) để cùng tham số với đợt KD-MoE kd8/ce9 (alpha 7e-5, lambda 0.5, block cuối):

        S: token patch của student sau block cuối (trước LayerNorm cuối) -> lưới (F', T') -> căn kênh conv 1x1 (C_s -> 768)
        M: che ngẫu nhiên từng vị trí (M = 0 nếu R < lambda)
        G: conv3x3 - ReLU - conv3x3 (768 kênh) tái tạo feature đầy đủ của teacher
        T: token lớp 12 BEATs (nghe nhiễu sạch, fold out-of-fold) xuất ở lưới (T_t, F_t) = (25, 4), pool thích nghi về (T', F')
           của student như _teacher_grid của FRCRN_KD_MoE (patch chồng lấn: 25 -> 48 theo thời gian)
        L = sum_{c, f, t} (T - G(align(S) * M))^2 / B   (clip thiếu đặc trưng teacher bị loại khỏi tử và mẫu)
    Lớp căn + khối sinh chỉ dùng lúc train, không vào best_model.pt."""

    def __init__(self, student_dim: int, teacher_dim: int = 768, mask_ratio: float = 0.5,
                 teacher_grid: tuple[int, int] = (25, 4)) -> None:
        super().__init__()
        self.align = nn.Conv2d(student_dim, teacher_dim, 1)
        self.generation = nn.Sequential(nn.Conv2d(teacher_dim, teacher_dim, 3, padding=1), nn.ReLU(inplace=True),
                                        nn.Conv2d(teacher_dim, teacher_dim, 3, padding=1))
        self.mask_ratio = mask_ratio
        self.teacher_grid = tuple(teacher_grid)
        self.teacher_dim = teacher_dim

    def forward(self, tokens: Tensor, student_grid: tuple[int, int], teacher: Tensor, mask: Tensor | None = None) -> Tensor:
        """tokens [B, F' * T', C_s] f-major; student_grid (F', T'); teacher [B, 1, T_t * F_t * 768] (hàng .npy làm phẳng);
        mask [B] (1 = có đặc trưng) -> loss vô hướng."""
        freq, time = student_grid
        batch = tokens.shape[0]
        student = tokens.float().reshape(batch, freq, time, -1).permute(0, 3, 1, 2)          # (B, C_s, F', T')
        t = teacher.float().reshape(batch, *self.teacher_grid, self.teacher_dim)             # (B, T_t, F_t, 768)
        t = F.adaptive_avg_pool2d(t.permute(0, 3, 1, 2), (time, freq)).permute(0, 1, 3, 2)  # (B, 768, F', T')
        feature = self.align(student)
        keep = (torch.rand(batch, 1, freq, time, device=feature.device) >= self.mask_ratio).to(feature.dtype)
        per_clip = (t - self.generation(feature * keep)).pow(2).flatten(1).sum(dim=1)        # [B]
        if mask is None:
            return per_clip.mean()
        m = mask.float()
        return (per_clip * m).sum() / m.sum().clamp_min(1.0)


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

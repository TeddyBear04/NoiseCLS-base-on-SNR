"""Feature-based KD cho student ViT (CED) từ teacher BEATs: MGD, ViTKD, MiniLMv2 (Q/K/V). Chỉ dùng khi train.

Mọi feature được đưa về lưới student (F', T') = (4, 25); teacher đã pool về (25, 4) token t-major (chỉ số t * 4 + f) khi xuất.

MGD (Yang et al., ECCV 2022, arXiv 2205.01529): student qua lớp căn 1x1 -> che ngẫu nhiên từng vị trí với xác suất lambda
    (M = 0 nếu R < lambda) -> block sinh G = conv3x3 - ReLU - conv3x3 -> tái tạo feature đầy đủ của teacher;
    L = sum (T - G(align(S) * M))^2 / B, trọng số alpha (paper: 7e-5, lambda 0.5, phân loại ImageNet).

ViTKD (Yang et al., arXiv 2209.02432): hai thành phần. Bắt chước các layer nông: L_lr = sum_l sum (T_l - fc_l(S_l))^2 / B (fc tuyến tính
    căn chiều), alpha 3e-5. Sinh layer cuối: fc căn chiều, token bị che (r_i < lambda) thay bằng mask token học được, block sinh MGD,
    chỉ tính loss trên token bị che: L_gen = sum Mask_i (T - G)^2 / B, beta 3e-6, lambda 0.5.

GFD-Cosine (Global Cosine Feature Distillation; cùng họ PEFD, Chen et al., NeurIPS 2022): token student sau block b -> mean
    theo vị trí -> projector Linear(C_s, 768) + LayerNorm(768) -> so với mean token lớp l của BEATs (xuất sẵn, lưới 1 x 1);
    L = mean_tap mean_i (1 - cos(z_i, t_i)). Gộp toàn clip nên không cần khớp số token / lưới. Mỗi tap một projector, bỏ khi
    suy luận. PEFD: đặc trưng lớp cuối trước classifier, projector chỉ ở student, 25 x (2 - 2 cos) (~ trọng số 50 ở thang này).

MiniLMv2 (Wang et al., ACL-IJCNLP 2021, arXiv 2012.15828): Q/K/V của layer cuối (student) và layer được chọn (teacher) được chia thành
    A_r relation head; R_{XX,a} = softmax(X_a X_a^T / sqrt(d_r)) (N x N); L = sum_{X in {Q,K,V}} mean_{a,t} KL(R^T_{XX,a,t} || R^S_{XX,a,t}),
    mỗi cặp hệ số 1. Số head hai bên được phép khác nhau vì ma trận quan hệ luôn N x N.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn

TEACHER_DIM = 768


def teacher_to_student_layout(tokens: Tensor) -> Tensor:
    """(B, T', F', C) token teacher t-major (đã pool về lưới student) -> (B, C, F', T') như feature map của student."""
    return tokens.float().permute(0, 3, 2, 1)


def tokens_f_major_to_grid(tokens: Tensor, grid: tuple[int, int]) -> Tensor:
    """(B, F' * T', C) token student f-major ('b (f t) c' của CED) -> (B, C, F', T')."""
    freq, time = grid
    return tokens.reshape(tokens.shape[0], freq, time, tokens.shape[-1]).permute(0, 3, 1, 2)


class GenerationBlock(nn.Module):
    """G(F) = W_l2(ReLU(W_l1(F))): hai conv 3x3 (MGD, Eq. 4); số kênh = chiều teacher."""

    def __init__(self, channels: int = TEACHER_DIM) -> None:
        super().__init__()
        self.net = nn.Sequential(nn.Conv2d(channels, channels, 3, padding=1), nn.ReLU(inplace=True),
                                 nn.Conv2d(channels, channels, 3, padding=1))

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


MGD_NORMS = ("", "pkd", "ln")


def channel_standardize(x: Tensor, eps: float = 1e-6) -> Tensor:
    """(B, C, F', T') -> mỗi kênh trung bình 0, phương sai 1 trên cả batch và vị trí (PKD, Cao et al., NeurIPS 2022)."""
    mean = x.mean(dim=(0, 2, 3), keepdim=True)
    var = x.var(dim=(0, 2, 3), unbiased=False, keepdim=True)
    return (x - mean) / (var + eps).sqrt()


class MGDLoss(nn.Module):
    """norm "": MGD gốc trên feature thô. Token BEATs L12 thô có vài kênh biên độ rất lớn: loss epoch 1 ~5e6 (grad aux/CE ~1e4),
    rồi block sinh tự khớp phần lớn MSE và grad tới encoder tắt dần (aux/CE 0.1 ở epoch 6 -> 0.001 ở epoch 30, đợt kd5 2026-11-02).
    norm "pkd": chuẩn hoá cả G(align(S) M) lẫn T theo kênh (PKD, Cao et al., NeurIPS 2022): L = mean (G^ - T^)^2 / 2 = 1 - Pearson,
        kênh lớn không còn chiếm gradient (paper alpha 6-10 cho detector).
    norm "ln": chỉ chuẩn hoá target teacher bằng layer norm không tham số theo chiều kênh của từng token (data2vec, Baevski et al.,
        ICML 2022: "prevents layers with high norm to dominate"), loss giữ dạng MGD sum / B."""

    def __init__(self, student_dim: int, mask_ratio: float = 0.5, norm: str = "") -> None:
        super().__init__()
        if norm not in MGD_NORMS:
            raise ValueError(f"MGD norm must be one of {MGD_NORMS}, got {norm!r}")
        self.align = nn.Conv2d(student_dim, TEACHER_DIM, 1)
        self.generation = GenerationBlock()
        self.mask_ratio = mask_ratio
        self.norm = norm

    def forward(self, student: Tensor, teacher: Tensor) -> Tensor:
        """student (B, C_s, F', T'), teacher (B, 768, F', T') -> loss MGD theo `norm`."""
        feature = self.align(student.float())
        keep = (torch.rand(feature.shape[0], 1, *feature.shape[2:], device=feature.device) >= self.mask_ratio).to(feature.dtype)
        generated = self.generation(feature * keep)
        teacher = teacher.float()
        if self.norm == "pkd":
            return (channel_standardize(generated) - channel_standardize(teacher)).pow(2).mean() / 2
        if self.norm == "ln":
            teacher = F.layer_norm(teacher.permute(0, 2, 3, 1), (teacher.shape[1],)).permute(0, 3, 1, 2)
        return (teacher - generated).pow(2).sum() / student.shape[0]


class ViTKDLoss(nn.Module):
    def __init__(self, student_dim: int, num_shallow: int, mask_ratio: float = 0.5) -> None:
        super().__init__()
        self.shallow = nn.ModuleList(nn.Conv2d(student_dim, TEACHER_DIM, 1) for _ in range(num_shallow))  # 1x1 = linear theo token
        self.align = nn.Conv2d(student_dim, TEACHER_DIM, 1)
        self.mask_token = nn.Parameter(torch.zeros(1, TEACHER_DIM, 1, 1))
        self.generation = GenerationBlock()
        self.mask_ratio = mask_ratio

    def forward(self, shallow: list[Tensor], last: Tensor, teacher_shallow: list[Tensor],
                teacher_last: Tensor) -> tuple[Tensor, Tensor]:
        """Trả (L_lr, L_gen). Mọi tensor dạng (B, C, F', T')."""
        batch = last.shape[0]
        lr = sum((t - fc(s.float())).pow(2).sum() for fc, s, t in zip(self.shallow, shallow, teacher_shallow)) / batch
        feature = self.align(last.float())
        masked = (torch.rand(batch, 1, *feature.shape[2:], device=feature.device) < self.mask_ratio).to(feature.dtype)
        generated = self.generation(feature * (1 - masked) + self.mask_token * masked)
        gen = (masked * (teacher_last - generated).pow(2)).sum() / batch
        return lr, gen


class GlobalCosineLoss(nn.Module):
    """GFD-Cosine: mỗi tap (block student <-> lớp teacher) một projector Linear(C_s, 768) + LayerNorm(768)."""

    def __init__(self, student_dim: int, num_taps: int = 1) -> None:
        super().__init__()
        self.projectors = nn.ModuleList(nn.Sequential(nn.Linear(student_dim, TEACHER_DIM), nn.LayerNorm(TEACHER_DIM))
                                        for _ in range(num_taps))

    def forward(self, students: list[Tensor], teacher: Tensor) -> Tensor:
        """students: num_taps feature map (B, C_s, F', T') theo thứ tự block tăng dần; teacher (B, num_taps, 768) cùng thứ tự
        lớp. -> mean_tap mean_batch (1 - cos)."""
        if len(students) != len(self.projectors) or teacher.shape[1] != len(self.projectors):
            raise ValueError(f"GFD: {len(students)} student taps / {teacher.shape[1]} teacher layers / "
                             f"{len(self.projectors)} projectors")
        losses = []
        for k, (student, projector) in enumerate(zip(students, self.projectors)):
            z = F.normalize(projector(student.float().mean(dim=(2, 3))), dim=-1)   # mean theo (F', T') -> (B, 768)
            t = F.normalize(teacher[:, k].float(), dim=-1)
            losses.append((1.0 - (z * t).sum(dim=-1)).mean())
        return torch.stack(losses).mean()


def relation_matrices(x: Tensor, heads: int) -> Tensor:
    """(B, N, C) -> (B, A_r, N, N): softmax(X_a X_a^T / sqrt(d_r)) theo từng relation head (MiniLMv2)."""
    batch, count, channels = x.shape
    if channels % heads:
        raise ValueError(f"{channels} channels cannot be split into {heads} relation heads")
    x = x.float().reshape(batch, count, heads, channels // heads).permute(0, 2, 1, 3)
    return (x @ x.transpose(-1, -2) / (channels // heads) ** 0.5).softmax(dim=-1)


def qkv_relation_loss(student_qkv: tuple[Tensor, Tensor, Tensor], teacher_qkv: tuple[Tensor, Tensor, Tensor],
                      heads: int) -> Tensor:
    """sum_{Q,K,V} mean_{a,t} KL(R_T || R_S); mọi tensor (B, N, C) cùng thứ tự token (t-major)."""
    total = 0.0
    for s, t in zip(student_qkv, teacher_qkv):
        r_t, r_s = relation_matrices(t, heads), relation_matrices(s, heads)
        kl = (r_t * (torch.log(r_t.clamp_min(1e-8)) - torch.log(r_s.clamp_min(1e-8)))).sum(-1)   # (B, A_r, N)
        total = total + kl.mean()
    return total

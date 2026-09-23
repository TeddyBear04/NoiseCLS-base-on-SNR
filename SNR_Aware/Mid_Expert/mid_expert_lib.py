"""Hàm thuần dùng chung giữa notebook và unit test. Không import BEATs, không cần GPU."""
from __future__ import annotations

import torch
from torch import Tensor

SNR_MIN_DB = -5.0
SNR_MAX_DB = 20.0


def mid_slice_mask(snr_db: Tensor, low: float = 5.0, high: float = 10.0) -> Tensor:
    """True cho các mẫu nằm trong dải mid, hai đầu đóng."""
    return (snr_db >= low) & (snr_db <= high)


def normalize_snr(snr_db: Tensor) -> Tensor:
    """Đưa SNR về [0, 1] để làm đầu vào cho FiLM."""
    return (snr_db - SNR_MIN_DB) / (SNR_MAX_DB - SNR_MIN_DB)


class FiLM(torch.nn.Module):
    """Sinh (gamma, beta) từ SNR rồi điều biến embedding đã pool.

    Lớp cuối khởi tạo bằng 0 để module bắt đầu ở đúng identity: gamma=1, beta=0.
    Nếu không, FiLM ngẫu nhiên sẽ phá embedding BEATs pretrained ngay bước đầu
    và finetune 8 epoch không đủ để hồi lại.

    Lưu ý: ở khởi tạo forward trả về LayerNorm(z), KHÔNG phải z, vì FiLM luôn
    áp LayerNorm trước khi điều biến bằng (gamma, beta).
    """

    def __init__(self, dim: int, hidden: int = 128) -> None:
        super().__init__()
        self.dim = dim
        self.norm = torch.nn.LayerNorm(dim)
        self.to_gamma_beta = torch.nn.Sequential(
            torch.nn.Linear(1, hidden),
            torch.nn.GELU(),
            torch.nn.Linear(hidden, 2 * dim),
        )
        torch.nn.init.zeros_(self.to_gamma_beta[-1].weight)
        torch.nn.init.zeros_(self.to_gamma_beta[-1].bias)
        self.last_gamma_beta: tuple[Tensor, Tensor] | None = None

    def forward(self, z: Tensor, snr_db: Tensor) -> Tensor:
        conditioning = normalize_snr(snr_db).to(z.dtype).unsqueeze(-1)
        gamma, beta = self.to_gamma_beta(conditioning).chunk(2, dim=-1)
        gamma = gamma + 1.0
        self.last_gamma_beta = (gamma.detach(), beta.detach())
        return gamma * self.norm(z) + beta


def sample_negatives(
    labels: Tensor,
    bank_labels: Tensor,
    n: int,
    generator: torch.Generator,
    mode: str = "different_label",
) -> Tensor:
    """Chỉ số của n negative trong bank cho từng mẫu trong batch.

    ``different_label`` tránh đẩy xa hai clip cùng lớp — thứ sẽ chống lại chính
    mục tiêu phân loại. CRD cho phép cả hai; đây là switch để ablate.
    Lấy có hoàn lại, vì n=4096 trên bank 30k thì trùng lặp không đáng kể và
    lấy không hoàn lại theo từng hàng sẽ chậm hơn nhiều.
    """
    batch = labels.shape[0]
    size = bank_labels.shape[0]
    if mode == "random":
        return torch.randint(0, size, (batch, n), generator=generator)
    if mode != "different_label":
        raise ValueError(f"mode không hợp lệ: {mode!r}")

    out = torch.empty(batch, n, dtype=torch.long)
    for row in range(batch):
        allowed = (bank_labels != labels[row]).nonzero(as_tuple=True)[0]
        if allowed.numel() == 0:
            raise ValueError("Bank không có mẫu khác nhãn nào.")
        picks = torch.randint(0, allowed.numel(), (n,), generator=generator)
        out[row] = allowed[picks]
    return out


def film_deviation(film: FiLM) -> float:
    """Khoảng cách của (gamma, beta) so với identity. Gần 0 => FiLM đã sụp."""
    if film.last_gamma_beta is None:
        return 0.0
    gamma, beta = film.last_gamma_beta
    return float(torch.cat([gamma - 1.0, beta], dim=-1).norm(dim=-1).mean())


class Projection(torch.nn.Module):
    """Chiếu tuyến tính rồi chuẩn hoá L2, đúng như g_S / g_T trong CRD."""

    def __init__(self, in_dim: int, out_dim: int = 128) -> None:
        super().__init__()
        self.linear = torch.nn.Linear(in_dim, out_dim)

    def forward(self, x: Tensor) -> Tensor:
        return torch.nn.functional.normalize(self.linear(x), dim=-1)


class CRDLoss(torch.nn.Module):
    """Contrastive Representation Distillation (Tian, Krishnan, Isola, ICLR 2020).

    Theo đúng công thức NCE của repo tham chiếu (HobbitLong/RepDistiller,
    ``crd/memory.py`` + ``crd/criterion.py``), gồm HAI bước — bản đầu tiên
    trong kế hoạch (Task 4 Step 7) chỉ có bước 1 và thiếu bước 2, khiến giá
    trị loss ở bước 9 lớn gấp ~2500 lần CE (đo được 9066.96, xem báo cáo):

        1. out = exp(<v, v'> / tau)
        2. out = out / Z,  Z = out.mean() * n_data, tính MỘT LẦN ở batch đầu
           rồi giữ cố định.

    Không chuẩn hoá Z thì giá trị đưa vào critic là exponential thô chứ
    không phải xác suất, nên ``log(1 - h_neg)`` không gần 0 và tổng theo
    4096 negative nổ lên hàng nghìn. Có Z, ``P_neg`` ở cỡ ``1/n_data`` nên
    ``log_D0`` gần 0 và tổng vẫn nhỏ — đúng như CRD gốc.

        P      = exp(<g_t, g_s> / tau) / Z        (Z cố định sau batch đầu)
        Pn     = 1 / n_data
        log_D1 = log( P_pos / (P_pos + m*Pn) )
        log_D0 = log( m*Pn  / (P_neg + m*Pn) )
        loss   = -(log_D1.sum() + log_D0.sum()) / batch_size

    Tổng theo negative rồi chia cho batch size (không chia cho n_neg) —
    đúng như repo tham chiếu.
    """

    def __init__(self, n_data: int, tau: float = 0.07, eps: float = 1e-7) -> None:
        super().__init__()
        self.n_data, self.tau, self.eps = n_data, tau, eps
        self.register_buffer("Z", torch.tensor(-1.0))

    def forward(self, z_s: Tensor, z_t_pos: Tensor, z_t_neg: Tensor) -> Tensor:
        # z_s [B, D], z_t_pos [B, D], z_t_neg [B, m, D] -- đã L2-normalize
        batch = z_s.shape[0]
        m = z_t_neg.shape[1]
        pos = torch.exp((z_s * z_t_pos).sum(-1, keepdim=True) / self.tau)  # [B, 1]
        neg = torch.exp(
            torch.bmm(z_t_neg, z_s.unsqueeze(-1)).squeeze(-1) / self.tau
        )  # [B, m]

        if self.Z.item() < 0:
            with torch.no_grad():
                self.Z.fill_(torch.cat([pos, neg], dim=1).mean().item() * self.n_data)

        p_pos, p_neg = pos / self.Z, neg / self.Z
        pn = 1.0 / float(self.n_data)
        log_d1 = torch.log(p_pos / (p_pos + m * pn + self.eps))
        log_d0 = torch.log((m * pn) / (p_neg + m * pn + self.eps))
        return -(log_d1.sum() + log_d0.sum()) / batch

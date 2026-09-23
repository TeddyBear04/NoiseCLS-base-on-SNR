"""Ham thuan dung chung giua notebook va unit test. Khong import BEATs, khong can GPU."""
from __future__ import annotations

import torch
from torch import Tensor

SNR_MIN_DB = -5.0
SNR_MAX_DB = 20.0


def mid_slice_mask(snr_db: Tensor, low: float = 5.0, high: float = 10.0) -> Tensor:
    """True cho cac mau nam trong dai mid, hai ?au ?ong."""
    return (snr_db >= low) & (snr_db <= high)


def normalize_snr(snr_db: Tensor) -> Tensor:
    """?ua SNR ve [0, 1] ?e lam ?au vao cho FiLM."""
    return (snr_db - SNR_MIN_DB) / (SNR_MAX_DB - SNR_MIN_DB)


class FiLM(torch.nn.Module):
    """Sinh (gamma, beta) tu SNR roi ?ieu bien embedding ?a pool.

    Lop cuoi khoi tao bang 0 ?e module bat ?au o ?ung identity: gamma=1, beta=0.
    Neu khong, FiLM ngau nhien se pha embedding BEATs pretrained ngay buoc ?au
    va finetune 8 epoch khong ?u ?e hoi lai.

    Luu y: o khoi tao forward tra ve LayerNorm(z), KHONG phai z, vi FiLM luon
    ap LayerNorm truoc khi ?ieu bien bang (gamma, beta).
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
    """Chi so cua n negative trong bank cho tung mau trong batch.

    ``different_label`` tranh ?ay xa hai clip cung lop  -  thu se chong lai chinh
    muc tieu phan loai. CRD cho phep ca hai; ?ay la switch ?e ablate.
    Lay co hoan lai, vi n=4096 tren bank 30k thi trung lap khong ?ang ke va
    lay khong hoan lai theo tung hang se cham hon nhieu.
    """
    batch = labels.shape[0]
    size = bank_labels.shape[0]
    if mode == "random":
        return torch.randint(0, size, (batch, n), generator=generator)
    if mode != "different_label":
        raise ValueError(f"mode khong hop le: {mode!r}")

    out = torch.empty(batch, n, dtype=torch.long)
    for row in range(batch):
        allowed = (bank_labels != labels[row]).nonzero(as_tuple=True)[0]
        if allowed.numel() == 0:
            raise ValueError("Bank khong co mau khac nhan nao.")
        picks = torch.randint(0, allowed.numel(), (n,), generator=generator)
        out[row] = allowed[picks]
    return out


def film_deviation(film: FiLM) -> float:
    """Khoang cach cua (gamma, beta) so voi identity. Gan 0 => FiLM ?a sup."""
    if film.last_gamma_beta is None:
        return 0.0
    gamma, beta = film.last_gamma_beta
    return float(torch.cat([gamma - 1.0, beta], dim=-1).norm(dim=-1).mean())


class Projection(torch.nn.Module):
    """Chieu tuyen tinh roi chuan hoa L2, ?ung nhu g_S / g_T trong CRD."""

    def __init__(self, in_dim: int, out_dim: int = 128) -> None:
        super().__init__()
        self.linear = torch.nn.Linear(in_dim, out_dim)

    def forward(self, x: Tensor) -> Tensor:
        return torch.nn.functional.normalize(self.linear(x), dim=-1)


class CRDLoss(torch.nn.Module):
    """Contrastive Representation Distillation (Tian, Krishnan, Isola, ICLR 2020).

    Theo ?ung cong thuc NCE cua repo tham chieu (HobbitLong/RepDistiller,
    ``crd/memory.py`` + ``crd/criterion.py``), gom HAI buoc  -  ban ?au tien
    trong ke hoach (Task 4 Step 7) chi co buoc 1 va thieu buoc 2, khien gia
    tri loss o buoc 9 lon gap ~2500 lan CE (?o ?uoc 9066.96, xem bao cao):

        1. out = exp(<v, v'> / tau)
        2. out = out / Z,  Z = out.mean() * n_data, tinh MOT LAN o batch ?au
           roi giu co ?inh.

    Khong chuan hoa Z thi gia tri ?ua vao critic la exponential tho chu
    khong phai xac suat, nen ``log(1 - h_neg)`` khong gan 0 va tong theo
    4096 negative no len hang nghin. Co Z, ``P_neg`` o co ``1/n_data`` nen
    ``log_D0`` gan 0 va tong van nho  -  ?ung nhu CRD goc.

        P      = exp(<g_t, g_s> / tau) / Z        (Z co ?inh sau batch ?au)
        Pn     = 1 / n_data
        log_D1 = log( P_pos / (P_pos + m*Pn) )
        log_D0 = log( m*Pn  / (P_neg + m*Pn) )
        loss   = -(log_D1.sum() + log_D0.sum()) / batch_size

    Tong theo negative roi chia cho batch size (khong chia cho n_neg)  - 
    ?ung nhu repo tham chieu.
    """

    def __init__(self, n_data: int, tau: float = 0.07, eps: float = 1e-7) -> None:
        super().__init__()
        self.n_data, self.tau, self.eps = n_data, tau, eps
        self.register_buffer("Z", torch.tensor(-1.0))

    def forward(self, z_s: Tensor, z_t_pos: Tensor, z_t_neg: Tensor) -> Tensor:
        # z_s [B, D], z_t_pos [B, D], z_t_neg [B, m, D] -- ?a L2-normalize
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

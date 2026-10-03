"""Ham thuan dung chung giua notebook va unit test. Khong import BEATs, khong can GPU."""
from __future__ import annotations

import math

import torch
from torch import Tensor

SNR_MIN_DB = -5.0
SNR_MAX_DB = 20.0


def band_slice_mask(snr_db: Tensor, low: float = 5.0, high: float = 10.0) -> Tensor:
    """True cho cac mau nam trong dai [low, high], hai dau dong."""
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


class AttentionMaskPool(torch.nn.Module):
    """Soft attention mask A(t, f) tren patch token roi pool co trong so (Huong 2).

    Diem cua mask theo gated attention cua Ilse, Tomczak, Welling (ICML 2018):
        s_k = w' ( tanh(V h_k) * sigmoid(U h_k) )
    Lech paper: Ilse chuan hoa bang softmax_k(s_k). O day A_k = sigmoid(s_k) nam
    trong [0, 1] dung nhu so do ve, roi moi chuan hoa:
        z = sum_k A_k h_k / sum_k A_k
    Giu A o dang mask [0, 1] de sau nay giam sat duoc bang IRM that tu
    clean_path / noise_path; softmax khong cho lam vay.

    Voi BEATs moi token la mot o 16x16 tren log-mel (16 frame x 16 mel bin), nen
    clip 4 s cho luoi 24 thoi gian x 8 dai tan = 192 token, thu tu t*8 + f.

    w khoi tao bang 0 => A = 0.5 o moi token => z dung bang mean-pool. Run nay
    bat dau dung tu diem cua run1_baseline, giong cach FiLM bat dau o identity.
    """

    def __init__(self, dim: int, hidden: int = 128) -> None:
        super().__init__()
        self.V = torch.nn.Linear(dim, hidden)
        self.U = torch.nn.Linear(dim, hidden)
        self.w = torch.nn.Linear(hidden, 1)
        torch.nn.init.zeros_(self.w.weight)
        torch.nn.init.zeros_(self.w.bias)
        self.last_mask: Tensor | None = None

    def forward(self, tokens: Tensor) -> tuple[Tensor, Tensor]:
        # tokens [B, N, D] -> pooled [B, D], mask [B, N]
        score = self.w(torch.tanh(self.V(tokens)) * torch.sigmoid(self.U(tokens)))
        mask = torch.sigmoid(score.squeeze(-1))
        weights = mask / mask.sum(dim=1, keepdim=True)
        self.last_mask = mask.detach()
        return (weights.unsqueeze(-1) * tokens).sum(dim=1), mask


def mask_deviation(mask: Tensor) -> float:
    """Do lech chuan cua A theo token, trung binh tren batch. Gan 0 => mask deu
    khap noi, tuc attention da sup ve mean-pool va run nay chi la baseline train lai."""
    return float(mask.float().std(dim=1).mean())


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


# ---------------------------------------------------------------- finetune recipe
#
# The BEATs paper's own fine-tuning recipe (Chen et al., ICML 2023, appendix
# Table 4, ESC-50 column - the closest to this task: few clips, one label each):
# SpecAugment 0.3, roll augmentation, peak lr 1e-4 with warmup + cosine decay,
# layer-wise lr decay 0.2, dropout 0.1, layer dropout 0.1, weight decay 0.01.
# The code this project started from had none of it - constant lr, no
# augmentation, dropout 0 - and every BEATs finetune memorised the training set
# within an epoch (train loss 0.006, validation loss rising from epoch 1).


def spec_augment(fbank: Tensor, ratio: float, generator: torch.Generator) -> Tensor:
    """One time mask and one frequency mask per clip, each up to `ratio` of its axis.

    fbank: [B, frames, bins], already normalised, so 0 is the dataset mean and a
    masked band reads as "average energy" rather than silence. The paper gives
    the SpecAugment strength as a single 0.3 without its exact parameterisation;
    one mask per axis of width U[0, ratio*size) is the reading used here.
    """
    if ratio <= 0:
        return fbank
    out = fbank.clone()
    batch, frames, bins = out.shape
    for axis_size, axis in ((frames, 1), (bins, 2)):
        widths = (torch.rand(batch, generator=generator) * ratio * axis_size).long()
        starts = (torch.rand(batch, generator=generator)
                  * (axis_size - widths).clamp(min=1)).long()
        for i in range(batch):
            if widths[i] == 0:
                continue
            if axis == 1:
                out[i, starts[i]:starts[i] + widths[i], :] = 0
            else:
                out[i, :, starts[i]:starts[i] + widths[i]] = 0
    return out


def roll_waveform(waveform: Tensor, generator: torch.Generator) -> Tensor:
    """Circular time shift by a random offset per clip (the paper's roll augmentation)."""
    length = waveform.shape[-1]
    shifts = (torch.rand(waveform.shape[0], generator=generator) * length).long()
    return torch.stack([torch.roll(clip, int(s), dims=-1)
                        for clip, s in zip(waveform, shifts)])


def warmup_cosine(step: int, total: int, warmup: int) -> float:
    """LR multiplier: linear warmup to 1, then cosine decay to 0 at `total`."""
    if total <= 0:
        return 1.0
    if warmup > 0 and step < warmup:
        return (step + 1) / warmup
    progress = min(1.0, (step - warmup) / max(1, total - warmup))
    return 0.5 * (1.0 + math.cos(math.pi * progress))


def layer_decay_scales(num_blocks: int, decay: float) -> list[float]:
    """LR scale per transformer block, bottom (index 0) to top: decay^(depth from top).

    Implemented as optimizer parameter groups because the BEATs checkpoint's own
    `layer_wise_gradient_decay_ratio` multiplies gradients, and AdamW divides a
    constant gradient scale straight back out - it has almost no effect there.
    """
    return [decay ** (num_blocks - 1 - index) for index in range(num_blocks)]


@torch.no_grad()
def sam_ascend(params: list, rho: float) -> list:
    """SAM's first half (Foret et al., ICLR 2021, Eq. 2 with p = 2): move every
    parameter to w + eps, eps = rho * g / ||g||_2, the norm taken over ALL of them.

    Reads the gradients already in `.grad` and returns the eps it added, so
    `sam_descend` can put the weights back after the second backward pass. A
    parameter without a gradient gets no eps (None) and stays where it is.
    """
    grads = [p.grad for p in params if p.grad is not None]
    if not grads:
        return [None] * len(params)
    norm = torch.norm(torch.stack([g.detach().float().norm(2) for g in grads]), 2)
    scale = rho / (norm + 1e-12)
    eps = []
    for p in params:
        if p.grad is None:
            eps.append(None)
            continue
        e = (p.grad.float() * scale).to(p.dtype)
        p.add_(e)
        eps.append(e)
    return eps


@torch.no_grad()
def sam_descend(params: list, eps: list) -> None:
    """Undo `sam_ascend`: back to w, keeping the gradient taken at w + eps."""
    for p, e in zip(params, eps):
        if e is not None:
            p.sub_(e)


def distill_kl(student_logits: Tensor, teacher_logits: Tensor, temperature: float) -> Tensor:
    """Hinton KD: KL(softmax(t/T) || softmax(s/T)) * T^2, averaged over the batch."""
    return torch.nn.functional.kl_div(
        torch.nn.functional.log_softmax(student_logits / temperature, dim=-1),
        torch.nn.functional.softmax(teacher_logits / temperature, dim=-1),
        reduction="batchmean") * (temperature ** 2)

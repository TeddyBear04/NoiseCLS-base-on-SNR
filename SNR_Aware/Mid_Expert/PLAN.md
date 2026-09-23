# Mid-SNR Expert Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Dựng một expert phân loại 36 nhãn noise cho dải 5–10 dB vượt BEATs baseline (acc 0.6681 / macro-F1 0.6602 trên lát mid của test).

**Architecture:** Hai giai đoạn. Giai đoạn 1 huấn luyện teacher BEATs trên waveform noise **sạch** (`noise_path`) — đây là privileged information chỉ có lúc train — rồi precompute toàn bộ embedding + logits của teacher thành một bank tĩnh. Giai đoạn 2 huấn luyện student BEATs trên **mixture, toàn bộ dải SNR**, điều kiện hoá bằng FiLM theo SNR, với loss `CE + KD + CRD` trong đó CRD kéo embedding mixture về embedding noise sạch bằng contrastive với 4096 negative lấy từ bank tĩnh. Lúc infer chỉ còn student + mixture.

**Tech Stack:** PyTorch, BEATs (vendored tại `SNR_Aware/BEATs_Experts/models/beats/`), soundfile, torchaudio, scikit-learn. Chạy trên molab (GPU CUDA, bf16 khi có).

**Spec:** `SNR_Aware/Mid_Expert/DESIGN.md`

> **Tiến độ thật nằm ở `STATUS.md` — đọc file đó trước.** Plan này là tài liệu lập kế hoạch; một số chỗ đã bị thực tế lật (Task 1 đã chạy và bác bỏ giả định dedup; `crd_loss` ở Task 4 đã bị thay bằng `CRDLoss`). `STATUS.md` giữ danh sách cập nhật.

## Global Constraints

Sao nguyên văn từ spec — mọi task đều ngầm chịu ràng buộc này:

- **Deliverable nằm TRONG repo, tại `SNR_Aware/Mid_Expert/`, trên nhánh `mid-expert`.** Molab lấy code bằng cách clone repo, nên mọi thứ cần chạy đều phải commit + push. Các thư mục khác của repo chỉ được **đọc** và import.
- **`.gitignore` đã chặn `*.pt`** — checkpoint và bank teacher không vào git; chúng sinh ra và ở lại trên molab.
- **Local không có pytest.** Chạy test bằng `python test_mid_expert.py` (runner thuần ở cuối file), không `pip install`.
- **Dataset có bản local** tại `36_labels/`, cùng schema với `/marimo/dataset/mix-dataset` — dùng được để audit và test nhỏ.
- **Giao hàng là một notebook `.ipynb` mới**, CONFIG gộp trong **một cell duy nhất**, không tách file config riêng.
- **Training chạy trên molab**, không chạy local, không `pip install`. Unit test cho hàm thuần thì chạy local được (torch có sẵn).
- Pretrained: `BEATs_iter3_plus_AS2M_finetuned_on_AS2M_cpt2.pt`.
- Dataset: `/marimo/dataset/mix-dataset`, `manifest.csv`, `labels.txt`.
- Seed toàn cục: **2026** (khớp mọi config hiện có).
- Siêu tham số loss **lấy từ repo CRD, không tự đặt**: `r=1.0`, `a=1.0`, `b=0.8`, `ρ=4.0`, `τ=0.07`, chiều chiếu `128`, `N=4096` negative, L2-normalize trước khi nhân trong.
- Audio: 16 kHz, 4.0 s ⇒ 64.000 mẫu.
- Metric chọn checkpoint: **macro-F1 trên lát mid (5 và 10 dB) của validation**, không phải toàn bộ validation.
- Mốc phải vượt: **acc 0.6681, macro-F1 0.6602**. Mid expert cũ: 0.6616 / 0.6547.

---

## File Structure

Tất cả nằm trong scratchpad `C:\Users\DELL\AppData\Local\Temp\claude\d--language-Project-IV-Capstone\21487991-3225-41ca-8937-d65c770fd43c\scratchpad\`:

| File | Trách nhiệm |
|---|---|
| `mid_expert.ipynb` | Deliverable. CONFIG cell + toàn bộ pipeline, chạy trên molab. |
| `test_mid_expert.py` | Unit test local cho các hàm thuần (FiLM, CRD, bank negative, remix, chọn lát mid). Không chạy trên molab. |
| `mid_expert_lib.py` | Các hàm thuần mà cả notebook lẫn test cùng import. Notebook `%run` file này hoặc paste vào một cell — xem Task 2. |

Lý do tách `mid_expert_lib.py`: hàm thuần phải test được local mà không cần GPU, dataset hay BEATs. Toàn bộ thứ cần GPU/dataset ở lại trong notebook.

Notebook mở đầu bằng một cell `sys.path.insert` trỏ vào `SNR_Aware/BEATs_Experts` để import lại `load_beats`, `metrics`, `MixNoiseDataset`, `initialize_head` thay vì viết lại.

---

### Task 1: Audit manifest và check tuyến tính

Chặn cho Task 8. Phải chạy **đầu tiên** vì nó quyết định có 3 hay 4 run.

**Files:**
- Create: `mid_expert.ipynb` (cell 1–3)

**Interfaces:**
- Consumes: `noise_pipeline.mix_data.load_mix_manifest`, `load_float_audio` từ repo.
- Produces: biến notebook `LINEAR_OK: bool`, `UNIQUE_NOISE_TRAIN: int`, và kết luận in ra.

- [ ] **Step 1: Cell bootstrap đường dẫn**

```python
import sys, json, random, math, time
from pathlib import Path
import numpy as np
import torch

REPO = Path("/content/Capstone/SNR_Aware/BEATs_Experts")   # chỉnh theo layout molab
sys.path.insert(0, str(REPO))
DATA_ROOT = Path("/marimo/dataset/mix-dataset")

from noise_pipeline.mix_data import load_mix_manifest, load_float_audio
rows = load_mix_manifest(DATA_ROOT)
print("rows", len(rows), "columns", sorted(rows[0].keys()))
```

Kỳ vọng: 43.200 hàng, và trong cột có `noise_ytid`, `noise_path`, `mixture_path`, `target_snr_db`, `label_names`, `split`, `sample_id`.

- [ ] **Step 2: Đếm clip noise duy nhất (xác nhận giả định §3.1 của spec)**

```python
train_rows = [r for r in rows if r["split"] == "train"]
UNIQUE_NOISE_TRAIN = len({r["noise_path"] for r in train_rows})
print("train rows", len(train_rows), "unique noise_path", UNIQUE_NOISE_TRAIN)
print("unique noise_ytid", len({r["noise_ytid"] for r in train_rows}))
print("ratio", len(train_rows) / UNIQUE_NOISE_TRAIN)
```

Kỳ vọng: `ratio` ≈ 6.0, tức mỗi clip noise dùng lại qua cả 6 mức SNR. **Nếu ratio ≈ 1.0 thì mỗi hàng có noise riêng** — bỏ dedup ở Task 5 và ghi lại phát hiện này, vì nó đổi kích thước tập train của teacher.

- [ ] **Step 3: Check tuyến tính trên 200 clip**

```python
rng = random.Random(2026)
sample = rng.sample(train_rows, 200)
errs = []
for r in sample:
    mix = load_float_audio(DATA_ROOT / r["mixture_path"], 16_000, 64_000)
    noi = load_float_audio(DATA_ROOT / r["noise_path"], 16_000, 64_000)
    speech = mix - noi
    err = (mix - (speech + noi)).norm() / (mix.norm() + 1e-12)
    errs.append(err.item())
errs = np.array(errs)
print(f"err mean={errs.mean():.3e} max={errs.max():.3e}")
```

Check này đúng là tautology về mặt đại số (`mix − (mix − noi + noi) ≡ 0`). Nó chỉ bắt lỗi numeric/shape. **Check thật sự cần là Step 4.**

- [ ] **Step 4: Check remix có tái tạo được SNR trong manifest không**

Đây mới là điều kiện thật cho Task 8: liệu `speech = mix − noise` có phải tín hiệu speech hợp lệ, và SNR đo được có khớp `target_snr_db` không.

```python
def snr_db(signal, noise):
    return 10 * torch.log10(signal.pow(2).mean() / (noise.pow(2).mean() + 1e-12))

deltas = []
for r in sample:
    mix = load_float_audio(DATA_ROOT / r["mixture_path"], 16_000, 64_000)
    noi = load_float_audio(DATA_ROOT / r["noise_path"], 16_000, 64_000)
    speech = mix - noi
    measured = snr_db(speech, noi).item()
    deltas.append(measured - float(r["target_snr_db"]))
deltas = np.array(deltas)
print(f"delta_snr mean={deltas.mean():.3f} std={deltas.std():.3f} "
      f"absmax={np.abs(deltas).max():.3f}")
LINEAR_OK = bool(np.abs(deltas).max() < 1.0)
print("LINEAR_OK =", LINEAR_OK)
```

Ngưỡng: `|delta| < 1.0` dB trên **mọi** clip mẫu.
- Pass ⇒ `speech` khôi phục đúng, `noise_path` đã được scale sẵn theo SNR đích ⇒ Task 8 chạy được.
- Fail ⇒ bỏ Task 8. Ghi lại `delta` quan sát được để giải thích trong báo cáo.

- [ ] **Step 5: Ghi kết quả audit**

```python
AUDIT = {
    "rows": len(rows), "train_rows": len(train_rows),
    "unique_noise_train": UNIQUE_NOISE_TRAIN,
    "reuse_ratio": len(train_rows) / UNIQUE_NOISE_TRAIN,
    "delta_snr_absmax": float(np.abs(deltas).max()),
    "LINEAR_OK": LINEAR_OK,
}
print(json.dumps(AUDIT, indent=2))
```

Dán output này vào phần kết quả của báo cáo — nó là căn cứ cho quyết định có Task 8 hay không.

---

### Task 2: CONFIG cell và thư viện hàm thuần

**Files:**
- Modify: `mid_expert.ipynb` (thêm CONFIG cell)
- Create: `mid_expert_lib.py`
- Test: `test_mid_expert.py`

**Interfaces:**
- Produces: `CONFIG: dict`; `mid_expert_lib.mid_slice_mask(snr: Tensor) -> Tensor`; `mid_expert_lib.normalize_snr(snr_db: Tensor) -> Tensor`.

- [ ] **Step 1: Viết CONFIG cell (một cell duy nhất, toàn bộ tham số)**

```python
CONFIG = {
    "seed": 2026,
    "data_root": "/marimo/dataset/mix-dataset",
    "repo": "/content/Capstone/SNR_Aware/BEATs_Experts",
    "pretrained": "checkpoint/pretrained/BEATs_iter3_plus_AS2M_finetuned_on_AS2M_cpt2.pt",
    "out_dir": "artifacts/mid_expert",

    "audio": {"sample_rate": 16000, "seconds": 4.0, "num_samples": 64000},
    "mid_band_db": [5.0, 10.0],

    # --- giai đoạn 1: teacher trên noise sạch ---
    "teacher": {
        "dedup_by": "noise_path",
        "head_epochs": 50, "head_lr": 1e-3, "head_batch_size": 512, "head_patience": 10,
        "finetune_epochs": 8, "trainable_blocks": 12,
        "head_ft_lr": 1e-4, "encoder_lr": 1e-5,
        "batch_size": 32, "accumulation_steps": 1, "workers": 4,
        "checkpoint": "teacher_noise_best.pt",
        "bank": "teacher_bank.pt",
    },

    # --- giai đoạn 2: student trên mixture ---
    "student": {
        "finetune_epochs": 8, "trainable_blocks": 12,
        "head_lr": 1e-4, "encoder_lr": 1e-5, "film_lr": 1e-3,
        "batch_size": 32, "validation_batch_size": 64,
        "accumulation_steps": 1, "workers": 4, "patience": 3,
        "film": {"enabled": True, "hidden": 128, "snr_jitter_db": 2.0,
                 "placement": "pooled"},      # "pooled" | "per_layer" (ablation)
    },

    # --- loss: giá trị lấy từ repo CRD chính thức, KHÔNG tự đặt ---
    "loss": {
        "r_ce": 1.0,          # repo CRD -r
        "a_kd": 1.0,          # repo CRD -a
        "b_crd": 0.8,         # repo CRD -b
        "rho_kd": 4.0,        # nhiệt độ KD
        "tau_nce": 0.07,      # CRD, cấu hình ImageNet
        "proj_dim": 128,      # CRD
        "n_negatives": 4096,  # CRD: chênh với 16384 < 0.1%
        "negative_mode": "different_label",   # "different_label" | "random"
    },

    # --- remix, chỉ bật nếu Task 1 Step 4 pass ---
    "remix": {"enabled": False, "snr_low_db": 3.0, "snr_high_db": 12.0, "prob": 0.5},

    # --- run nào đang chạy ---
    "run": "run2_ce_only",   # run1_teacher | run2_ce_only | run3_kd_crd | run4_remix
}
```

Tắt/bật từng thành phần cho các run bằng cách sửa `run` và đặt `a_kd=0, b_crd=0` cho run 2. Không tạo file config riêng.

- [ ] **Step 2: Viết failing test cho hai hàm thuần**

```python
# test_mid_expert.py
import torch
from mid_expert_lib import mid_slice_mask, normalize_snr


def test_mid_slice_mask_selects_only_5_and_10_db():
    snr = torch.tensor([-5.0, 0.0, 5.0, 10.0, 15.0, 20.0])
    mask = mid_slice_mask(snr, low=5.0, high=10.0)
    assert mask.tolist() == [False, False, True, True, False, False]


def test_mid_slice_mask_includes_interior_values():
    snr = torch.tensor([4.9, 5.0, 7.5, 10.0, 10.1])
    assert mid_slice_mask(snr, low=5.0, high=10.0).tolist() == [
        False, True, True, True, False
    ]


def test_normalize_snr_maps_band_to_unit_interval():
    snr = torch.tensor([-5.0, 7.5, 20.0])
    out = normalize_snr(snr)
    assert torch.allclose(out, torch.tensor([0.0, 0.5, 1.0]), atol=1e-6)
```

- [ ] **Step 3: Chạy test, xác nhận FAIL**

Run: `python -m pytest test_mid_expert.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'mid_expert_lib'`

- [ ] **Step 4: Viết implementation tối thiểu**

```python
# mid_expert_lib.py
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
```

- [ ] **Step 5: Chạy test, xác nhận PASS**

Run: `python -m pytest test_mid_expert.py -v`
Expected: 3 passed

---

### Task 3: FiLM conditioning

**Files:**
- Modify: `mid_expert_lib.py`
- Test: `test_mid_expert.py`

**Interfaces:**
- Consumes: `normalize_snr`.
- Produces: `mid_expert_lib.FiLM(dim: int, hidden: int = 128)` với `forward(z: Tensor[B, D], snr_db: Tensor[B]) -> Tensor[B, D]`, và thuộc tính `last_gamma_beta: tuple[Tensor, Tensor] | None` để theo dõi rủi ro "sụp về identity" ở §10 của spec.

**Lệch paper — ghi rõ trong báo cáo:** FiLM gốc (Perez et al.) điều kiện hoá ở **nhiều tầng** của mạng. Ở đây mặc định chỉ đặt **một lần trên embedding đã pool**, vì làm per-layer phải vá vào BEATs vendored. Bản per-layer là bản trung thành với paper và nằm ở Task 9 ablation. Không được viết "áp dụng FiLM" trơn trong báo cáo mà không nói chỗ đặt.

- [ ] **Step 1: Viết failing test**

```python
from mid_expert_lib import FiLM


def test_film_is_identity_at_initialisation():
    """Khởi tạo lớp cuối bằng 0 ⇒ gamma=1, beta=0 ⇒ không phá embedding pretrained."""
    torch.manual_seed(0)
    film = FiLM(dim=8)
    z = torch.randn(4, 8)
    snr = torch.tensor([5.0, 10.0, -5.0, 20.0])
    assert torch.allclose(film(z, snr), z, atol=1e-6)


def test_film_output_depends_on_snr_after_perturbing_weights():
    torch.manual_seed(0)
    film = FiLM(dim=8)
    with torch.no_grad():
        film.to_gamma_beta[-1].weight.normal_(0, 0.5)
        film.to_gamma_beta[-1].bias.normal_(0, 0.5)
    z = torch.randn(1, 8).expand(2, 8).contiguous()
    out = film(z, torch.tensor([5.0, 20.0]))
    assert not torch.allclose(out[0], out[1], atol=1e-4)


def test_film_records_gamma_beta_for_collapse_monitoring():
    film = FiLM(dim=8)
    film(torch.randn(3, 8), torch.tensor([5.0, 10.0, 15.0]))
    gamma, beta = film.last_gamma_beta
    assert gamma.shape == (3, 8) and beta.shape == (3, 8)
```

- [ ] **Step 2: Chạy test, xác nhận FAIL**

Run: `python -m pytest test_mid_expert.py -k film -v`
Expected: FAIL — `ImportError: cannot import name 'FiLM'`

- [ ] **Step 3: Implementation**

```python
class FiLM(torch.nn.Module):
    """Sinh (gamma, beta) từ SNR rồi điều biến embedding đã pool.

    Lớp cuối khởi tạo bằng 0 để module bắt đầu ở đúng identity: gamma=1, beta=0.
    Nếu không, FiLM ngẫu nhiên sẽ phá embedding BEATs pretrained ngay bước đầu
    và finetune 8 epoch không đủ để hồi lại.
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
```

Lưu ý: ở khởi tạo `gamma=1, beta=0` nên `forward` trả về `LayerNorm(z)`, **không** phải `z`. Test 1 sẽ fail. Sửa: giữ đường tắt identity bằng cách khởi tạo `norm` là identity-preserving không đủ — thay vào đó test phải so với `self.norm(z)`. Cập nhật test 1 thành:

```python
def test_film_is_identity_at_initialisation():
    torch.manual_seed(0)
    film = FiLM(dim=8)
    z = torch.randn(4, 8)
    snr = torch.tensor([5.0, 10.0, -5.0, 20.0])
    assert torch.allclose(film(z, snr), film.norm(z), atol=1e-6)
```

Đây là hành vi đúng: FiLM mở đầu như một LayerNorm thuần, phần điều kiện hoá bằng 0.

- [ ] **Step 4: Chạy test, xác nhận PASS**

Run: `python -m pytest test_mid_expert.py -k film -v`
Expected: 3 passed

- [ ] **Step 5: Ghi lại số liệu theo dõi collapse**

Thêm hàm để dùng trong vòng train:

```python
def film_deviation(film: FiLM) -> float:
    """Khoảng cách của (gamma, beta) so với identity. Gần 0 ⇒ FiLM đã sụp."""
    if film.last_gamma_beta is None:
        return 0.0
    gamma, beta = film.last_gamma_beta
    return float(torch.cat([gamma - 1.0, beta], dim=-1).norm(dim=-1).mean())
```

Test:

```python
def test_film_deviation_is_zero_at_initialisation():
    from mid_expert_lib import film_deviation
    film = FiLM(dim=8)
    film(torch.randn(3, 8), torch.tensor([5.0, 10.0, 15.0]))
    assert film_deviation(film) < 1e-6
```

Run: `python -m pytest test_mid_expert.py -k film -v` → 4 passed

---

### Task 4: CRD loss với bank negative tĩnh

> ⚠️ **PHẦN `crd_loss` TRONG TASK NÀY ĐÃ BỊ THAY.** Bản rút gọn thiếu chuẩn hoá
> phân hoạch `Z` của CRD nên loss lúc khởi tạo đo được 9066.96 (CE ≈ 3.58).
> Bản đúng là class `CRDLoss` ở cuối Step 7. Các test ở Step 5 bên dưới viết theo
> API hàm cũ — code thật trong `mid_expert_lib.py` dùng API module.

Đây là phần dễ sai nhất trong cả plan. Sai ở đây thì run 3 sẽ ra kết quả vô nghĩa mà vẫn chạy trót lọt.

**Files:**
- Modify: `mid_expert_lib.py`
- Test: `test_mid_expert.py`

**Interfaces:**
- Produces:
  - `mid_expert_lib.sample_negatives(labels: Tensor[B], bank_labels: Tensor[M], n: int, generator, mode: str) -> Tensor[B, n]` — chỉ số vào bank.
  - `mid_expert_lib.CRDLoss(n_data: int, tau: float).forward(z_s: Tensor[B, D], z_t_pos: Tensor[B, D], z_t_neg: Tensor[B, m, D]) -> Tensor[]` — đầu vào **đã L2-normalize**. Là module chứ không phải hàm, vì phải giữ buffer `Z`.
  - `mid_expert_lib.Projection(in_dim: int, out_dim: int = 128)` — Linear + L2 norm.

- [ ] **Step 1: Viết failing test cho lấy mẫu negative**

```python
from mid_expert_lib import sample_negatives


def test_sample_negatives_never_picks_same_label_in_different_label_mode():
    labels = torch.tensor([0, 1])
    bank_labels = torch.tensor([0, 0, 1, 1, 2, 2, 3, 3])
    g = torch.Generator().manual_seed(0)
    idx = sample_negatives(labels, bank_labels, n=4, generator=g,
                           mode="different_label")
    assert idx.shape == (2, 4)
    assert (bank_labels[idx[0]] != 0).all()
    assert (bank_labels[idx[1]] != 1).all()


def test_sample_negatives_random_mode_may_include_same_label():
    labels = torch.zeros(1, dtype=torch.long)
    bank_labels = torch.zeros(64, dtype=torch.long)
    g = torch.Generator().manual_seed(0)
    idx = sample_negatives(labels, bank_labels, n=8, generator=g, mode="random")
    assert idx.shape == (1, 8)


def test_sample_negatives_is_reproducible_under_same_seed():
    labels = torch.tensor([2])
    bank_labels = torch.arange(36).repeat(10)
    a = sample_negatives(labels, bank_labels, n=16,
                         generator=torch.Generator().manual_seed(7),
                         mode="different_label")
    b = sample_negatives(labels, bank_labels, n=16,
                         generator=torch.Generator().manual_seed(7),
                         mode="different_label")
    assert torch.equal(a, b)
```

- [ ] **Step 2: Chạy test, xác nhận FAIL**

Run: `python -m pytest test_mid_expert.py -k negatives -v`
Expected: FAIL — `cannot import name 'sample_negatives'`

- [ ] **Step 3: Implementation lấy mẫu negative**

```python
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
```

- [ ] **Step 4: Chạy test, xác nhận PASS**

Run: `python -m pytest test_mid_expert.py -k negatives -v`
Expected: 3 passed

- [ ] **Step 5: Viết failing test cho CRD loss**

```python
from mid_expert_lib import Projection, crd_loss


def _unit(x):
    return torch.nn.functional.normalize(x, dim=-1)


def test_crd_loss_is_lower_when_positive_pair_aligns():
    torch.manual_seed(0)
    z_s = _unit(torch.randn(4, 16))
    neg = _unit(torch.randn(4, 32, 16))
    aligned = crd_loss(z_s, z_s.clone(), neg, tau=0.07, dataset_size=1000)
    opposed = crd_loss(z_s, -z_s.clone(), neg, tau=0.07, dataset_size=1000)
    assert aligned < opposed


def test_crd_loss_is_positive_and_finite():
    torch.manual_seed(0)
    z_s = _unit(torch.randn(8, 16))
    pos = _unit(torch.randn(8, 16))
    neg = _unit(torch.randn(8, 32, 16))
    value = crd_loss(z_s, pos, neg, tau=0.07, dataset_size=1000)
    assert torch.isfinite(value) and value > 0


def test_crd_loss_backpropagates_to_student_only():
    torch.manual_seed(0)
    z_s = _unit(torch.randn(4, 16)).requires_grad_(True)
    pos = _unit(torch.randn(4, 16))
    neg = _unit(torch.randn(4, 32, 16))
    crd_loss(z_s, pos, neg, tau=0.07, dataset_size=1000).backward()
    assert z_s.grad is not None and torch.isfinite(z_s.grad).all()


def test_projection_output_is_unit_norm():
    proj = Projection(16, 8)
    out = proj(torch.randn(5, 16))
    assert out.shape == (5, 8)
    assert torch.allclose(out.norm(dim=-1), torch.ones(5), atol=1e-5)
```

- [ ] **Step 6: Chạy test, xác nhận FAIL**

Run: `python -m pytest test_mid_expert.py -k "crd or projection" -v`
Expected: FAIL — `cannot import name 'Projection'`

- [ ] **Step 7: Implementation CRD**

```python
class Projection(torch.nn.Module):
    """Chiếu tuyến tính rồi chuẩn hoá L2, đúng như g_S / g_T trong CRD."""

    def __init__(self, in_dim: int, out_dim: int = 128) -> None:
        super().__init__()
        self.linear = torch.nn.Linear(in_dim, out_dim)

    def forward(self, x: Tensor) -> Tensor:
        return torch.nn.functional.normalize(self.linear(x), dim=-1)


def crd_loss(
    z_s: Tensor,
    z_t_pos: Tensor,
    z_t_neg: Tensor,
    tau: float,
    dataset_size: int,
) -> Tensor:
    """Contrastive Representation Distillation (Tian, Krishnan, Isola, ICLR 2020).

    Critic:  h = exp(<g_T, g_S>/tau) / ( exp(<g_T, g_S>/tau) + N/M )
    Loss:    -( E_pos[log h] + N * E_neg[log(1 - h)] )

    z_s      [B, D]     embedding student, đã L2-normalize
    z_t_pos  [B, D]     embedding teacher của CÙNG clip, đã L2-normalize
    z_t_neg  [B, N, D]  embedding teacher của N clip khác, đã L2-normalize
    """
    ...
```

> ⚠️ **BẢN TRÊN ĐÃ BỊ THAY. ĐỪNG IMPLEMENT NÓ.**
>
> Bản rút gọn ban đầu bỏ mất bước chuẩn hoá phân hoạch `Z` của CRD, khiến loss
> lúc khởi tạo đo được **9066.96** so với CE ≈ 3.58 — gấp 2531 lần. Đã đối chiếu
> `HobbitLong/RepDistiller` (`crd/memory.py` + `crd/criterion.py`): pipeline thật
> có **hai** bước, bản cũ chỉ có bước một.
>
> 1. `out = exp(⟨v, v'⟩ / T)`
> 2. `out = out / Z`, với `Z = out.mean() · n_data`, tính **một lần** ở batch đầu
>    rồi giữ cố định.
>
> Thiếu bước 2 thì giá trị vào critic là exponential thô chứ không phải xác suất,
> nên `log(1 − h_neg)` không nằm gần 0 và cộng 4096 số hạng thì nổ.
>
> **Không chia số hạng negative cho `n_neg`.** Repo gốc cộng qua negative và chỉ
> chia cho batch size — phần đó của plan vốn đã đúng. Chia cho `n_neg` mới là
> lệch paper; implement `Z` là quay về đúng paper.

Bản đúng, là một module vì nó phải giữ buffer `Z`:

```python
class CRDLoss(torch.nn.Module):
    """Contrastive Representation Distillation (Tian, Krishnan, Isola, ICLR 2020).

        P      = exp(<g_t, g_s> / tau) / Z        (Z cố định sau batch đầu)
        Pn     = 1 / n_data
        log_D1 = log( P_pos / (P_pos + m*Pn) )
        log_D0 = log( m*Pn  / (P_neg + m*Pn) )
        loss   = -(log_D1.sum() + log_D0.sum()) / batch_size
    """

    def __init__(self, n_data: int, tau: float = 0.07, eps: float = 1e-7) -> None:
        super().__init__()
        self.n_data, self.tau, self.eps = n_data, tau, eps
        self.register_buffer("Z", torch.tensor(-1.0))

    def forward(self, z_s, z_t_pos, z_t_neg):
        # z_s [B, D], z_t_pos [B, D], z_t_neg [B, m, D] — đều đã L2-normalize
        batch = z_s.shape[0]
        m = z_t_neg.shape[1]
        pos = torch.exp((z_s * z_t_pos).sum(-1, keepdim=True) / self.tau)
        neg = torch.exp(torch.bmm(z_t_neg, z_s.unsqueeze(-1)).squeeze(-1) / self.tau)

        if self.Z.item() < 0:
            with torch.no_grad():
                self.Z.fill_(torch.cat([pos, neg], dim=1).mean().item() * self.n_data)

        p_pos, p_neg = pos / self.Z, neg / self.Z
        pn = 1.0 / float(self.n_data)
        log_d1 = torch.log(p_pos / (p_pos + m * pn + self.eps))
        log_d0 = torch.log((m * pn) / (p_neg + m * pn + self.eps))
        return -(log_d1.sum() + log_d0.sum()) / batch
```

Độ lớn kỳ vọng lúc khởi tạo ≈ `log(m+1)` ≈ 8.3 với `m=4096` — cùng thang với
CE ≈ 3.58, nên `b=0.8` của repo CRD dùng được nguyên xi.

`import math` phải có ở đầu `mid_expert_lib.py`.

- [ ] **Step 8: Chạy test, xác nhận PASS**

Run: `python -m pytest test_mid_expert.py -v`
Expected: toàn bộ passed (11 test)

- [ ] **Step 9: Kiểm tra tỉnh táo về độ lớn của loss**

```python
def test_crd_loss_magnitude_is_not_dominated_by_negative_term():
    """N * E_neg có N=4096 số hạng; nếu nó nuốt mất CE thì b=0.8 là sai thang."""
    torch.manual_seed(0)
    z_s = _unit(torch.randn(4, 128))
    pos = _unit(torch.randn(4, 128))
    neg = _unit(torch.randn(4, 4096, 128))
    value = crd_loss(z_s, pos, neg, tau=0.07, dataset_size=30240)
    print("crd_loss at init:", float(value))
    assert torch.isfinite(value)
```

Run: `python -m pytest test_mid_expert.py -k magnitude -s -v`

**Đã chạy, 2026-09-24: đo được 9066.96 so với CE ≈ 3.58 — gấp 2531 lần.** Chốt dừng này đã bật.

Nguyên nhân **không phải** `b=0.8` sai, mà là bản `crd_loss` rút gọn trong plan thiếu bước chuẩn hoá phân hoạch `Z` của CRD. Đã đối chiếu `HobbitLong/RepDistiller`. Cách xử lý đúng là **implement `Z`** (quay về đúng paper), **không** chia số hạng negative cho `n_neg` (đó mới là lệch paper). Xem `CRDLoss` ở Step 7. Sau khi sửa, độ lớn kỳ vọng ≈ `log(m+1)` ≈ 8.3 — cùng thang với CE, nên `b=0.8` dùng được nguyên xi và **không phát sinh mục lệch-paper nào**.

---

### Task 5: Teacher — giai đoạn 1

**Files:**
- Modify: `mid_expert.ipynb`

**Interfaces:**
- Consumes: `train_beats_head.load_beats`, `initialize_head`, `metrics`; `noise_pipeline.mix_data.MixNoiseDataset`.
- Produces: file `teacher_noise_best.pt` (encoder blocks + head), và `teacher_bank.pt` chứa `{"z": [R, 768] float16, "logits": [R, 36] float16, "y": [R], "snr": [R], "split": [R], "sample_id": list[str]}` với `R` = tổng số hàng manifest, **thứ tự đúng bằng thứ tự hàng trong `manifest.csv`**.

- [ ] **Step 1: Dataset trả về noise sạch thay vì mixture**

```python
from torch.utils.data import Dataset

class NoiseOnlyDataset(Dataset):
    """Waveform noise sạch + nhãn. Dùng cho teacher ở giai đoạn 1."""

    def __init__(self, root: Path, rows: list[dict], labels: list[str]):
        self.root, self.rows = Path(root), rows
        self.label_to_index = {name: i for i, name in enumerate(labels)}

    def __len__(self): return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        return {
            "audio": load_float_audio(self.root / row["noise_path"], 16_000, 64_000),
            "target": self.label_to_index[row["label_names"]],
            "snr": int(float(row["target_snr_db"])),
        }
```

- [ ] **Step 2: Dedup theo `noise_path` cho tập train của teacher**

```python
def dedup_rows(rows, key="noise_path"):
    seen, out = set(), []
    for r in rows:
        if r[key] not in seen:
            seen.add(r[key]); out.append(r)
    return out

teacher_train = dedup_rows([r for r in rows if r["split"] == "train"])
teacher_val   = dedup_rows([r for r in rows if r["split"] == "validation"])
print("teacher train", len(teacher_train), "val", len(teacher_val))
```

Kỳ vọng ≈ 5.040 / 1.080 nếu Task 1 Step 2 cho ratio ≈ 6. **Nếu ratio ≈ 1 thì bỏ dedup** và dùng thẳng toàn bộ hàng.

- [ ] **Step 3: Train head trên embedding đóng băng, rồi finetune 12 block**

Dùng lại đúng recipe của repo, chỉ đổi nguồn audio từ `mixture` sang `noise`:
1. `load_beats(device, pretrained)` → model đóng băng.
2. Trích embedding `sequence.mean(dim=1)` cho `teacher_train` / `teacher_val`, cache lại.
3. `initialize_head(checkpoint, labels, label_to_mid, device)` — warm-start head 36 lớp từ hàng predictor AudioSet, **không** khởi tạo ngẫu nhiên.
4. Train head: `epochs=50, lr=1e-3, batch=512, patience=10`, chọn theo macro-F1 validation.
5. Finetune 12 block cuối: `epochs=8, head_lr=1e-4, encoder_lr=1e-5, batch=32`, bf16, clip 5.0.

- [ ] **Step 4: In acc teacher và đối chiếu**

```python
print(f"TEACHER val_accuracy={m['accuracy']:.4f} val_macro_f1={m['macro_f1']:.4f}")
```

Đây là **trần trên** của cả hướng tiếp cận. Ghi vào báo cáo.

Nếu teacher acc **> 0.95**: soft label gần one-hot, KD sẽ không tải được thông tin gì (rủi ro §10 của spec). Xử lý: tăng `rho_kd` lên 8 và ghi lại lý do. Không lặng lẽ bỏ qua.
Nếu teacher acc **< 0.75**: teacher không mạnh hơn baseline mixture bao nhiêu ⇒ toàn bộ tiền đề privileged-information lung lay. **Dừng lại, báo cáo, hỏi trước khi chạy tiếp Task 6.**

- [ ] **Step 5: Precompute bank teacher cho TOÀN BỘ hàng manifest**

Không dedup ở bước này — student cần tra theo từng hàng.

```python
@torch.inference_mode()
def build_bank(model, head, rows, root, device, batch_size=64, workers=4):
    ds = NoiseOnlyDataset(root, rows, labels)
    loader = DataLoader(ds, batch_size=batch_size, num_workers=workers,
                        shuffle=False, pin_memory=True)
    zs, lg, ys, snrs = [], [], [], []
    for batch in loader:
        seq, _ = model.extract_features(batch["audio"].to(device))
        z = seq.mean(dim=1).float()
        zs.append(z.half().cpu())
        lg.append(head(z).half().cpu())
        ys.append(batch["target"]); snrs.append(batch["snr"])
    return {"z": torch.cat(zs), "logits": torch.cat(lg),
            "y": torch.cat(ys), "snr": torch.cat(snrs),
            "sample_id": [r["sample_id"] for r in rows]}
```

Chạy trên `rows` đầy đủ (43.200). Bank ≈ 43.200 × 768 × 2 B ≈ 66 MB. Lưu `teacher_bank.pt`.

- [ ] **Step 6: Xác minh bank căn đúng hàng**

```python
bank = torch.load(OUT / "teacher_bank.pt")
assert bank["z"].shape[0] == len(rows)
assert bank["sample_id"] == [r["sample_id"] for r in rows]
manifest_y = torch.tensor([labels.index(r["label_names"]) for r in rows])
assert torch.equal(bank["y"], manifest_y), "bank lệch hàng so với manifest"
print("bank OK", bank["z"].shape, bank["logits"].shape)
```

Assert này bắt buộc. Bank lệch một hàng thì CRD học trên cặp sai mà **không hề báo lỗi** — chỉ ra kết quả tồi và ta sẽ đổ oan cho phương pháp.

---

### Task 6: Student run 2 — chỉ CE, full SNR, có FiLM

Control quan trọng nhất. Trả lời: bao nhiêu phần cải thiện chỉ đến từ việc có thêm data.

**Files:**
- Modify: `mid_expert.ipynb`

**Interfaces:**
- Consumes: `FiLM`, `mid_slice_mask`, `film_deviation`, `metrics`.
- Produces: `student_run2.pt`, `run2_metrics.json`; hàm `train_student(cfg, use_kd, use_crd, use_remix)` dùng lại cho Task 7 và 8.

- [ ] **Step 1: Model student = BEATs + FiLM + head**

```python
class StudentModel(torch.nn.Module):
    def __init__(self, beats, head, film):
        super().__init__()
        self.beats, self.head, self.film = beats, head, film

    def forward(self, wav, snr_db):
        seq, _ = self.beats.extract_features(wav)
        z = seq.mean(dim=1)
        z = self.film(z, snr_db) if self.film is not None else z
        return self.head(z), z
```

`forward` trả cả `z` vì Task 7 cần nó cho CRD.

- [ ] **Step 2: Dataset trả thêm chỉ số hàng manifest**

```python
class MixtureIndexedDataset(Dataset):
    """Mixture + nhãn + SNR + row_index, để tra vào bank teacher."""

    def __init__(self, root, rows, labels, row_index):
        self.root, self.rows, self.row_index = Path(root), rows, row_index
        self.label_to_index = {n: i for i, n in enumerate(labels)}

    def __len__(self): return len(self.rows)

    def __getitem__(self, i):
        row = self.rows[i]
        return {
            "mixture": load_float_audio(self.root / row["mixture_path"], 16_000, 64_000),
            "target": self.label_to_index[row["label_names"]],
            "snr": float(row["target_snr_db"]),
            "row_index": self.row_index[i],
        }
```

`row_index` là vị trí của hàng trong `manifest.csv` đầy đủ — chính là chỉ số vào `bank["z"]`.

- [ ] **Step 3: Jitter SNR lúc train**

```python
def jittered_snr(snr_db, sigma_db, generator):
    """Mô phỏng sai số của gate lúc infer. sigma=2.0 dB là GIẢ ĐỊNH (spec §3.2)."""
    noise = torch.randn(snr_db.shape, generator=generator) * sigma_db
    return snr_db + noise.to(snr_db.device)
```

Chỉ áp dụng lúc train. Lúc validate/test dùng SNR thật (oracle), và ghi rõ trong báo cáo rằng con số báo cáo là **oracle-SNR**, chưa tính lỗi gate.

- [ ] **Step 4: Vòng train, chọn checkpoint theo macro-F1 LÁT MID**

```python
def evaluate_mid(model, loader, device, labels, band=(5.0, 10.0)):
    model.eval()
    L, Y, S = [], [], []
    with torch.inference_mode():
        for b in loader:
            logits, _ = model(b["mixture"].to(device), b["snr"].float().to(device))
            L.append(logits.float().cpu()); Y.append(b["target"]); S.append(b["snr"])
    L, Y, S = torch.cat(L), torch.cat(Y), torch.cat(S)
    full = metrics(L, Y, S.long(), labels)
    mask = mid_slice_mask(S, *band)
    mid = metrics(L[mask], Y[mask], S[mask].long(), labels)
    return {"full": full, "mid": mid}
```

Chọn theo `result["mid"]["macro_f1"]`. Đây là chỗ duy nhất việc chuyên biệt hoá cho mid đi vào quy trình chính.

- [ ] **Step 5: Chạy run 2**

CONFIG: `run="run2_ce_only"`, `loss.a_kd=0.0`, `loss.b_crd=0.0`, `remix.enabled=False`, `film.enabled=True`.

Mỗi epoch in: `train_loss`, `val_mid_acc`, `val_mid_macro_f1`, `film_deviation`.

- [ ] **Step 6: Ghi kết quả và so mốc**

```python
BASELINE = {"acc": 0.6681, "macro_f1": 0.6602}
OLD_MID  = {"acc": 0.6616, "macro_f1": 0.6547}
r = test_result["mid"]
print(f"run2 mid acc={r['accuracy']:.4f} (baseline {BASELINE['acc']:.4f}, "
      f"delta {r['accuracy']-BASELINE['acc']:+.4f})")
print(f"run2 mid f1 ={r['macro_f1']:.4f} (baseline {BASELINE['macro_f1']:.4f}, "
      f"delta {r['macro_f1']-BASELINE['macro_f1']:+.4f})")
```

Lưu `run2_metrics.json`. **Nếu `film_deviation` gần 0 suốt quá trình** thì FiLM đã sụp về identity — ghi nhận, và run 2 thực chất là "baseline train lại", vẫn là control hợp lệ.

---

### Task 7: Student run 3 — thêm KD và CRD

**Files:**
- Modify: `mid_expert.ipynb`

**Interfaces:**
- Consumes: `teacher_bank.pt`, `Projection`, `CRDLoss`, `sample_negatives`, `train_student` từ Task 6.
- Produces: `student_run3.pt`, `run3_metrics.json`.

- [ ] **Step 1: Nạp bank lên GPU một lần**

```python
bank = torch.load(OUT / "teacher_bank.pt")
bank_z      = bank["z"].to(device).float()        # [R, 768]
bank_logits = bank["logits"].to(device).float()   # [R, 36]
bank_y      = bank["y"]                            # [R], giữ trên CPU để lấy mẫu
DATASET_SIZE = bank_z.shape[0]
```

43.200 × 768 float32 ≈ 133 MB trên GPU. Chấp nhận được. Nếu thiếu VRAM thì giữ `bank_z` ở half.

- [ ] **Step 2: Chiếu bank qua `g_T` một lần mỗi epoch, không mỗi batch**

`g_T` có học, nên đầu ra đổi theo epoch. Chiếu lại toàn bank mỗi đầu epoch và cache — rẻ hơn nhiều so với chiếu 4096 negative mỗi batch.

```python
@torch.no_grad()
def refresh_bank_projection(g_t, bank_z, chunk=8192):
    outs = [g_t(bank_z[i:i+chunk]) for i in range(0, bank_z.shape[0], chunk)]
    return torch.cat(outs)   # [R, 128], đã L2-normalize
```

**Cảnh báo:** cache này stale trong phạm vi một epoch. CRD gốc cũng chịu stale (memory buffer cuốn chiếu), nên đây không phải lệch khỏi paper — nhưng phải ghi rõ trong báo cáo là "bank teacher tĩnh, phần chiếu làm mới mỗi epoch", đừng viết là "hoàn toàn không stale".

- [ ] **Step 3: Số hạng loss trong vòng train**

```python
logits_s, z_s = model(wav, snr_in)
ce = F.cross_entropy(logits_s, target)

t_logits = bank_logits[row_index]
kd = F.kl_div(
    F.log_softmax(logits_s / rho, dim=-1),
    F.softmax(t_logits / rho, dim=-1),
    reduction="batchmean",
) * (rho ** 2)

p_s   = g_s(z_s)                       # [B, 128]
p_pos = bank_proj[row_index]           # [B, 128]
neg_idx = sample_negatives(target.cpu(), bank_y, n_neg, gen, mode).to(device)
p_neg = bank_proj[neg_idx]             # [B, n_neg, 128]
crd = crd_criterion(p_s, p_pos, p_neg)   # CRDLoss(n_data=DATASET_SIZE, tau=tau), tạo 1 lần trước vòng train

loss = r_ce * ce + a_kd * kd + b_crd * crd
```

`g_t` chỉ được cập nhật qua `refresh_bank_projection` ⇒ để `g_t` **có gradient** thì phải chiếu `p_pos` trực tiếp thay vì lấy từ cache. Quyết định: **đóng băng `g_t` hoàn toàn** (chỉ `g_s` học). CRD gốc học cả hai; đóng băng một bên là đơn giản hoá.

**Đây là lệch khỏi paper — phải ghi vào báo cáo.** Lý do: teacher tĩnh nên `g_t` không có tín hiệu nào ngoài chính contrastive loss, và việc chiếu lại 43.200 vector có gradient mỗi batch là không khả thi. Nếu còn compute, ablation: cho `g_t` học bằng cách chiếu lại chỉ `p_pos` (batch nhỏ) mỗi bước, còn negative vẫn lấy từ cache.

- [ ] **Step 4: In tách riêng ba số hạng mỗi epoch**

```python
print(f"epoch={e} ce={ce_avg:.4f} kd={kd_avg:.4f} crd={crd_avg:.4f} "
      f"val_mid_f1={vm:.4f} film_dev={fd:.3f}")
```

Bắt buộc. Nếu `crd` không giảm thì phần contrastive không học được gì, và mọi cải thiện là do KD + data — phải biết điều đó trước khi viết kết luận.

- [ ] **Step 5: Chạy run 3 và so với run 2**

CONFIG: `run="run3_kd_crd"`, `a_kd=1.0`, `b_crd=0.8`, `rho_kd=4.0` (hoặc 8.0 nếu Task 5 Step 4 bắt được teacher > 0.95).

```python
print(f"run3 - run2: acc {r3['accuracy']-r2['accuracy']:+.4f} "
      f"f1 {r3['macro_f1']-r2['macro_f1']:+.4f}")
```

Nếu chênh lệch nhỏ hơn ~0.005 thì coi như CRD+KD không đóng góp. Đó là kết quả hợp lệ. **Không quét siêu tham số cho tới khi ra số đẹp** (spec §10).

---

### Task 8: Student run 4 — remix augmentation

**Chỉ chạy nếu Task 1 Step 4 cho `LINEAR_OK = True`.** Nếu False, bỏ task này, ghi lý do vào báo cáo, sang thẳng Task 9.

**Files:**
- Modify: `mid_expert_lib.py` (hàm thuần `remix_at_snr`), `mid_expert.ipynb`
- Test: `test_mid_expert.py`

**Interfaces:**
- Produces: `mid_expert_lib.remix_at_snr(speech, noise, target_snr_db) -> Tensor`; `student_run4.pt`, `run4_metrics.json`.

- [ ] **Step 1: Viết failing test**

```python
from mid_expert_lib import remix_at_snr


def _snr_db(sig, noi):
    return 10 * torch.log10(sig.pow(2).mean() / noi.pow(2).mean())


def test_remix_produces_requested_snr():
    torch.manual_seed(0)
    speech = torch.randn(16_000)
    noise = torch.randn(16_000) * 3.0
    for target in [3.0, 5.0, 7.5, 10.0, 12.0]:
        mixed, scaled_noise = remix_at_snr(speech, noise, target)
        assert abs(_snr_db(speech, scaled_noise).item() - target) < 0.05
        assert torch.allclose(mixed, speech + scaled_noise, atol=1e-6)


def test_remix_handles_silent_noise_without_nan():
    speech = torch.randn(1000)
    mixed, scaled = remix_at_snr(speech, torch.zeros(1000), 5.0)
    assert torch.isfinite(mixed).all() and torch.isfinite(scaled).all()
```

- [ ] **Step 2: Chạy test, xác nhận FAIL**

Run: `python -m pytest test_mid_expert.py -k remix -v`
Expected: FAIL — `cannot import name 'remix_at_snr'`

- [ ] **Step 3: Implementation**

```python
def remix_at_snr(speech: Tensor, noise: Tensor, target_snr_db: float,
                 eps: float = 1e-10) -> tuple[Tensor, Tensor]:
    """Scale noise để hỗn hợp đạt đúng SNR yêu cầu. Trả (mixture, noise đã scale).

    Hệ số tính từ RMS THỰC ĐO của hai nguồn, không giả định mức nào sẵn.
    """
    speech_power = speech.pow(2).mean()
    noise_power = noise.pow(2).mean()
    target = torch.as_tensor(target_snr_db, dtype=speech.dtype)
    gain = torch.sqrt(speech_power / (noise_power + eps) / (10.0 ** (target / 10.0)))
    scaled = noise * gain
    return speech + scaled, scaled
```

- [ ] **Step 4: Chạy test, xác nhận PASS**

Run: `python -m pytest test_mid_expert.py -k remix -v`
Expected: 2 passed

- [ ] **Step 5: Đưa remix vào dataset**

Với xác suất `remix.prob = 0.5`, thay mixture gốc bằng bản remix ở `snr ~ U(3, 12)`:

```python
speech = mixture - noise
new_snr = rng.uniform(3.0, 12.0)
mixture, _ = remix_at_snr(speech, noise, new_snr)
snr_for_film = new_snr        # QUAN TRỌNG: FiLM phải nhận SNR MỚI
```

`row_index` giữ nguyên — nhãn và noise sạch không đổi, nên cặp CRD và soft label KD vẫn hợp lệ.

- [ ] **Step 6: Chạy run 4 và so với run 3**

CONFIG: `run="run4_remix"`, `remix.enabled=True`, mọi thứ khác giữ như run 3.

---

### Task 9: Báo cáo và ablation

**Files:**
- Modify: `mid_expert.ipynb`

**Interfaces:**
- Consumes: `run2_metrics.json`, `run3_metrics.json`, `run4_metrics.json` (nếu có), teacher metrics.
- Produces: `mid_expert_report.json` + bảng in ra.

- [ ] **Step 1: Bảng tổng hợp**

Mỗi dòng một run, mỗi cột một metric, **tách riêng 5 dB và 10 dB**, luôn có hai dòng mốc:

| Run | acc@5 | acc@10 | acc mid | F1 mid | Δacc vs baseline |
|---|---|---|---|---|---|
| BEATs baseline | 0.6806 | 0.6556 | 0.6681 | 0.6602 | — |
| Mid expert cũ | 0.6787 | 0.6444 | 0.6616 | 0.6547 | −0.0065 |
| Teacher (noise sạch) | — | — | — | — | trần trên |
| Run 2 (CE, full data) | | | | | |
| Run 3 (+KD+CRD) | | | | | |
| Run 4 (+remix) | | | | | |

- [ ] **Step 2: F1 theo lớp, so với baseline**

Bốn lớp yếu nhất ở mid theo baseline: Hi-hat 0.321, Drawer open or close 0.373, Microwave oven 0.462, Squeak 0.474. Nếu CRD có tác dụng thật thì phải thấy cải thiện tập trung ở đúng nhóm này — tín hiệu từ noise sạch giúp nhiều nhất ở chỗ speech che lấp nặng nhất. In riêng delta cho 4 lớp này.

- [ ] **Step 3: Danh sách những chỗ đã lệch khỏi paper**

Gom vào một cell markdown để dán thẳng vào báo cáo. Tính đến giờ đã biết bốn chỗ:
1. FiLM đặt một lần trên embedding đã pool, không per-layer như paper gốc (Task 3).
2. `g_t` đóng băng, chỉ `g_s` học; CRD gốc học cả hai (Task 7 Step 3).
3. Phần chiếu bank làm mới mỗi epoch ⇒ stale trong phạm vi epoch (Task 7 Step 2).
4. Nếu Task 4 Step 9 buộc phải chuẩn hoá lại số hạng negative thì đó là chỗ thứ tư.

Cộng với bốn chỗ ngoại suy đã liệt ở §2 của spec. Trong báo cáo, mọi mục ở đây viết "theo tinh thần của", không viết "như đã chứng minh".

- [ ] **Step 4: Ablation nếu còn compute**

Theo thứ tự đáng làm (spec §7): (a) `b_crd=0` giữ KD; (b) `negative_mode="random"`; (c) `w(snr)` tam giác đặt tay; (d) quét `rho_kd ∈ {2, 4, 8}`.

---

## Self-Review

**Spec coverage.** §1 → Task 9 Step 1 (bảng mốc). §2 → Task 9 Step 3 (danh sách lệch paper). §3.1 teacher + dedup → Task 5. §3.2 student + FiLM + jitter → Task 3, Task 6. §4 loss + siêu tham số → Task 2 CONFIG, Task 4, Task 7. §4.1 bank tĩnh + negative khác nhãn → Task 4, Task 7. §5 full data + chọn theo lát mid → Task 6 Step 4. §5.1 check tuyến tính → Task 1 (và đã sửa: check trong spec là tautology, bản dùng được là so SNR đo với `target_snr_db`). §6 remix → Task 8. §7 chuỗi run → Task 6, 7, 8 + Task 9 Step 4. §8 đánh giá → Task 9 Step 1–2. §9 giao hàng → Global Constraints. §10 rủi ro → Task 5 Step 4 (teacher quá mạnh / quá yếu), Task 6 Step 6 (FiLM sụp), Task 7 Step 5 (CRD = 0), Task 1 (LINEAR_OK).

**Đã sửa inline so với spec:**
- §5.1 của spec kiểm tra `‖mix − (speech + noise)‖` với `speech := mix − noise` — luôn bằng 0 về mặt đại số, không kiểm tra được gì. Thay bằng so SNR đo được với `target_snr_db` (Task 1 Step 4).
- §3.2 nói FiLM đè lên 3 block cuối; đổi thành pooled embedding, ghi là lệch paper (Task 3).
- §4.1 nói bank "không stale"; đúng với `z` nhưng không đúng với phần chiếu `g_t`, đã sửa lại cách diễn đạt (Task 7 Step 2).

**Type consistency.** `mid_slice_mask(snr_db, low, high)`, `normalize_snr(snr_db)`, `FiLM(dim, hidden).forward(z, snr_db)`, `film_deviation(film)`, `Projection(in_dim, out_dim).forward(x)`, `sample_negatives(labels, bank_labels, n, generator, mode)`, `CRDLoss(n_data, tau).forward(z_s, z_t_pos, z_t_neg)`, `remix_at_snr(speech, noise, target_snr_db) -> (mixture, scaled_noise)` — dùng nhất quán ở Task 3, 4, 6, 7, 8. Bank keys `z / logits / y / snr / sample_id` nhất quán giữa Task 5 Step 5, Step 6 và Task 7 Step 1.

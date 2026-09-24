# Cấu trúc các phiên bản (run)

Năm phiên bản, chọn bằng `--run`, không sửa file config (lý do: config nằm trong git,
sửa tay sẽ mất sau mỗi `git pull` — xem `STATUS.md`).

```bash
python -u main.py student36 --config config/train_config.json --run <ten>
python -u main.py test36    --config config/train_config.json --run <ten>
```

## Chung cho cả năm phiên bản

| | |
|---|---|
| Teacher | Cùng một checkpoint (BEATs trên `noise_path`, acc ≈ 0.78) |
| Dữ liệu train | 30.240 clip, **toàn bộ 6 mức SNR** (không chỉ mid) |
| Kiến trúc student | BEATs pretrained, 12 block cuối trainable |
| Learning rate | encoder 1e-5, head 1e-4 |
| Batch size | 32, `accumulation_steps=1` |
| Seed | 2026 |
| Đánh giá | test 6.480 clip, tách theo `full` / `mid` (5–10 dB) / từng mức SNR |

Năm phiên bản khác nhau **đúng ở bốn công tắc**: `a_kd`, `b_crd`, `FiLM`, và slice dùng
để chọn checkpoint (`select_on`).

## Bảng cấu hình

| `--run` | `a_kd` | `b_crd` | FiLM | Chọn checkpoint theo |
|---|:-:|:-:|:-:|:-:|
| `run1_baseline` | 0 | 0 | **tắt** | **toàn bộ validation** |
| `run2_ce_only` | 0 | 0 | bật | lát mid |
| `run3b_crd_only` | 0 | **0.8** | bật | lát mid |
| `run3c_kd_only` | **1.0** | 0 | bật | lát mid |
| `run3_kd_crd` | **1.0** | **0.8** | bật | lát mid |
| `run4_remix` *(chưa chạy)* | 1.0 | 0.8 | bật | lát mid, + remix augmentation |

## Bốn công tắc là gì

**`a_kd`** (trọng số KD) — bật thì student bắt chước phân phối softmax của teacher trên
cùng clip đó, ở nhiệt độ `rho=4`. Teacher nhìn noise **sạch**, student nhìn mixture. Kênh
truyền "đáp án đã làm mềm".

**`b_crd`** (trọng số CRD) — bật thì kéo embedding của mixture về gần embedding noise
sạch **cùng clip**, đẩy xa 4096 clip khác nhãn (Contrastive Representation Distillation,
Tian et al. ICLR 2020). Kênh truyền **biểu diễn** — cơ chế "đảo ngược separation" mà
thiết kế ban đầu xây quanh.

**FiLM** — MLP sinh `(γ, β)` từ SNR rồi điều biến embedding đã pool. Cho model biết nó
đang ở mức nhiễu nào.

**`select_on`** — `full` chọn epoch tốt nhất trên toàn bộ validation (đúng công thức
baseline gốc); `mid` chỉ nhìn lát 5–10 dB (chuyên biệt hoá cho nhánh mid được giao).

## Vì sao có `run1_baseline`

Baseline công bố (`BEATs_Experts/checkpoint/test_metrics_36.json`, acc 0.6681 trên lát
mid) đến từ một lượt train **khác**, không có dự đoán từng clip, và checkpoint của nó đã
mất khỏi molab (`*.pt` không đi theo git). Không so cặp (McNemar) được với nó, và không
kiểm soát được phương sai giữa các lượt train.

`run1_baseline` dựng lại đúng công thức baseline (CE thuần, không FiLM, chọn theo toàn
bộ validation) **ngay trong pipeline này**, cùng teacher với bốn phiên bản kia, đi qua
cùng `test36`. Nó đạt **0.6736** trên lát mid — cao hơn baseline công bố 0.55 điểm, dù
cùng một công thức. Đó chính là phương sai giữa các lượt train (đã đo riêng là ~0.70
điểm) — không có `run1_baseline` thì mọi so sánh sau này đều lẫn cả phương sai đó vào
kết luận.

## Kết quả cuối (lát mid 5–10 dB, 2.160 clip)

| Phiên bản | Top-1 | Macro-F1 | mAP | Macro-AUC |
|---|---|---|---|---|
| beats_baseline (công bố, không so cặp được) | 0.6681 | 0.6602 | – | – |
| **run1_baseline** | **0.6736** | 0.6652 | **0.7512** | **0.9712** |
| run2_ce_only | 0.6653 | 0.6560 | 0.7451 | 0.9704 |
| run3b_crd_only | 0.6690 | 0.6619 | 0.7472 | 0.9697 |
| run3c_kd_only | 0.6745 | 0.6651 | 0.7464 | 0.9697 |
| run3_kd_crd | 0.6759 | 0.6678 | 0.7382 | 0.9678 |

McNemar so với `run1_baseline`: không phiên bản nào vượt có ý nghĩa thống kê ở lát mid
(`run3_kd_crd` p=0.757). Nhưng xét theo xu hướng trên cả sáu mức SNR, mức lợi của
`run3_kd_crd` so với baseline **tăng có ý nghĩa thống kê theo SNR** (Spearman ρ=0.943,
p=0.0048), đạt đỉnh +2.13 điểm ở 20 dB (McNemar p=0.0225 riêng lẻ). Phương pháp có tác
dụng thật — ở nhánh high, không phải nhánh mid.

Chi tiết đầy đủ, các bảng số liệu, và diễn giải: xem `STATUS.md`.

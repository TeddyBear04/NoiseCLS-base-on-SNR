# Cấu trúc các phiên bản (run) — nhánh high SNR (15–20 dB)

Tám phiên bản, chọn bằng `--run` (sáu đầu đã có code chạy được; hai cuối còn chờ §3–§5
của spec được duyệt trước khi viết switch tương ứng — xem `STATUS.md`).

```bash
python -u main.py student36 --config config/train_config_high.json --run <ten>
python -u main.py test36    --config config/train_config_high.json --run <ten>
```

## Vì sao dự án này đổi từ mid sang high

`../Mid_Expert/STATUS.md` kết luận: teacher–student distillation từ noise sạch **không
vượt baseline có ý nghĩa thống kê** ở dải 5–10 dB (McNemar p = 0.757), nhưng đo theo từng
mức SNR thì hiệu ứng **tăng đơn điệu theo SNR** — Spearman(SNR, Δ) = 0.943, p = 0.0048 —
và đạt ý nghĩa ở 20 dB (+2.13 điểm, p = 0.0225). Cơ chế khớp: SNR càng cao, speech càng
át noise, mixture càng khác noise sạch, và teacher (nhìn noise sạch, acc phẳng mọi SNR)
càng có nhiều thứ để dạy. Dự án này chuyển hẳn phương pháp sang nơi nó thật sự có tác
dụng: **dải 15–20 dB**, thay cho `DPCRN_Noise_Target` (đang 0.144, sập vì resynthesis
artifact — SI-SDR noise ước lượng ở 20 dB là −2.09 dB, xem `Fusion` v3 trong Result.xlsx).

## Mốc phải vượt

`Result.xlsx`, sheet "Theo tung SNR", hàng `BEATs-Mixture` (công bố):

| | 15 dB | 20 dB | gộp (band, trọng số support bằng nhau) |
|---|---|---|---|
| Top-1 accuracy | 0.5769 | 0.4861 | **0.5315** |
| Macro-F1 | 0.5757 | 0.4830 | **0.52935** |

Đây **không phải** file `BEATs_Experts/checkpoint/test_metrics_36.json` trên đĩa (checkpoint
đó cho 0.6102/0.5315 ở 15/20 dB — một lượt train khác, không có dự đoán từng clip nên
không so cặp McNemar được). Bài học lặp lại đúng những gì `Mid_Expert/STATUS.md` đã học:
gate chốt theo số **công bố**, `run1_baseline` dựng lại chỉ dùng làm control nội bộ.

Trần trên tham khảo (BEATs-NoiseOnly, oracle nhìn noise sạch): 0.7611 / 0.7519, gộp 0.7565.

## Chung cho cả tám phiên bản

| | |
|---|---|
| Backbone | SSLAM (`ta012/SSLAM_pretrain`, ICLR 2025), thay BEATs — xem `models/sslam.py` |
| Teacher | Cùng một checkpoint (SSLAM finetune trên `noise_path`, band 15–20 dB) |
| Dữ liệu train | 10.080 clip lát high (2 mức × ~5.040), student train trên **toàn bộ** 6 mức SNR |
| CRD | Mức patch (`crd_level: "patch"`), không pooled — xem README cũ để biết lý do |
| Test | 2.160 clip (1.080 mỗi mức 15/20 dB), khớp support trong Result.xlsx |
| Seed | 2026 |
| Đánh giá | test 6.480 clip, tách theo `full` / `band` (15–20 dB) / từng mức SNR |

Sáu phiên bản đầu khác nhau đúng ở bốn công tắc: `a_kd`, `b_crd`, `FiLM`, `select_on`
(giống thang gốc của `Mid_Expert`, chỉ đổi band). Hai phiên bản cuối thêm cơ chế mới,
nhắm thẳng vào nguyên nhân high-SNR khó: speech át noise trong mixture.

## Bảng cấu hình

| `--run` | `a_kd` | `b_crd` | FiLM | Chọn checkpoint theo | Cơ chế thêm | Trạng thái |
|---|:-:|:-:|:-:|:-:|---|:-:|
| `run1_baseline` | 0 | 0 | **tắt** | toàn bộ validation | — | ✅ code sẵn |
| `run2_ce_only` | 0 | 0 | bật | lát band | — | ✅ code sẵn |
| `run3b_crd_only` | 0 | 0.8 | bật | lát band | CRD (Tian et al., ICLR 2020) | ✅ code sẵn |
| `run3c_kd_only` | 1.0 | 0 | bật | lát band | KD (soft label, ρ=4) | ✅ code sẵn |
| `run3_kd_crd` | 1.0 | 0.8 | bật | lát band | KD + CRD | ✅ code sẵn |
| `run5_remix` | 1.0 | 0.8 | bật | lát band | + remix augmentation | ⬜ chờ duyệt §3 |
| `run6_grl` | 1.0 | 0.8 | bật | lát band | + head phụ đoán speech, gradient reversal | ⬜ chờ duyệt §4 |

`run4_remix` là tên cũ trong `Mid_Expert`, ở đây đổi số vì band đã cắm vào giữa thang.

## Phương pháp của từng run

**`run1_baseline`** — CE thuần, KHÔNG FiLM, không KD, không CRD. Chọn checkpoint theo
toàn bộ tập validation (không thiên vị lát high). Đây là baseline dựng lại trong đúng
điều kiện train của các run khác — dùng để so cặp McNemar, KHÔNG dùng để chốt "vượt".

**`run2_ce_only`** — thêm FiLM (điều biến embedding theo SNR) và chọn checkpoint theo
riêng lát 15–20 dB. Không dùng thông tin đặc quyền nào (`noise_path`). Control cho
`run3*` — tách được phần cải thiện đến từ FiLM/checkpoint-selection khỏi phần đến từ
distillation.

**`run3b_crd_only`** — thêm Contrastive Representation Distillation: kéo embedding
mixture (patch-level) về gần embedding noise sạch cùng clip, đẩy xa negative khác nhãn.
Setting chính của paper CRD gốc (`-a 0 -b 0.8`).

**`run3c_kd_only`** — thêm Knowledge Distillation: student bắt chước phân phối softmax
đã làm mềm (ρ=4) của teacher trên cùng clip. Không có CRD.

**`run3_kd_crd`** — cả hai kênh cùng lúc. Phương pháp đầy đủ đã dùng ở nhánh mid, retarget
sang band 15–20 dB. Đây là phiên bản có bằng chứng thống kê mạnh nhất ủng hộ nó hoạt động
ở SNR cao (xem phần "Vì sao đổi" ở trên).

**`run5_remix`** *(chưa code)* — `run3_kd_crd` + tăng dữ liệu bằng remix: sinh cặp
`(clean + gain·noise)` mới trên band 15–20 dB, `gain` tính từ RMS thực đo (đã verify
`mixture = clean + noise`, sai số tối đa 5.96e-08 trên 200 clip lát high — xem log pair
với molab 2026-09-28). Tăng đúng chỗ band 15–20 dB đang thiếu dữ liệu nhất so với các
band khác không remix được (`noise_scale`/`post_gain` đã nướng sẵn cho 6 mức cố định).

**`run6_grl`** *(chưa code)* — `run3_kd_crd` + một head phụ trên embedding, đoán
speech/speaker, gradient đảo dấu trước khi lan vào backbone (domain-adversarial training,
Ganin & Lempitsky, ICML 2015). Ép embedding vứt bỏ thông tin speech thay vì chỉ kéo về
phía noise sạch — tấn công trực tiếp nguyên nhân high-SNR khó (speech át noise), khác cơ
chế với CRD/KD.

## Sáu điều đã xác minh trước khi viết (pair với molab, 2026-09-28)

1. SSLAM load được trên molab: `EATModel`, 90.0M tham số, `transformers 4.57.6`
   (bắt buộc `<5`) + `timm 1.0.30` + `torchaudio` cu130.
2. **`torchvision` phải ép bản `+cu130`** qua
   `--index-url https://download.pytorch.org/whl/cu130 --reinstall-package torchvision`
   — bản PyPI mặc định làm `timm` chết với `torchvision::nms does not exist`.
3. Forward pass thật: `mel (4,1,398,128) → tokens (4,193,768)`.
4. Dataset đã có sẵn trên molab (`/marimo/dataset/mix-dataset`, 43.200 hàng, đúng 6 mức
   SNR × 7.200 hàng).
5. Lát high (15+20 dB) trên molab: train 10.080 / validation 2.160 / test 2.160 — khớp
   support của `Result.xlsx`.
6. GPU: RTX PRO 6000 Blackwell, **102 GB VRAM** — batch size hiện tại trong
   `config/train_config_high.json` (16, accumulation 2) đặt cho GPU nhỏ hơn nhiều; có thể
   nâng khi vào giai đoạn implementation, không đổi ở bước design này.

## Nhật ký chạy

### 2026-09-28 — lượt 1: teacher `SSLAM_pretrain` dừng ở gate ⛔

| | val acc (noise sạch) |
|---|---|
| Teacher BEATs (`Mid_Expert`, AS2M-finetuned) | 0.7799 |
| **Teacher SSLAM `ta012/SSLAM_pretrain`** | **0.6576** — dưới ngưỡng 0.75, GATE=STOP |

Đường cong: val đạt đỉnh ngay epoch 1 (0.6537) rồi đi ngang tới epoch 7, trong khi train acc
lên 1.0000 từ epoch 5. Finetune không thêm gì — trần nằm ở chất lượng đặc trưng.

**Nguyên nhân: nhầm loại checkpoint.** Bản BEATs dùng
`BEATs_iter3_plus_AS2M_finetuned_on_AS2M_cpt2` — **đã finetune có giám sát** trên AudioSet.
`SSLAM_pretrain` chỉ self-supervised. Con số 0.502 mAP đưa SSLAM lên đầu bảng là của bản
**`ta012/SSLAM_AS2M_Finetuned`** (có trên HF, đã verify trên molab: cùng
`extract_features → (B, 193, 768)`, thêm head 527 lớp). So pretrain với finetuned là so
khập khiễng; đổi sang bản finetuned cho công bằng với BEATs.

Không student nào được chạy — gate làm đúng việc: teacher yếu thì KD/CRD dạy nhiễu.

### Thay đổi sau lượt 1

- `backbone.model_id` → `ta012/SSLAM_AS2M_Finetuned`.
- `evaluation.band_only = true`: validation (chọn checkpoint) và test **chỉ** dùng hàng
  15–20 dB, theo yêu cầu chỉ tập trung nhánh high. Train vẫn dùng đủ 6 mức SNR.
- Quy tắc tinh chỉnh: mọi quyết định chỉnh tham số dựa trên **validation** lát high; test
  chỉ để báo cáo cấu hình đã chọn.

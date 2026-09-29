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
| `run7_attn` | 0 | 0 | **tắt** | toàn bộ validation | soft attention mask A(t,f) thay mean-pool (Ilse et al., ICML 2018) | ✅ code sẵn |

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

**`run7_attn`** — Hướng 2, chạy riêng: `run1_baseline` đổi đúng một thứ — trung bình
patch token được thay bằng mask mềm `A(t,f) ∈ [0,1]` rồi pool có trọng số
`z = Σ A·h / Σ A`. Điểm mask theo gated attention (Ilse, Tomczak, Welling, ICML 2018);
**lệch paper**: họ chuẩn hoá bằng softmax, ở đây dùng sigmoid để A đúng là mask [0,1]
như sơ đồ và sau này giám sát được bằng IRM thật. `w = 0` lúc khởi tạo ⇒ bắt đầu đúng
bằng mean-pool. Không KD/CRD/FiLM ⇒ McNemar với `run1_baseline` cô lập tác dụng của
mask. Log in `mask_dev`; test36 lưu mask từng clip (`attention_mask` trong `.npz`) và
trung bình A theo 8 dải mel. Kết hợp KD/CRD sau khi có kết quả run này.

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

### 2026-09-28 — lượt 2: teacher `SSLAM_AS2M_Finetuned` dừng ở gate ⛔

Teacher val acc **0.6912** (lượt 1: 0.6576; BEATs: 0.7799). Vẫn dưới 0.75.

**Chẩn đoán không cần train** — cho head AudioSet 527 lớp có sẵn của checkpoint đoán thẳng
trên 720 clip noise sạch lát high (20 clip × 36 nhãn, validation), chỉ xét 36 cột của nhãn mình:

| Đầu vào mel | Zero-shot acc36 |
|---|---|
| **không pad, chia 2·std** (code đang dùng) | **0.7292** |
| không pad, chia std | 0.6333 |
| pad 1024 frame, chia 2·std | 0.7069 |
| pad 1024 frame, chia std | 0.5292 |

Hai kết luận:

1. **`norm_divisor = 2.0` đúng** — gỡ nghi vấn "chưa xác minh" trong README.
2. **Zero-shot (0.7292) cao hơn teacher đã finetune (0.6912).** Finetune làm *mất* 4 điểm.
   Nguyên nhân: checkpoint phân loại bằng `fc_norm(CLS)` → head (`eat_model.EAT.forward`),
   còn pipeline lấy **trung bình patch, bỏ CLS**, đưa vào **head ngẫu nhiên**. Đúng hiện
   tượng LP-FT mô tả (Kumar et al., *Fine-Tuning can Distort Pretrained Features*, ICLR
   2022): head ngẫu nhiên kéo méo đặc trưng tốt trong lúc tự học lại.

### Thay đổi sau lượt 2

- `backbone.pooling = "cls_fcnorm"` — vector gộp là `fc_norm(CLS)`, đúng đường phân loại
  của checkpoint. CRD mức patch vẫn dùng patch token như cũ.
- `backbone.head_init = "audioset"` — head 36 lớp khởi tạo từ 36 hàng tương ứng của head
  AudioSet (map qua `label_mids` → `models/audioset_class_labels_indices.csv`, đủ 36/36).
  Epoch 0 vì thế phải tái tạo xấp xỉ con số zero-shot 0.7292 — đó là phép kiểm tra rằng
  warm-start cài đúng.

### 2026-09-28 — lượt 3: pooling `fc_norm(CLS)` + head AudioSet

Epoch 0 = **0.7273**, khớp zero-shot 0.7292 → warm-start cài đúng. Nhưng finetune làm tụt
(0.6875 → 0.70) và checkpoint tốt nhất vẫn là epoch 0.

**Lỗi cấu hình tìm ra:** `optimizer_for` ưu tiên `teacher.head_lr` (1e-3, vốn cho pha
train riêng head — pha này không tồn tại trong bản SSLAM) thay vì `head_ft_lr` (1e-4). Head
vừa được warm-start bị đánh văng khỏi điểm tốt ngay epoch đầu.

### 2026-09-28 — sweep teacher (3 biến thể song song, cùng seed, chọn theo val lát high)

| Biến thể | Block train | lr head | lr encoder | Val acc tốt nhất |
|---|:-:|:-:|:-:|:-:|
| A — chỉ train head | 0 | 1e-3 | — | 0.7273 (epoch 0) |
| B — sửa lr head | 6 | 1e-4 | 1e-5 | 0.7292 |
| **C — finetune nhẹ** | 6 | 1e-4 | **3e-6** | **0.7380** |

Chọn **C**. Trần teacher SSLAM ở đây ≈ 0.74 (BEATs trên mid: 0.78).

### Hiệu chỉnh lại gate teacher

Ngưỡng tuyệt đối 0.75 mang từ dự án mid, nơi baseline mixture ≈ 0.67 (teacher 0.78 → hơn
**+0.11**). Ở dải high baseline là 0.5315: teacher C 0.738 hơn **+0.21**, gấp đôi. Thông báo
"teacher barely beats the baseline" sai với dải này. Ngưỡng đã được đánh dấu *chưa hiệu
chỉnh cho SSLAM/high* ngay từ commit đầu tiên.

Gate mới: `teacher ≥ baseline_band_accuracy + teacher_min_headroom` với headroom **0.10**
(= 0.6315), giữ tinh thần của mid. Teacher C qua gate.

### 2026-09-28 — lượt 4: teacher C (0.7380, GATE=PASS) + `run1_baseline` — **dừng lại để kiểm tra**

Theo yêu cầu: chạy xong `run1_baseline` thì lưu kết quả và dừng, chưa chạy run2–run3.
File gốc lưu ở `results/2026-09-28_run1_baseline/`.

Test, lát high (1.080 clip mỗi mức):

| | 15 dB acc | 15 dB F1 | 20 dB acc | 20 dB F1 | band acc | band F1 |
|---|---|---|---|---|---|---|
| **Mốc — BEATs-Mixture công bố** | 0.5769 | 0.5757 | 0.4861 | 0.4830 | 0.5315 | 0.5294 |
| `run1_baseline` (SSLAM, CE, không FiLM) | 0.5278 | 0.5264 | 0.4648 | 0.4674 | **0.4963** | **0.4977** |
| Δ | −4.91 | −4.93 | −2.13 | −1.56 | **−3.52** | **−3.16** |

**`run1_baseline` KHÔNG vượt mốc.** Đổi backbone BEATs → SSLAM tự nó không đem lại lợi thế
trên mixture ở dải high; ngược lại thua 3.5 điểm.

Đường cong validation (lát high): epoch 0 = 0.3755 (zero-shot trên mixture), đỉnh epoch 1 =
**0.5204**, sau đó tụt (0.5116, 0.5088) → early stop. Cùng dáng với teacher trước khi hạ lr
encoder: thuộc lòng nhanh, finetune làm hại sau epoch đầu. Student vẫn dùng `encoder_lr`
1e-5, trong khi sweep teacher cho thấy 3e-6 tốt hơn.

### 2026-09-28 — lượt 5: đủ 5 run (cùng teacher C 0.7380, cùng config) ✅ chạy xong

File gốc: `results/2026-09-28_5runs/` (CSV của report, JSON test/history từng run, log đã lọc).

**Test, lát high, top-1 accuracy** (1.080 clip mỗi mức; band = trung bình hai mức):

| Run | 15 dB | 20 dB | band | band macro-F1 | Δ band vs mốc |
|---|---|---|---|---|---|
| **Mốc — BEATs-Mixture công bố** | **0.5769** | **0.4861** | **0.5315** | **0.5294** | — |
| `run1_baseline` | 0.5278 | 0.4648 | 0.4963 | 0.4969 | −3.52 |
| `run2_ce_only` | 0.5287 | 0.4620 | 0.4954 | 0.4959 | −3.61 |
| `run3b_crd_only` | 0.5417 | 0.4722 | 0.5069 | 0.5048 | −2.46 |
| `run3c_kd_only` | **0.5648** | 0.4889 | 0.5269 | 0.5252 | −0.46 |
| `run3_kd_crd` | 0.5611 | **0.4926** | 0.5269 | 0.5245 | −0.46 |

**So với mốc công bố:** không run nào vượt ở mức band (tốt nhất −0.46 điểm). Ở 20 dB,
`run3_kd_crd` (0.4926) và `run3c_kd_only` (0.4889) **cao hơn** mốc 0.4861 (+0.65 / +0.28), nhưng
ở 15 dB vẫn thấp hơn (−1.58 / −1.21). Mốc công bố không có dự đoán từng clip nên không kiểm
định cặp được với nó.

**So với control dựng lại `run1_baseline` (McNemar theo cặp, cùng test set):**

| vs `run1_baseline` | Δ band | p band | Δ 15 dB | p | Δ 20 dB | p |
|---|---|---|---|---|---|---|
| `run2_ce_only` (FiLM) | −0.09 | 0.89 | +0.09 | 1.00 | −0.28 | 0.73 |
| `run3b_crd_only` | +1.06 | **0.035** | +1.39 | 0.068 | +0.74 | 0.32 |
| `run3c_kd_only` | **+3.06** | **8.6e-05** | +3.70 | **0.0008** | +2.41 | **0.035** |
| `run3_kd_crd` | **+3.06** | **1.2e-04** | +3.33 | **0.003** | +2.78 | **0.016** |

### Đọc kết quả

1. **Phương pháp có tác dụng thật và mạnh ở dải high**: KD cho +3.06 điểm, p < 0.001 —
   mạnh hơn nhiều so với dải mid (+0.23, p = 0.757). Khớp đúng dự đoán rằng hiệu ứng nằm ở high.
2. **KD là nguồn chính**, CRD đóng góp ít (+1.06 riêng lẻ, gần như không cộng thêm khi đã có
   KD). Giống hệt quy trách nhiệm ở dự án mid. FiLM không giúp gì.
3. **Vì sao vẫn chưa vượt mốc:** nền SSLAM thấp hơn nền BEATs công bố 3.52 điểm; +3.06 của KD
   bù gần hết nhưng thiếu 0.46. Vấn đề nằm ở backbone chứ không phải ở phương pháp.
4. Validation của mọi run đạt đỉnh sớm (epoch 1–3) rồi tụt; student vẫn dùng `encoder_lr` 1e-5,
   trong khi sweep teacher cho thấy 3e-6 tốt hơn trên cùng backbone.

### 2026-09-28 — tạm dừng

Theo yêu cầu, đã dừng mọi tiến trình trên molab và lưu kết quả về máy:
`results/molab_2026-09-28/` — dự đoán từng clip (`.npz`), test/history (`.json`), report CSV
của 5 run SSLAM; summary/history của lượt teacher 1–3 và sweep A/B/C; toàn bộ log molab.
Checkpoint `.pt` (6 × ~360 MB) **không** tải về, vẫn nằm trong `artifacts/` trên sandbox.

### Việc cần chạy lần sau: BEATs + công thức finetune của paper

Config: `config/train_config_high_beats.json` (output `artifacts_beats/`). Đã kiểm tra trên
molab trước khi dừng: lớp chuyển đổi BEATs khớp `extract_features` gốc (chênh 0.0), zero-shot
head AudioSet trên noise sạch lát high = 0.7375, augmentation/dropout chỉ bật khi train.

Smoke test lần đầu dừng ở `teacher36` vì
`ValueError: some parameters appear in more than one parameter group` — BEATs dùng chung
bias vị trí tương đối giữa các tầng. Đã sửa (`cbc0269`), **chưa chạy lại**.

Lệnh (từ `SNR_Aware/SNR_Expert_SSLAM`, sau `git pull`), chạy smoke trước:

```bash
python -u main.py teacher36 --config config/train_config_high_beats.json
for R in run1_baseline run2_ce_only run3b_crd_only run3c_kd_only run3_kd_crd; do
  python -u main.py student36 --config config/train_config_high_beats.json --run $R
  python -u main.py test36    --config config/train_config_high_beats.json --run $R
done
python -u main.py report36 --config config/train_config_high_beats.json
```

Sandbox mới thì phải cài lại môi trường (xem README: `transformers<5`, `timm`, torchvision
cu130) và tải checkpoint BEATs vào `../BEATs_Experts/checkpoint/pretrained/`.

# STATUS — Mid-SNR Expert

**Đọc file này đầu tiên khi mở phiên mới.** Nó nói đã làm tới đâu và bước kế tiếp là gì.

Cập nhật: 2026-09-24 — Task 1 xong, chạy local. Task 2–4 đang chạy.

---

## Bối cảnh 30 giây

Nhánh mid (5–10 dB) của pipeline SNR-aware đang **thua** BEATs baseline trên chính lát SNR của nó. Việc cần làm: dựng một mid expert vượt được mốc đó.

| | acc lát mid | macro-F1 lát mid |
|---|---|---|
| **Mốc phải vượt** — BEATs baseline | **0.6681** | **0.6602** |
| Mid expert cũ (đang thua) | 0.6616 | 0.6547 |

Nguyên nhân: expert cũ chỉ train trên 1/3 data (10.080 clip mid thay vì 30.240) và không dùng thông tin nào mà baseline không có.

Cách chữa: (1) student train trên **toàn bộ** SNR, chuyên biệt hoá bằng FiLM + chọn checkpoint theo lát mid; (2) khai thác `noise_path` (noise sạch) làm **privileged information** qua teacher–student distillation + CRD.

Phạm vi: **chỉ nhánh mid.** Không đụng low, không đụng high (DPCRN high đang 0.144 — đã biết, cố ý bỏ qua).

## File ở đâu

Tất cả trong `SNR_Aware/Mid_Expert/` trên nhánh **`mid-expert`**:

| File | Vai trò | Có chưa? |
|---|---|---|
| `DESIGN.md` | Spec đã duyệt. Vì sao làm thế này. | ✅ |
| `PLAN.md` | Plan 9 task, có code cụ thể từng step. | ✅ |
| `STATUS.md` | File này. Tiến độ + handoff. | ✅ |
| `mid_expert_lib.py` | Hàm thuần, test local được. | ✅ |
| `test_mid_expert.py` | Unit test local — `python test_mid_expert.py` → 16 passed. | ✅ |
| `main.py` + `tasks/run_36.py` | Stage `teacher36`, `bank36` | ✅ |
| `tasks/student_36.py` | Stage `student36`, `test36` — dùng chung code path cho run 2/3/4 | 🟡 |
| `config/train_config.json` | Toàn bộ tham số. Đổi `run` + `loss.a_kd` + `loss.b_crd` để chuyển giữa các run | ✅ |
| `README.md` | Lệnh chạy, các stage, ý nghĩa chốt dừng | ✅ |

**Molab lấy code bằng cách clone repo**, nên mọi thứ cần chạy đều phải commit + push lên nhánh `mid-expert`. `.gitignore` đã chặn `*.pt`, nên checkpoint và bank teacher không vào git — chúng sinh ra và ở lại trên molab.

## Ràng buộc cố định

- **Deliverable là script CLI, không phải notebook.** User chạy bằng `nohup ... > pipeline.log 2>&1 &` rồi `tail -f`, nên theo đúng khuôn `main.py <stage> --config <json>` của `BEATs_Experts` và `DPCRN_Noise_Target`. Notebook đã bị gỡ ở commit sau đó vì không hợp cách chạy này.
- Training chạy **trên molab**, không local, không `pip install`.
- Local có torch 2.13.0+cpu, numpy, scipy, soundfile, sklearn — **không có pytest**. Test chạy bằng `python test_mid_expert.py` với runner thuần ở cuối file.
- Dataset local: `36_labels/` (cùng schema với `/marimo/dataset/mix-dataset`). Dùng được để audit và test nhỏ. Local **không có `torchaudio`** (molab thì có), nên `noise_pipeline.mix_data` không import được ở local.
- **Checkpoint pretrained BEATs không đi theo git** (`.gitignore` chặn `*.pt`). Trên máy này bản duy nhất nằm ở `BEATs/C_Noise_Separation_Fusion/checkpoint/checkpoint4/pretrained/` (347 MB) — `SNR_Aware/BEATs_Experts/checkpoint/pretrained/` chỉ có `.gitkeep`. Trên molab phải có sẵn; notebook thử lần lượt các đường dẫn trong `CONFIG["pretrained_candidates"]` và dừng với thông báo rõ nếu không thấy.
- Seed 2026.
- Siêu tham số loss lấy từ repo CRD chính thức: `r=1, a=1, b=0.8, ρ=4, τ=0.07, proj=128, N=4096`. **Không tự đặt.**
- Trước mọi thay đổi phương pháp: tra paper (ICML/ICLR/NeurIPS) xem có làm vậy không, rồi mới sửa. Ghi rõ chỗ nào có paper chống lưng, chỗ nào là ý riêng.

## Ai chạy cái gì

| Loại | Task | Ai làm |
|---|---|---|
| Code thuần, chạy local | 2, 3, 4, Task 8 step 1–4 | Subagent làm trọn, test xanh mới tính xong |
| Cần GPU trên molab | 5, 6, 7, Task 8 step 5–6, 9 | Subagent viết cell notebook; **user bấm chạy trên molab** rồi đưa output về |

Task molab không "xong" khi code viết xong — chỉ xong khi có output thật dán vào mục Kết quả.

---

## Tiến độ

⬜ chưa bắt đầu · 🟡 code xong, chờ chạy molab · ✅ xong và có kết quả · ⛔ bị chặn

| # | Task | Nơi chạy | Trạng thái | Ghi chú |
|---|---|---|---|---|
| 1 | Audit manifest + check tuyến tính | local | ✅ | Xong. Kết quả bên dưới — **lật 2 giả định của spec** |
| 2 | CONFIG + `mid_slice_mask` / `normalize_snr` | local | ✅ | 16 test pass |
| 3 | FiLM conditioning | local | ✅ | Khởi tạo bằng 0 ⇒ identity; có `film_deviation` để bắt collapse |
| 4 | CRD loss + bank negative | local | ✅ | **Chốt dừng đã bật và đã xử lý** — xem bên dưới |
| 5 | Teacher trên noise sạch + bank | molab | ✅ | **acc 0.7799, GATE=PASS**, bank 43.200 hàng verify xong |
| 6 | Student run 2 (CE only) | molab | ✅ | **test mid 0.6657 / 0.6565, hoà baseline (−0.24pt)** |
| 7 | Student run 3 (+KD+CRD) | molab | 🟡 | Cùng code path với run 2, chỉ đổi `a_kd=1.0, b_crd=0.8` |
| 8 | Student run 4 (remix) | local + molab | ⬜ | **Đã mở khoá** — `LINEAR_OK=True` |
| 9 | Báo cáo + ablation | molab | ⬜ | |

**Bước kế tiếp:** Task 6 — student run 2 (chỉ CE, full SNR, có FiLM). Đây là **control quan trọng nhất**: không có nó thì không phát biểu được gì về CRD ở run 3.

Dự đoán để đối chiếu sau (ghi trước khi chạy, để khỏi tự lừa mình):

| Run | acc lát mid dự kiến | vs baseline 0.6681 |
|---|---|---|
| Run 2 (CE, full data + FiLM) | 0.665 – 0.678 | ≈ 0, ±0.5pt |
| Run 3 (+KD+CRD) | 0.672 – 0.690 | +0.5 đến +2pt |
| Run 4 (+remix) | 0.675 – 0.695 | +1 đến +3pt |

Dự đoán teacher trước đó là 0.76–0.78; thực tế 0.7799 — trúng mép trên.

## Hai chốt dừng

Gặp một trong hai thì **dừng, báo cáo, hỏi**:

1. **Task 4 Step 9** — độ lớn `crd_loss` lúc khởi tạo. Số hạng negative cộng 4096 phần tử; nếu nó lớn hơn CE (ln 36 ≈ 3.58) quá ~100 lần thì `b=0.8` của repo CRD không bê nguyên sang được, phải chuẩn hoá lại và **ghi vào danh sách lệch paper**.
2. **Task 5 Step 4** — acc của teacher.
   - `> 0.95` ⇒ soft label gần one-hot, KD vô dụng. Tăng `ρ` lên 8, ghi lý do.
   - `< 0.75` ⇒ teacher không mạnh hơn baseline mixture bao nhiêu, **tiền đề privileged-information lung lay**. Dừng hẳn, hỏi trước khi chạy Task 6.

---

## Kết quả

### Task 1 — audit (✅ chạy local trên `36_labels/`, 2026-09-24)

Manifest có **nhiều cột hơn spec tưởng**: ngoài `mixture_path` / `noise_path` còn có
`clean_path`, `noise_scale`, `post_gain`, `achieved_snr_db`.

| Kiểm tra | Kết quả | Hệ quả |
|---|---|---|
| `mixture == clean + noise`? | Đúng, sai số tối đa **5.96e-08** trên 300 clip | `noise_path` và `clean_path` là hai nguồn **đã scale sẵn**; mixture là tổng thuần |
| SNR đo được vs `target_snr_db` | Lệch **0.000 tuyệt đối**, cả 6 mức, 300/300 clip | FiLM nuôi bằng `target_snr_db` là chính xác |
| `achieved_snr_db` vs `target_snr_db` | Lệch max **0.0** trên cả 43.200 hàng | — |
| `unique noise_path` / train rows | 30.240 / 30.240 = **1.0** | ⚠️ **Spec sai.** Không có chuyện tái sử dụng 6 lần |
| `unique noise_ytid` (train) | 2.692 | Cùng nguồn YouTube nhưng mỗi hàng là đoạn/scale khác nhau |
| Số hàng | 43.200 (train 30.240 / val 6.480 / test 6.480) | Khớp spec |

**`LINEAR_OK = True`** → Task 8 remix chạy được.

**Hai sửa đổi bắt buộc so với spec/plan:**

1. **Bỏ hẳn bước dedup ở Task 5 Step 2.** Spec §3.1 giả định mỗi clip noise dùng lại qua 6 mức SNR nên teacher chỉ còn ~5.040 clip. Sai: tỉ lệ là 1.0, teacher train trên **đủ 30.240 clip**. Dedup theo `noise_path` là no-op; dedup theo `noise_ytid` sẽ **sai** vì cắt mất các đoạn khác nhau của cùng nguồn.
2. **Task 8 dùng `clean_path` trực tiếp**, không dựng `speech = mixture − noise`. Chính xác hơn và rẻ hơn. `remix = clean + gain · noise`, `gain` tính từ RMS thực đo.

`noise_scale` và `post_gain` chỉ là metadata provenance, đã nướng vào file — không cần dùng lúc train.

### Task 4 Step 9 — độ lớn crd_loss (✅ chốt dừng đã bật và đã xử lý)

| | giá trị |
|---|---|
| Đo lần đầu (bản rút gọn trong plan) | **9066.96** |
| CE lúc khởi tạo, 36 lớp = ln 36 | 3.58 |
| Tỉ lệ | **2531×** — vượt xa ngưỡng 100× |
| Sau khi sửa | **10.97** |
| Dự đoán lý thuyết `log(m+1)`, m=4096 | 8.32 |

**Nguyên nhân không phải `b=0.8` sai, mà là bản `crd_loss` trong plan của tôi thiếu một bước của CRD.** Đối chiếu `HobbitLong/RepDistiller` (`crd/memory.py` + `crd/criterion.py`): pipeline thật có hai bước, plan chỉ có bước một.

1. `out = exp(⟨v, v'⟩ / T)`
2. `out = out / Z`, với `Z = out.mean() · n_data`, tính **một lần** ở batch đầu rồi giữ cố định.

Thiếu bước 2 ⇒ giá trị vào critic là exponential thô chứ không phải xác suất ⇒ `log(1 − h_neg)` không nằm gần 0 ⇒ cộng 4096 số hạng thì nổ. Có `Z` thì `P_neg ≈ 1/n_data` nên `log_D0 ≈ 0`, cộng bao nhiêu cũng vẫn nhỏ — đúng như CRD gốc.

**Cách xử lý: implement `Z` (quay về đúng paper), KHÔNG chia số hạng negative cho `n_neg`.** Repo gốc cộng qua negative rồi chỉ chia cho batch size — phần đó plan vốn đã đúng. Chia cho `n_neg` mới là lệch paper.

Hệ quả: `b=0.8` của repo CRD **dùng được nguyên xi**, và mục lệch-paper #4 **không phát sinh**.

### Task 5 — teacher (✅ chạy trên molab, 2026-09-24)

```json
{
  "teacher_val_accuracy": 0.7799,
  "teacher_val_macro_f1": 0.7795,
  "teacher_gate": "PASS",
  "audit": { "rows": 43200, "unique_source": 30240, "reuse_ratio": 1.0 }
}
```

Bank: `z=(43200, 768)`, `logits=(43200, 36)`, 69 MB, assert alignment **pass**.

**Trần trên = 0.7799.** Khoảng trống so với baseline trên lát mid: **+11.2 điểm**
(0.7799 − 0.6681). `reuse_ratio = 1.0` xác nhận quyết định không dedup là đúng.

Đối chiếu: bản `BEATs/B_Noise_only/` cũ đạt 0.7528 với 2 block trainable. Recipe 12 block
ở đây được **+2.7 điểm**, nên phần finetune sâu là đáng.

#### Đường cong — hai điều đáng chú ý

```
head  val_f1 : [0.7694, 0.766, 0.7593, 0.7429, ... 0.7367]   ← đỉnh ngay epoch 1
ft train_acc : [0.9588, 0.9995, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
ft train_loss: [0.1683, 0.01, 0.0026, 0.0011, 0.0006, 0.0003, 0.0002, 0.0001]
ft val_acc   : [0.7792, 0.7781, 0.7759, 0.7789, 0.7799, 0.7787, 0.7787, 0.7773]
```

1. **Giai đoạn head đạt đỉnh ở epoch 1 rồi tụt đều.** Warm-start từ hàng predictor
   AudioSet đã gần tối ưu sẵn; train thêm chỉ làm hỏng. 11 epoch chạy để lấy epoch 1.
   Với student có thể hạ `head_patience` xuống 3.
2. **Teacher thuộc lòng tập train từ epoch 3** — train acc 1.0000, loss 0.0001, trong khi
   val đứng yên 0.776–0.780. Khoảng cách tổng quát hoá 22 điểm. Toàn bộ phần finetune
   đóng góp (+1.05 điểm so với head) đến ở epoch 1; epoch 2–8 không thêm gì.

### ⚠️ Gate đo nhầm đại lượng — đã sửa

Gate kiểm tra **validation accuracy** (0.7799 → PASS). Nhưng KD tiêu thụ logits của
teacher trên chính **các hàng train**, nơi teacher đã thuộc lòng. Accuracy không nhìn
thấy được việc thuộc lòng; entropy thì có.

Đo trên bank, 30.240 hàng train:

| ρ | max_prob | entropy / 3.5835 | |
|---|---|---|---|
| 1 | 0.9996 | 0.0036 (0%) | one-hot tuyệt đối |
| 2 | 0.9582 | 0.2746 (8%) | |
| **4** | **0.5490** | **2.1665 (60%)** | **vùng hợp lý** |
| 8 | 0.1683 | 3.3640 (94%) | gần như đều — dạy nhiễu |
| 16 | 0.0712 | 3.5473 (99%) | |

**Lời khuyên `RAISE_RHO` cũ của tôi là có hại và đã bị gỡ.** DESIGN.md §10 và gate bảo
"teacher quá mạnh thì nâng ρ lên 8"; đo thật thì ρ=8 đẩy phân phối về 94% entropy tối đa,
KD sẽ dạy nhiễu thay vì quan hệ giữa các lớp. **ρ cao hơn KHÔNG an toàn hơn.**

### KD vẫn dùng được — đã kiểm chứng

Lo "teacher thuộc lòng ⇒ soft label vô dụng" là **sai**. Đo cấu trúc dark knowledge trên
TRAIN (ρ=4) so với cấu trúc nhầm lẫn thật trên VALID:

- cosine trung bình: **0.42**
- đoán đúng lớp bị nhầm nhiều nhất: **26.5%** vs ngẫu nhiên 2.9% — gấp **9 lần**
- và có nghĩa âm học: Trumpet→Clarinet, Tearing↔Scissors, Writing→Scissors

Teacher thuộc *nhãn*, nhưng biểu diễn của nó vẫn mã hoá quan hệ giống nhau giữa các lớp —
đúng thứ KD cần. **Giữ `ρ=4`, `a_kd=1.0`, `b_crd=0.8` như kế hoạch.**

### Bảng so sánh chính

| Run | acc@5 | acc@10 | acc mid | F1 mid | Δacc vs baseline |
|---|---|---|---|---|---|
| BEATs baseline | 0.6806 | 0.6556 | 0.6681 | 0.6602 | — |
| Mid expert cũ | 0.6787 | 0.6444 | 0.6616 | 0.6547 | −0.0065 |
| **Teacher (noise sạch)** | — | — | **0.7799** | **0.7795** | **+0.1118 = trần trên** |
| **Run 2 (CE, full data + FiLM)** | 0.6694 | 0.6620 | **0.6657** | **0.6565** | **−0.0024** |
| Run 3 (+KD+CRD) | | | | | |
| Run 4 (+remix) | | | | | |

---

### Task 6 — run 2 (✅ chạy trên molab, 2026-09-24)

```
best val_mid_accuracy = 0.6935   val_mid_macro_f1 = 0.6925
test mid accuracy     = 0.6657   macro_f1        = 0.6565
                        baseline  0.6681           0.6602
                        delta    −0.0024          −0.0037
per_snr: 5 dB → 0.6694,  10 dB → 0.6620
```

Dự đoán trước khi chạy là 0.665–0.678; thực tế 0.6657, mép dưới nhưng trong khoảng.

#### ⚠️ Chênh lệch validation–test 2.8 điểm

`val_mid 0.6935` so với `test_mid 0.6657`. Checkpoint được chọn theo chính lát mid của
validation qua 8 epoch, nên con số validation **bị thiên lệch lạc quan** và **không được
đưa vào báo cáo**. Chỉ báo cáo test.

Thiên lệch này áp cho mọi run như nhau, nên so run 3 với run 2 **trên test** vẫn công
bằng. Đừng bao giờ so val của run này với test của run kia.

#### ⚠️ Giả thuyết "đói dữ liệu" không đứng vững

| | acc lát mid | dữ liệu train |
|---|---|---|
| Mid expert cũ | 0.6616 | 10.080 clip (chỉ mid) |
| Run 2 | 0.6657 | 30.240 clip (mọi SNR) |
| Baseline | 0.6681 | 30.240 clip (mọi SNR) |

Gấp **3 lần dữ liệu** chỉ đổi được **+0.41 điểm** — nằm gọn trong nhiễu (±1.0). DESIGN.md
§1 nêu hai nguyên nhân khiến mid expert cũ thua; **nguyên nhân "đói dữ liệu" giờ đo được
là không đáng kể**. Cộng thêm FiLM và việc chọn checkpoint theo lát mid cũng không kéo
được gì.

Điều này làm thí nghiệm **sạch hơn**, không phải tệ hơn: toàn bộ gánh nặng chuyển sang
KD + CRD. Nếu run 3 vượt baseline thì đó là **privileged information** làm nên chuyện,
không phải dữ liệu, không phải FiLM, không phải cách chọn checkpoint. Control đã loại
xong ba biến gây nhiễu.

Trong báo cáo: đừng viết "expert cũ thua vì thiếu dữ liệu". Số liệu nói ngược.

## Ý nghĩa thống kê — bắt buộc cho Task 9

Lát mid của test chỉ có **2.160 clip**. Sai số chuẩn của accuracy ở vùng p≈0.67 là

```
sqrt(0.67 × 0.33 / 2160) ≈ 0.0101  →  ±1.0 điểm
```

Mức cải thiện dự kiến của run 3 là +0.5 đến +2 điểm, tức **nằm trong hoặc sát nhiễu** nếu
so theo kiểu hai mẫu độc lập. Chính việc mid expert cũ thua 0.65 điểm cũng nằm trong đó.

**Task 9 phải dùng kiểm định McNemar theo cặp**, không so hai tỉ lệ. Mọi model đều chấm
trên đúng 2.160 clip giống nhau, nên chỉ cần đếm số clip mà hai model bất đồng:

```
b = số clip A đúng, B sai
c = số clip A sai,  B đúng
χ² = (|b − c| − 1)² / (b + c)     # hiệu chỉnh liên tục, 1 bậc tự do
```

Nhạy hơn hẳn vì bỏ qua phần lớn clip mà cả hai cùng đúng hoặc cùng sai. Không có nó thì
không được phát biểu "vượt baseline".

## ⚠️ Số liệu phản bác một lập luận trong DESIGN.md §2

Teacher đạt ~0.78 **phẳng đều ở mọi mức SNR** (per_snr: −5 → 0.7796, 20 → 0.7676), vì bài
của nó là noise sạch, không phụ thuộc SNR. So với baseline theo từng mức:

| SNR | baseline | teacher | khoảng trống |
|---|---|---|---|
| −5 | 0.7278 | ~0.78 | +5.2 |
| 0 | 0.7083 | ~0.78 | +7.2 |
| **5** | **0.6806** | ~0.78 | **+10.0** |
| **10** | **0.6556** | ~0.78 | **+12.5** |
| 15 | 0.6102 | ~0.78 | +17.0 |
| 20 | 0.5315 | ~0.78 | +24.9 |

Khoảng trống **tăng đơn điệu theo SNR**, lớn nhất ở 20 dB. DESIGN.md §2 lập luận mid là
nơi CRD lợi nhất; đo thật thì phần còn để học nhiều nhất nằm ở **high SNR**.

Lập luận của tôi dựa vào giả thuyết "ở 20 dB embedding quá xa để kéo về" — đã ghi trong
§2 là phần ngoại suy chưa ai đo, và giờ nó còn đi ngược thứ duy nhất đo được.

Không làm hỏng đồ án: phạm vi là nhánh mid, và mid vẫn có 10–12.5 điểm để khai thác.
Nhưng **trong báo cáo đừng viết "mid là nơi phương pháp này phát huy tốt nhất"** — bảng
trên phản bác ngay.

## Danh sách lệch khỏi paper (gom cho báo cáo)

Mỗi mục trong báo cáo phải viết "theo tinh thần của", **không** viết "như đã chứng minh".

1. **FiLM đặt một lần trên embedding đã pool**, không per-layer như paper gốc. Lý do: per-layer phải vá BEATs vendored. Bản per-layer để làm ablation.
2. **`g_t` đóng băng, chỉ `g_s` học.** CRD gốc học cả hai projection. Lý do: chiếu lại 43.200 vector có gradient mỗi batch là bất khả thi.
3. **Phần chiếu bank làm mới mỗi epoch** ⇒ stale trong phạm vi một epoch. (Bank `z` thì tĩnh thật.)
4. ~~Nếu Task 4 Step 9 buộc chuẩn hoá lại số hạng negative.~~ **Không phát sinh.** Chốt dừng đã bật nhưng nguyên nhân là plan thiếu bước chuẩn hoá `Z` của CRD, không phải `b=0.8` sai. Sửa xong là quay về đúng paper, không lệch thêm gì.

Cộng bốn chỗ ngoại suy ở §2 của DESIGN.md: ngoại suy "gộp thắng chia" sang câu hỏi hẹp về lát mid; lập luận mid là nơi CRD lợi nhất; FiLM theo SNR liên tục cho phân loại noise; trọng số `w(snr)` đặt tay (đã đẩy xuống ablation).

## Lỗi đã sửa so với spec

- **§5.1 check tuyến tính là tautology.** Spec bảo đo `‖mixture − (speech + noise)‖` với `speech := mixture − noise` — luôn bằng 0, pass kể cả khi dataset trộn phi tuyến. Đã thay bằng: so SNR đo được với `target_snr_db`. Hoá ra còn không cần, vì `clean_path` có sẵn.
- **§3.1 giả định dedup 6:1 là sai** — xem Task 1.
- **§3.2** nói FiLM đè lên 3 block cuối → đổi thành pooled, xem lệch paper #1.
- **§4.1** nói bank "không stale" → nói quá, xem lệch paper #3.

## Nền văn liệu

| Thành phần | Paper |
|---|---|
| Teacher xem noise sạch, student chỉ xem mixture | Lopez-Paz, Bottou, Schölkopf, Vapnik — *Unifying Distillation and Privileged Information*, ICLR 2016 |
| Kéo embedding mixture về embedding noise sạch | Tian, Krishnan, Isola — *Contrastive Representation Distillation*, ICLR 2020 |
| Student ăn toàn bộ SNR thay vì chỉ mid | Rebuffi, Bilen, Vedaldi, NeurIPS 2017; Jacobs, Jordan, Nowlan, Hinton 1991; Shazeer et al., ICLR 2017 |

## Nếu kết quả âm

Nếu run 3 ≈ run 2, kết luận là **"với bài này, gộp data mới là thứ có tác dụng, CRD thì không"** và báo cáo đúng như vậy. Không quét siêu tham số cho tới khi ra số đẹp. Ghi ở đây để phiên sau không lách.

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
| `mid_expert_lib.py` | Hàm thuần, test local được. | ⬜ |
| `test_mid_expert.py` | Unit test local. | ⬜ |
| `mid_expert.ipynb` | Deliverable, chạy trên molab. | ⬜ |

**Molab lấy code bằng cách clone repo**, nên mọi thứ cần chạy đều phải commit + push lên nhánh `mid-expert`. `.gitignore` đã chặn `*.pt`, nên checkpoint và bank teacher không vào git — chúng sinh ra và ở lại trên molab.

## Ràng buộc cố định

- Notebook mới, CONFIG gộp **một cell**, không tách file config.
- Training chạy **trên molab**, không local, không `pip install`.
- Local có torch 2.13.0+cpu, numpy, scipy, soundfile, sklearn — **không có pytest**. Test chạy bằng `python test_mid_expert.py` với runner thuần ở cuối file.
- Dataset local: `36_labels/` (cùng schema với `/marimo/dataset/mix-dataset`). Dùng được để audit và test nhỏ.
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
| 2 | CONFIG + `mid_slice_mask` / `normalize_snr` | local | 🟡 | Subagent đang làm |
| 3 | FiLM conditioning | local | 🟡 | Subagent đang làm |
| 4 | CRD loss + bank negative | local | 🟡 | Subagent đang làm. Step 9 là chốt dừng |
| 5 | Teacher trên noise sạch + bank | molab | ⬜ | **Bỏ bước dedup** — xem Task 1 |
| 6 | Student run 2 (CE only) | molab | ⬜ | Control quan trọng nhất |
| 7 | Student run 3 (+KD+CRD) | molab | ⬜ | |
| 8 | Student run 4 (remix) | local + molab | ⬜ | **Đã mở khoá** — `LINEAR_OK=True` |
| 9 | Báo cáo + ablation | molab | ⬜ | |

**Bước kế tiếp:** chờ Task 2–4 xong → review → viết notebook cho Task 5.

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

### Task 4 Step 9 — độ lớn crd_loss
```
chưa có
```

### Task 5 — teacher (trần trên)
```
chưa chạy
```

### Bảng so sánh chính

| Run | acc@5 | acc@10 | acc mid | F1 mid | Δacc vs baseline |
|---|---|---|---|---|---|
| BEATs baseline | 0.6806 | 0.6556 | 0.6681 | 0.6602 | — |
| Mid expert cũ | 0.6787 | 0.6444 | 0.6616 | 0.6547 | −0.0065 |
| Teacher (noise sạch) | — | — | — | — | _trần trên_ |
| Run 2 (CE, full data) | | | | | |
| Run 3 (+KD+CRD) | | | | | |
| Run 4 (+remix) | | | | | |

---

## Danh sách lệch khỏi paper (gom cho báo cáo)

Mỗi mục trong báo cáo phải viết "theo tinh thần của", **không** viết "như đã chứng minh".

1. **FiLM đặt một lần trên embedding đã pool**, không per-layer như paper gốc. Lý do: per-layer phải vá BEATs vendored. Bản per-layer để làm ablation.
2. **`g_t` đóng băng, chỉ `g_s` học.** CRD gốc học cả hai projection. Lý do: chiếu lại 43.200 vector có gradient mỗi batch là bất khả thi.
3. **Phần chiếu bank làm mới mỗi epoch** ⇒ stale trong phạm vi một epoch. (Bank `z` thì tĩnh thật.)
4. _(có thể phát sinh)_ Nếu Task 4 Step 9 buộc chuẩn hoá lại số hạng negative.

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

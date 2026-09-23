# Mid-SNR expert (5–10 dB) — design spec

Ngày: 2026-09-24. Phạm vi: **chỉ nhánh mid**. Không đụng low, không đụng high.

## 1. Vấn đề

Expert mid hiện tại thua BEATs baseline trên chính lát SNR của nó.

| Model | acc @5dB | acc @10dB | acc lát mid | macro-F1 lát mid |
|---|---|---|---|---|
| BEATs baseline (train 30.240 clip, mọi SNR) | 0.6806 | 0.6556 | **0.6681** | **0.6602** |
| BEATs mid expert (train 10.080 clip mid) | 0.6787 | 0.6444 | 0.6616 | 0.6547 |
| | | | −0.65 pt | −0.55 pt |

Nguồn: `BEATs_Experts/checkpoint/test_metrics_36.json`, `.../mid_snr/test_metrics_36.json`.

Hai nguyên nhân, và thiết kế này đánh vào cả hai:

- **Đói dữ liệu.** Expert chỉ thấy 1/3 số clip. Cùng backbone, cùng recipe finetune 8 epoch — không có cơ chế nào để 1/3 data thắng full data.
- **Không có thông tin mới.** Expert nhìn đúng thứ baseline nhìn (mixture) và học đúng thứ baseline học (nhãn). Manifest có `noise_path` — waveform noise sạch — hoàn toàn chưa được dùng.

Mục tiêu: **acc > 0.6681 và macro-F1 > 0.6602** trên 2.160 clip test ở 5 và 10 dB, báo cáo tách riêng hai mức.

## 2. Nền văn liệu (đã verify)

| Quyết định thiết kế | Nguồn | Điều paper thật sự nói |
|---|---|---|
| Teacher xem `noise_path`, student chỉ xem mixture | Lopez-Paz, Bottou, Schölkopf, Vapnik — *Unifying Distillation and Privileged Information*, ICLR 2016 | Generalized distillation 3 bước: (1) học teacher `f_t` trên `{(x*_i, y_i)}`; (2) tính soft label `s_i = σ(f_t(x*_i)/T)`; (3) học student trên cả `{(x_i, y_i)}` và `{(x_i, s_i)}`. `x*` chỉ tồn tại lúc train. |
| Kéo embedding mixture về embedding noise sạch bằng contrastive | Tian, Krishnan, Isola — *Contrastive Representation Distillation*, ICLR 2020 | Critic `h(T,S) = e^{g_T(T)'g_S(S)/τ} / (e^{g_T(T)'g_S(S)/τ} + N/M)`; `g_T`, `g_S` chiếu tuyến tính về cùng chiều rồi **chuẩn hoá L2** trước khi nhân trong. Memory buffer để khỏi cần batch lớn. |
| Student ăn toàn bộ SNR thay vì chỉ mid | Rebuffi, Bilen, Vedaldi — NeurIPS 2017; Jacobs, Jordan, Nowlan, Hinton — *Adaptive Mixtures of Local Experts*, 1991; Shazeer et al., ICLR 2017 | Backbone dùng chung + <10% tham số riêng theo domain **giữ nguyên hoặc cải thiện** so với model riêng từng domain. Trong ME, mọi expert nhận gradient trên mọi ví dụ, nhân responsibility do gate học ra — **không chia cứng data**. |

**Phần KHÔNG có paper chống lưng.** Trong báo cáo phải viết "theo tinh thần của", không viết "như đã chứng minh":

- Ngoại suy "gộp data thắng chia data" sang câu hỏi hẹp *"specialist có thắng generalist trên chính lát của nó không"*. Văn liệu chỉ trả lời câu hỏi tổng thể.
- Lập luận rằng dải mid là nơi CRD có lợi nhất (ở −5 dB mixture đã gần noise nên bài kéo là tầm thường; ở 20 dB quá xa để kéo). Hợp lý nhưng chưa ai đo.
- FiLM điều kiện hoá theo SNR liên tục cho bài phân loại noise.
- Trọng số `w(snr)` tam giác đặt tay — **đã loại khỏi thiết kế chính**, chỉ còn một dòng ablation (§7c). Lý do loại: ME dùng responsibility **do gate học ra**, không phải hằng số người đặt; giữ nó lại là mượn hình thức mà bỏ mất cơ chế.

## 3. Kiến trúc

```
LÚC TRAIN
  noise_path  ──► Teacher BEATs_N ──► z_t (768) ──► logits_t (36)
                    [giai đoạn 1; đóng băng ở giai đoạn 2]
                                         │
                                  precompute 1 lần
                                         ▼
                              teacher bank (N_rows × 768)

  mixture_path ─► Student BEATs_M ─► z_s (768) ─► head ─► logits_s (36)
                        ▲
                   FiLM(γ, β ← SNR)

LÚC INFER
  mixture ──► Student + FiLM(SNR từ gate) ──► 36 lớp
```

Teacher bị vứt sau khi train xong. `noise_path` không phải đầu vào lúc test → không vi phạm điều kiện triển khai.

### 3.1 Teacher (giai đoạn 1)

- BEATs, cùng checkpoint pretrained `BEATs_iter3_plus_AS2M_finetuned_on_AS2M_cpt2.pt`.
- Đầu vào: waveform ở `noise_path`, 4.0 s, 16 kHz.
- **Dedup theo `noise_path`.** Nếu mỗi clip noise được tái sử dụng qua cả 6 mức SNR thì 30.240 hàng chỉ ứng với ~5.040 clip duy nhất; không dedup thì teacher thấy mỗi clip 6 lần trong một epoch. Bắt buộc in ra số clip duy nhất để xác nhận giả định này.
- Recipe: giống `train_config.json` hiện có (train head trên embedding cache trước, rồi finetune 12 block).
- Lưu lại: checkpoint + **bank `z_t` và `logits_t` cho toàn bộ hàng train/val/test**, khoá theo row index của manifest (không khoá theo `noise_path`, để tra thẳng khi duyệt student).
- Acc của teacher chính là **trần trên** của cả hướng tiếp cận. Ghi nhận rõ.

### 3.2 Student (giai đoạn 2)

- BEATs, cùng pretrained, đầu vào mixture, **toàn bộ 30.240 clip train, mọi mức SNR**.
- FiLM: MLP 2 lớp `snr_norm → 128 → 2·768`, sinh `(γ, β)` đè lên đầu ra 3 block transformer cuối: `h ← γ ⊙ LN(h) + β`. `snr_norm = (snr + 5) / 25 ∈ [0, 1]`.
  - Lúc train: SNR thật từ `target_snr_db`.
  - Lúc infer: SNR hồi quy từ gate. **Phải huấn luyện chịu được sai số này** — jitter `snr_norm` bằng `N(0, σ)` với σ ứng với RMSE của gate. Gate chưa xây xong nên dùng σ = 2 dB và ghi rõ đây là giả định cần chỉnh lại khi có gate thật.
- Hai projection head chỉ sống lúc train: `g_S: 768 → 128`, `g_T: 768 → 128`, **L2-normalize đầu ra** (theo CRD).

## 4. Hàm mất mát

Theo tham số hoá của repo chính thức CRD (`-r 1 -a 1 -b 0.8`), dạng cộng:

```
L = r · CE(logits_s, y)
  + a · ρ² · KL( σ(logits_s/ρ) ‖ σ(logits_t/ρ) )
  + b · L_CRD( g_S(z_s), g_T(z_t) )
```

Giá trị khởi điểm lấy từ paper/repo, **không tự đặt**:

| Tham số | Giá trị | Nguồn |
|---|---|---|
| `r` (CE) | 1.0 | repo CRD |
| `a` (KD) | 1.0 | repo CRD, lệnh "CRD+KD" |
| `b` (CRD) | 0.8 | repo CRD + paper |
| `ρ` (nhiệt độ KD) | 4.0 | thông lệ KD; Lopez-Paz nói T phụ thuộc bài toán ⇒ phải quét, xem §7d |
| `τ` (nhiệt độ InfoNCE) | 0.07 | CRD, cấu hình ImageNet |
| chiều chiếu | 128 | CRD |
| `N` (số negative) | 4096 | CRD: chênh so với 16384 dưới 0.1% |

Lopez-Paz dùng dạng **tổ hợp lồi** `(1−λ)·CE + λ·KD` thay vì cộng tự do. Hai dạng chỉ sai khác một hệ số tỉ lệ khi `r + a` cố định; chọn dạng cộng vì đó là thứ repo CRD thật sự chạy và tái lập được.

### 4.1 L_CRD — chi tiết implement

- **Positive**: `(z_s(mixture_i), z_t(noise_i))` — cùng clip.
- **Negative**: `z_t` của 4096 clip khác, lấy từ **teacher bank tĩnh**. CRD cho chọn giữa "random `x_j`, j ≠ i" và "mẫu khác nhãn". Mặc định lấy **khác nhãn** (`y_j ≠ y_i`) để khỏi đẩy xa hai clip cùng lớp — thứ sẽ chống lại chính mục tiêu phân loại. Để thành switch trong CONFIG, ablate ở §7b.
- **Không cần memory buffer động.** CRD phải dùng buffer cập nhật dần vì họ không precompute được; teacher của ta đóng băng ⇒ bank là **chính xác, không stale**. Đây là chỗ ta thuận hơn paper. Bank 30.240 × 128 float32 ≈ 15 MB.
- Critic: `h = exp(⟨g_T, g_S⟩ / τ) / ( exp(⟨g_T, g_S⟩ / τ) + N/M )`, `M` = số mẫu trong tập dữ liệu.
- `L_CRD = − ( E_pos[log h] + N · E_neg[log(1 − h)] )`.
- **Chỉ dùng in-batch negative là sai.** Batch finetune BEATs là 32 ⇒ 31 negative. Ablation trong CRD cho thấy N = 16 tệ thấy rõ. Nếu làm vậy phần CRD sẽ không tạo ra cải thiện và ta sẽ kết luận nhầm là "CRD không hợp bài này".

## 5. Dữ liệu

- Train: toàn bộ `split == train`, 30.240 clip, cả 6 mức SNR.
- Chọn checkpoint theo **macro-F1 trên lát mid của validation** (5 và 10 dB, 2.160 clip), không phải toàn bộ validation. Đây là chỗ duy nhất "chuyên biệt hoá cho mid" đi vào quy trình chính — và nó không tốn thêm gì.
- Test: chỉ báo cáo lát mid.

### 5.1 Check tuyến tính (chặn cho §6)

Trên 200 clip train bất kỳ, dựng `speech = mixture − noise` rồi đo
`err = ‖mixture − (speech + noise)‖ / ‖mixture‖`.

- `err < 1e-4` → coi như trộn tuyến tính và căn thời gian ⇒ bật §6.
- Ngược lại → bỏ §6. Phần còn lại của thiết kế không lung lay.

Chạy check này **trước tiên**, vì nó quyết định có 3 hay 4 run.

## 6. Remix augmentation (chỉ khi §5.1 pass)

Trộn lại tại SNR **liên tục** `U(3, 12)` dB thay vì hai điểm rời rạc 5 và 10. Hệ số scale tính từ RMS thực đo của hai nguồn, không giả định mức nào sẵn.

Lý do: model hiện chỉ từng thấy đúng hai giá trị SNR trong dải mid, trong khi lúc infer gate đưa vào một số thực bất kỳ. Dải `[3, 12]` rộng hơn `[5, 10]` để FiLM có tín hiệu ở hai biên.

Nhãn giữ nguyên. `target_snr_db` đưa vào FiLM là SNR mới.

## 7. Chuỗi run

| # | Run | Trả lời câu hỏi |
|---|---|---|
| 1 | Teacher trên noise sạch (đã dedup) | Trần trên là bao nhiêu |
| 2 | Student, chỉ CE, full data + FiLM | Bao nhiêu phần cải thiện chỉ đến từ việc có thêm data — **control quan trọng nhất** |
| 3 | Run 2 + KD + CRD | CRD có đáng không |
| 4 | Run 3 + remix | Augment có cộng dồn không |

Không có run 2 thì không thể phát biểu bất cứ điều gì về CRD.

Ablation nếu còn compute, xếp theo mức đáng làm:

- **(a)** `b = 0` (bỏ CRD, giữ KD) — tách đóng góp của representation khỏi đóng góp của logit.
- **(b)** Negative random thay vì khác nhãn.
- **(c)** `w(snr)` tam giác đặt tay — ý riêng, không có paper. Thắng thì là đóng góp; thua thì đã loại được một biến.
- **(d)** Quét `ρ ∈ {2, 4, 8}` — Lopez-Paz nhấn mạnh T phụ thuộc bài toán.

## 8. Đánh giá

Chỉ lát 5–10 dB của test, 2.160 clip. Báo cáo `accuracy`, `macro_f1`, `micro_f1`, `map`, `balanced_accuracy`, `macro_auc` — **tách riêng 5 dB và 10 dB**, vì baseline tụt từ 0.6806 xuống 0.6556 giữa hai mức; cải thiện ở 10 dB có giá trị hơn.

Bảng bắt buộc: mỗi run đặt cạnh baseline 0.6681 / 0.6602 và cạnh mid expert cũ 0.6616 / 0.6547.

Thêm: F1 theo lớp so với baseline. Các lớp yếu nhất ở mid — Hi-hat 0.321, Drawer open or close 0.373, Microwave oven 0.462, Squeak 0.474 — là nơi tín hiệu từ noise sạch đáng lẽ giúp nhiều nhất. Nếu CRD có tác dụng thật thì phải nhìn thấy ở đúng những lớp này.

## 9. Giao hàng

Một notebook `.ipynb` **mới** trong scratchpad, CONFIG gộp trong một cell, chạy trên molab. Không thêm file vào repo, không dùng git.

## 10. Rủi ro

| Rủi ro | Dấu hiệu | Xử lý |
|---|---|---|
| Teacher quá mạnh, soft label gần one-hot ⇒ KD không tải thông tin | teacher acc > 0.95, KL loss tụt ngay từ epoch đầu | tăng `ρ`; dựa vào phần CRD thay vì KD |
| §5.1 fail | `err` lớn | bỏ §6; còn 3 run thay vì 4 |
| FiLM sụp về identity | `γ → 1`, `β → 0`, kết quả không khác run không FiLM | theo dõi norm của `(γ − 1, β)`; nếu sụp thì FiLM không phải cơ chế đúng — báo cáo đúng như vậy |
| Cải thiện đến hết từ data, CRD đóng góp 0 | run 3 ≈ run 2 | Vẫn là kết quả hợp lệ và đáng báo cáo: "với bài này, gộp data mới là thứ có tác dụng". **Không** cứu bằng cách quét siêu tham số tới khi ra số đẹp. |
| Gate σ = 2 dB đoán sai | student nhạy với jitter SNR | đo lại khi gate xong; nếu lệch nhiều phải train lại run cuối |

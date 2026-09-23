# Mid-SNR expert (5–10 dB)

Nhánh mid của pipeline SNR-aware đang **thua** BEATs baseline trên chính lát SNR của nó.
Dự án này dựng một expert vượt được mốc đó.

| | acc lát mid | macro-F1 lát mid |
|---|---|---|
| **Mốc phải vượt** — BEATs baseline | **0.6681** | **0.6602** |
| Mid expert cũ | 0.6616 | 0.6547 |

Đọc `STATUS.md` để biết đã tới đâu, `DESIGN.md` để biết vì sao làm thế này, `PLAN.md`
để biết từng bước.

## Ý tưởng

Teacher nhìn **noise sạch** (`noise_path`), student nhìn **mixture**. Noise sạch là
*privileged information* (Lopez-Paz, Bottou, Schölkopf, Vapnik — ICLR 2016): chỉ tồn tại
lúc train, không bao giờ là đầu vào lúc infer.

```text
LÚC TRAIN
  noise_path  → Teacher BEATs → z_t, logits_t → bank tĩnh (43.200 hàng)
  mixture     → Student BEATs + FiLM(SNR) → z_s, logits_s

  L = CE + KD(logits_t) + CRD(z_s, z_t)

LÚC INFER
  mixture → Student + FiLM(SNR từ gate) → 36 lớp
```

CRD (Tian, Krishnan, Isola — ICLR 2020) kéo embedding của mixture về phía embedding của
noise sạch cùng clip. Đây là chỗ "đảo ngược separation" thực sự nằm: student học cách
biểu diễn mixture sao cho trông như noise sạch, mà **không** phải tổng hợp lại waveform
nào — tránh hẳn artifact đã làm nhánh DPCRN high sụp còn 0.144.

## Chạy

```bash
cd SNR_Aware/Mid_Expert

nohup bash -c '
set -e
python -u main.py teacher36 --config config/train_config.json
python -u main.py bank36    --config config/train_config.json
' > pipeline.log 2>&1 &

tail -f pipeline.log
```

`-u` để dòng log hiện ngay thay vì nằm trong buffer của `nohup`.

Chạy thử trước cho thông pipeline: đặt `runtime.smoke_test = true` trong config (400 clip
train, 1 epoch), xong rồi mới đặt lại `false`.

### Các stage

| Stage | Làm gì | Ghi ra |
|---|---|---|
| `teacher36` | Train teacher trên noise sạch: head trên embedding đóng băng, rồi finetune 12 block | `artifacts/teacher_noise_best.pt`, `teacher_history.json`, `teacher_summary.json` |
| `bank36` | Precompute embedding + logits của teacher cho **cả 43.200 hàng** | `artifacts/teacher_bank.pt` |
| `student36` | Train student trên mixture, toàn dải SNR, FiLM theo SNR | `artifacts/student_<run>.pt`, `student_<run>_history.json` |
| `test36` | Chấm student trên test, tách riêng lát mid và từng mức SNR | `student_<run>_test.json`, `student_<run>_predictions.npz` |

`teacher36` **thoát với mã 2** nếu chốt dừng bật, nên `set -e` sẽ chặn `bank36` chạy tiếp.

Teacher đã chạy xong (acc 0.7799, `GATE=PASS`), nên vòng tiếp theo chỉ cần:

```bash
nohup bash -c '
set -e
python -u main.py student36 --config config/train_config.json
python -u main.py test36    --config config/train_config.json
' > run2.log 2>&1 &

tail -f run2.log
```

### Các run

`run2` → `run3` → `run4` chạy **cùng một code path**, chỉ khác config. Cố ý như vậy:
nếu control và treatment đi qua hai nhánh code khác nhau thì chênh lệch giữa chúng có
thể đến từ code chứ không phải phương pháp, và ablation mất ý nghĩa.

| Run | `run` | `a_kd` | `b_crd` | `remix` | Trả lời câu hỏi |
|---|---|---|---|---|---|
| 2 | `run2_ce_only` | 0.0 | 0.0 | off | Bao nhiêu phần cải thiện chỉ do có thêm data — **control quan trọng nhất** |
| 3 | `run3_kd_crd` | 1.0 | 0.8 | off | CRD + KD có đáng không |
| 4 | `run4_remix` | 1.0 | 0.8 | on | Augment có cộng dồn không |

Không có run 2 thì không phát biểu được gì về run 3.

Mỗi epoch in **tách riêng `ce`, `kd`, `crd`**. Nếu `crd` không giảm thì phần contrastive
không học được gì, và mọi cải thiện là do KD cộng data — cần biết điều đó trước khi viết
kết luận.

Cũng in `film_dev` mỗi epoch. Nếu nó đứng ở 0 thì FiLM đã sụp về identity và run này thực
chất là "baseline train lại" — vẫn là control hợp lệ, nhưng phải nói đúng như vậy trong
báo cáo thay vì nhận là điều kiện hoá theo SNR có tác dụng.

## Chốt dừng

`teacher36` in ra `GATE=` ở cuối:

| Verdict | Nghĩa là | Làm gì |
|---|---|---|
| `PASS` | acc trong [0.75, 0.95] | Chạy tiếp |
| `STOP` | acc < 0.75 | **Dừng hẳn.** Teacher nhìn noise sạch mà vẫn không hơn baseline nhìn mixture bao nhiêu ⇒ tiền đề privileged-information sụp. Đừng train student. |
| `RAISE_RHO` | acc > 0.95 | Soft label gần one-hot, KD không tải được gì. Đặt `rho_kd = 8` cho student. |

Acc của teacher là **trần trên** của cả hướng tiếp cận. Dán khối JSON cuối log vào
`STATUS.md`.

## Điều kiện tiên quyết

**Checkpoint pretrained BEATs không đi theo `git clone`** — `.gitignore` chặn `*.pt`. Nó
phải có sẵn trên máy chạy. `resolve_pretrained` thử lần lượt các đường dẫn trong
`pretrained.candidates` rồi dừng với thông báo rõ nếu không thấy cái nào; thêm đường dẫn
đúng vào đó nếu cần.

Dataset ở `dataset.path`, cần `manifest.csv` và `labels.txt`. Manifest phải có các cột
`split`, `mixture_path`, `noise_path`, `clean_path`, `label_names`, `label_mids`,
`target_snr_db`, `sample_id`.

## Test

```bash
python test_mid_expert.py
```

16 test cho các hàm thuần trong `mid_expert_lib.py` (FiLM, CRDLoss, lấy mẫu negative,
slicing theo SNR). Chạy được ở local, không cần GPU. Máy local không có `pytest` nên file
tự mang runner ở cuối, nhưng vẫn viết theo kiểu pytest-compatible.

## Ghi chú cho báo cáo

`STATUS.md` giữ danh sách **những chỗ lệch khỏi paper** và **những chỗ ngoại suy**. Mỗi
mục trong đó khi viết báo cáo phải dùng "theo tinh thần của", không được viết "như đã
chứng minh".

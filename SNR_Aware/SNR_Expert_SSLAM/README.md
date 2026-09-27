# High-SNR expert (15–20 dB) trên backbone SSLAM

Cùng cơ chế (KD + patch-level CRD) và cùng backbone SSLAM của thí nghiệm mid ban đầu
(`Mid_Expert_SSLAM/`, chưa từng chạy — folder này là chính nó, đổi tên và đổi band sau
khi số liệu chỉ ra hiệu ứng nằm ở high, không phải mid), nhưng nhắm vào **dải 15–20 dB**,
thay cho [`../DPCRN_Noise_Target`](../DPCRN_Noise_Target) trong pipeline (đang 0.144, xem
`../pipeline_config_4_branches.json`).

## Vì sao đổi từ mid sang high

`../Mid_Expert/STATUS.md` kết luận: teacher–student distillation từ noise sạch **không
vượt baseline có ý nghĩa thống kê** ở dải 5–10 dB (+0.23 điểm, McNemar p=0.757). Nhưng
đo theo từng mức SNR, hiệu ứng **tăng đơn điệu**: Spearman(SNR, Δ) = 0.943, p = 0.0048,
và đạt ý nghĩa ở 20 dB (+2.13 điểm, p = 0.0225). Cơ chế khớp: SNR càng cao, speech càng
át noise, mixture càng khác noise sạch — càng nhiều thứ để teacher (nhìn noise sạch) dạy.

Chi tiết đầy đủ của thang run, mốc phải vượt, và những gì đã xác minh trên molab: xem
`VERSIONS.md`.

## Hai lựa chọn kỹ thuật kế thừa từ bản mid

### 1. CRD căn ở mức patch, không phải pooled

Bản BEATs nén chuỗi token thành **một vector 768 chiều** bằng `sequence.mean(dim=1)`
**trước khi** loss nhìn thấy, vứt toàn bộ cấu trúc thời gian–tần số. SSLAM căn khớp ở
**mức patch** và lấy trung bình qua cả 12 layer. Đo trên wiring thật: batch 16, clip 4
giây, patch-level cho **800 anchor mỗi batch** thay vì **4**.

Đổi lại bằng `loss.crd_level`: `"patch"` (mặc định) hoặc `"pooled"` (tái lập bản BEATs).

### 2. SSLAM thay BEATs

| | AudioSet mAP |
|---|---|
| BEATs iter3 | 0.480 |
| **SSLAM** | **0.502** |

SSLAM được pretrain **trên mixture** (Self-Supervised Learning from Audio Mixtures,
ICLR 2025) — đúng bối cảnh bài toán này, và càng đúng hơn ở high SNR vì mixture ở đó gần
speech nhất, xa noise nhất.

## Không có stage `bank36`

Bản BEATs precompute một bank embedding của teacher. Ở đây không làm được: bank ở mức
patch sẽ là `201 token × 768 × 43.200 hàng ≈ 13 GB`. Thay vào đó **teacher chạy online
cùng batch** (đóng băng, không gradient).

## Yêu cầu môi trường

```bash
pip install 'transformers<5' timm torchaudio soundfile
```

**`transformers` 5.x KHÔNG chạy được** — lỗi `AttributeError: 'EATModel' object has no
attribute 'all_tied_weights_keys'`. Đã xác minh chạy được với **transformers 4.57.6 +
timm 1.0.30** (pair với molab, 2026-09-28).

**Trên molab, `timm` kéo `torchvision` bản PyPI về là hỏng**
(`RuntimeError: operator torchvision::nms does not exist`). Sửa bằng:

```bash
uv pip install --index-url https://download.pytorch.org/whl/cu130 \
  --no-deps --reinstall-package torchvision torchvision
```

Checkpoint tự tải từ hub (`ta012/SSLAM_pretrain`, 90M tham số) — không cần file `.pt`
đặt sẵn như BEATs.

## Chạy

```bash
nohup bash -c '
set -e
python -u main.py teacher36 --config config/train_config_high.json
for R in run1_baseline run2_ce_only run3_kd_crd run3b_crd_only run3c_kd_only; do
  python -u main.py student36 --config config/train_config_high.json --run $R
  python -u main.py test36    --config config/train_config_high.json --run $R
done
python -u main.py report36 --config config/train_config_high.json
' > full_run.log 2>&1 &

tail -f full_run.log
```

`run5_remix` và `run6_grl` chưa có switch — chờ §3–§5 của spec được duyệt (xem
`VERSIONS.md`).

Trước khi chạy, **luôn kiểm tra không còn chuỗi cũ**:

```bash
ps -eo pid,etime,args | grep main.py | grep -v grep
```

`--run` **bắt buộc** với `student36`/`test36`; `teacher36` từ chối nó. Lý do: config nằm
trong git, sửa tay sẽ bị `git pull` ghi đè và run chạy sai trong im lặng.

## Khác biệt cần biết khi so với bản BEATs

**Head khởi tạo ngẫu nhiên.** `SSLAM_pretrain` không có classifier AudioSet để
warm-start head 36 lớp như BEATs.

**Đầu vào là mel tính sẵn, không phải waveform.** `models/sslam.py` tính Kaldi fbank 128
bins rồi chuẩn hoá bằng `mean=-4.268, std=4.569`.

**Batch nhỏ hơn, accumulation bù lại.** Config hiện dùng `batch_size=16,
accumulation_steps=2` (giữ batch hiệu dụng 32, đặt cho GPU nhỏ). GPU quan sát được trên
molab (RTX PRO 6000 Blackwell, 102 GB VRAM) thừa sức nâng batch — chưa đổi ở bước này,
để lại cho giai đoạn implementation.

## ⚠️ Một thứ chưa xác minh

Công thức chuẩn hoá mel dùng `(x − mean) / (std × norm_divisor)` với
`norm_divisor = 2.0`. Model card chỉ cho hai hằng số, không nói chia cho `std` hay
`2·std`; AST và EAT — dòng mà SSLAM kế thừa — dùng `2·std`, nên đó là mặc định.

**Nếu kết quả thấp bất thường, đây là chỗ thử đầu tiên** (`backbone.norm_divisor`
trong config).

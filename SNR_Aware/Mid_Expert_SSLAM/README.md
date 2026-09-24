# Mid-SNR expert trên backbone SSLAM

Cùng thí nghiệm với [`../Mid_Expert`](../Mid_Expert), đổi **hai** thứ.

## Vì sao có folder này

`../Mid_Expert` kết luận: ở dải 5–10 dB, phương pháp **không** vượt baseline có ý nghĩa
thống kê (+0.23 điểm, McNemar p=0.757). Tách riêng từng thành phần cho thấy **KD đóng góp
nhiều hơn CRD** (+0.92 so với +0.37) — nghĩa là cơ chế chuyển giao **biểu diễn**, thứ cả
thiết kế xây quanh, lại là thứ đóng góp ít nhất.

Hai thay đổi ở đây nhắm đúng vào chỗ đó.

### 1. CRD căn ở mức patch, không phải pooled

Bản BEATs nén chuỗi token thành **một vector 768 chiều** bằng `sequence.mean(dim=1)`
**trước khi** loss nhìn thấy, vứt toàn bộ cấu trúc thời gian–tần số. SSLAM — model đứng
đầu AudioSet hiện nay — căn khớp ở **mức patch** và lấy trung bình qua cả 12 layer.

Đo trên wiring thật: với batch 16 và clip 4 giây, patch-level cho **800 anchor mỗi batch**
thay vì **4**. Gấp 200 lần tín hiệu giám sát cho cùng một lượng dữ liệu.

Đổi lại bằng `loss.crd_level`: `"patch"` (mặc định) hoặc `"pooled"` (tái lập bản BEATs).

### 2. SSLAM thay BEATs

| | AudioSet mAP |
|---|---|
| BEATs iter3 | 0.480 |
| **SSLAM** | **0.502** |

Quan trọng hơn điểm số: SSLAM được pretrain **trên mixture** (Self-Supervised Learning
from Audio Mixtures, ICLR 2025), đúng bối cảnh của bài toán này. Cơ chế lõi của nó —
Source Retention Loss, ép biểu diễn của mixture khớp với trung bình biểu diễn các nguồn
thành phần — **cùng hình dạng** với teacher–student ở đây.

## Không có stage `bank36`

Bản BEATs precompute một bank embedding của teacher. Ở đây không làm được: bank ở mức
patch sẽ là `201 token × 768 × 43.200 hàng ≈ 13 GB`.

Thay vào đó **teacher chạy online cùng batch** (đóng băng, không gradient). Tốn thêm
~50% thời gian mỗi epoch, nhưng bỏ được vấn đề stale mà bản BEATs phải ghi chú, và
**khớp đúng cách SSLAM vận hành teacher của nó**.

## Yêu cầu môi trường

```bash
pip install 'transformers<5' timm
```

**`transformers` 5.x KHÔNG chạy được.** Code remote trên hub viết cho 4.x và sẽ lỗi
`AttributeError: 'EATModel' object has no attribute 'all_tied_weights_keys'`. Đã xác
minh chạy được với **transformers 4.57.6 + timm 1.0.30**.

Checkpoint tự tải từ hub (`ta012/SSLAM_pretrain`, 90M tham số) — **không cần file `.pt`
đặt sẵn** như BEATs.

## Chạy

```bash
nohup bash -c '
set -e
python -u main.py teacher36 --config config/train_config.json
for R in run1_baseline run2_ce_only run3_kd_crd run3b_crd_only run3c_kd_only; do
  python -u main.py student36 --config config/train_config.json --run $R
  python -u main.py test36    --config config/train_config.json --run $R
done
python -u main.py report36 --config config/train_config.json
' > full_run.log 2>&1 &

tail -f full_run.log
```

Trước khi chạy, **luôn kiểm tra không còn chuỗi cũ**:

```bash
ps -eo pid,etime,args | grep main.py | grep -v grep
```

Ba chuỗi chạy chồng nhau đã từng ghi đè lẫn nhau và làm epoch chậm gấp đôi.

## Các phiên bản

| `--run` | `a_kd` | `b_crd` | FiLM | Chọn theo |
|---|:-:|:-:|:-:|:-:|
| `run1_baseline` | 0 | 0 | tắt | toàn bộ validation |
| `run2_ce_only` | 0 | 0 | bật | lát mid |
| `run3_kd_crd` | 1.0 | 0.8 | bật | lát mid |
| `run3b_crd_only` | 0 | 0.8 | bật | lát mid |
| `run3c_kd_only` | 1.0 | 0 | bật | lát mid |

`--run` **bắt buộc** với `student36`/`test36`; `teacher36` từ chối nó. Lý do: config nằm
trong git, sửa tay sẽ bị `git pull` ghi đè và run chạy sai trong im lặng — đã xảy ra một
lần ở bản BEATs.

## Khác biệt cần biết khi so với bản BEATs

**Head khởi tạo ngẫu nhiên.** BEATs warm-start head 36 lớp từ hàng predictor AudioSet
tương ứng; `SSLAM_pretrain` không có classifier nên không có gì để warm-start. Kỳ vọng
giai đoạn head cần đủ epoch chứ không đạt đỉnh ngay epoch 1 như bản BEATs.

**Đầu vào là mel tính sẵn, không phải waveform.** BEATs tự tính fbank bên trong; ở đây
`models/sslam.py` tính Kaldi fbank 128 bins rồi chuẩn hoá bằng `mean=-4.268, std=4.569`.

**Batch nhỏ hơn, accumulation bù lại.** ViT-Base trên chuỗi 201 token tốn bộ nhớ hơn;
config dùng `batch_size=16, accumulation_steps=2` để giữ batch hiệu dụng 32 như bản BEATs.

## ⚠️ Một thứ chưa xác minh

Công thức chuẩn hoá mel dùng `(x − mean) / (std × norm_divisor)` với
`norm_divisor = 2.0`. Model card chỉ cho hai hằng số, không nói chia cho `std` hay
`2·std`; AST và EAT — dòng mà SSLAM kế thừa — dùng `2·std`, nên đó là mặc định.

**Nếu kết quả thấp bất thường, đây là chỗ thử đầu tiên** (`backbone.norm_divisor`
trong config). Nó là tham số có thể đổi chứ không phải con số ẩn trong code.

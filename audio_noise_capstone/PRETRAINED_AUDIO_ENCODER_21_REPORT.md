# Báo cáo pretrained audio encoder trên bộ 21 nhãn

## 1. Dữ liệu và giao thức

Bộ dữ liệu có 26.044 mixture PCM16, mono, 16 kHz: 18.231 train, 3.907
validation và 3.906 test. Đây là bài toán **multi-label** (trung bình khoảng
1,39 nhãn/mẫu, tối đa 5), không phải single-label dù tên thư mục có hậu tố
`_single`. Mỗi waveform được pad/crop còn 6 giây; train dùng crop xác định theo
mẫu, validation/test dùng center crop. Đường dẫn local được dựng từ `sample_id`
để không phụ thuộc đường dẫn tuyệt đối cũ trong manifest.

Ba encoder được đánh giá zero-shot bằng đúng 21 logit AudioSet tương ứng. Chỉ
validation được dùng để chọn model và siêu tham số. Sau fine-tune, ngưỡng từng
lớp được chọn trên validation theo macro-F1, khóa lại, rồi test được chạy đúng
một lần.

## 2. Kết quả pretrained zero-shot

| Encoder | Validation micro-F1@0,5 | Macro-F1@0,5 | mAP |
|---|---:|---:|---:|
| AST | 0,1913 | 0,0941 | 0,4980 |
| PANNs CNN14 | 0,2541 | 0,1499 | 0,3745 |
| BEATs | **0,3627** | **0,3240** | **0,5959** |

BEATs thắng cả ba metric và mọi khoảng SNR, nên được chọn để fine-tune. mAP là
metric chọn model chính vì không phụ thuộc ngưỡng; F1@0,5 của các model zero-shot
chịu ảnh hưởng mạnh bởi calibration khác nhau.

## 3. Phương pháp fine-tune BEATs

- Khởi tạo classification head từ 21 hàng tương ứng trong AudioSet predictor.
- Loss: `BCEWithLogitsLoss`, `pos_weight` theo tần suất train và chặn tối đa 20.
- Epoch 1 chỉ học head; epoch 2–5 mở 4 transformer block cuối (28.368.976 tham
  số encoder có thể học). Các phần còn lại của BEATs giữ frozen.
- AdamW: LR head `5e-5`, LR encoder `2e-6`, weight decay `1e-4`; BF16, batch
  train/validation 32/64. Checkpoint được chọn bằng validation macro-AP.

Quá trình học tăng mAP từ 0,5952 ở epoch 0 lên 0,6079 sau head-only, rồi lần
lượt 0,6480, 0,6665, 0,6764 và **0,6830** khi mở encoder. Tổng thời gian khoảng
31,4 phút; peak VRAM quan sát được là 6,34 GB.

## 4. Kết quả cuối

| Model / chế độ | Split | Micro-F1 | Macro-F1 | mAP | Exact match |
|---|---|---:|---:|---:|---:|
| BEATs zero-shot, ngưỡng 0,5 | Validation | 0,3627 | 0,3240 | 0,5959 | 0,1116 |
| Fine-tuned, ngưỡng 0,5 | Validation | 0,5045 | 0,4968 | 0,6830 | 0,1387 |
| Fine-tuned, ngưỡng theo lớp | Validation | **0,6640** | **0,6659** | 0,6830 | **0,4262** |
| Fine-tuned, ngưỡng theo lớp đã khóa | Test | **0,6423** | **0,6413** | **0,6733** | **0,4030** |

`pos_weight` làm ngưỡng 0,5 dự đoán quá nhiều nhãn: 3,48 nhãn/mẫu trên
validation so với 1,39 nhãn thật. Calibration giảm còn 1,52; trên test là 1,50
so với 1,39. mAP không đổi vì metric này không phụ thuộc ngưỡng. Chênh lệch
validation→test nhỏ: −0,0097 mAP và −0,0246 macro-F1.

## 5. Đánh giá từng lớp trên test

| Lớp | Support | AP | F1 đã hiệu chỉnh |
|---|---:|---:|---:|
| Siren | 200 | 0,8995 | 0,8507 |
| Music | 889 | 0,8297 | 0,7642 |
| Wind instrument, woodwind instrument | 191 | 0,8172 | 0,7418 |
| Drum | 164 | 0,7842 | 0,6612 |
| Train | 165 | 0,7748 | 0,6986 |
| Water | 202 | 0,7692 | 0,7482 |
| Animal | 393 | 0,7358 | 0,6612 |
| Rail transport | 154 | 0,7239 | 0,6616 |
| Percussion | 142 | 0,7037 | 0,6383 |
| Fowl | 201 | 0,7017 | 0,6866 |
| Vehicle | 525 | 0,6851 | 0,6320 |
| Chicken, rooster | 155 | 0,6843 | 0,6971 |
| Bowed string instrument | 141 | 0,6793 | 0,6756 |
| Domestic animals, pets | 221 | 0,6596 | 0,6727 |
| Bird | 244 | 0,6386 | 0,5870 |
| Engine | 168 | 0,6319 | 0,6199 |
| Aircraft | 155 | 0,6152 | 0,5879 |
| Musical instrument | 302 | 0,5234 | 0,5160 |
| Car | 233 | 0,5132 | 0,5252 |
| Speech | 411 | 0,4067 | 0,4114 |
| Inside, small room | 166 | 0,3620 | 0,4313 |

Fine-tune cải thiện AP validation của cả 21 lớp so với BEATs zero-shot. Mức tăng
lớn nhất thuộc Speech (+0,2666), Music (+0,1924), Musical instrument (+0,1889)
và Inside, small room (+0,1245). Tuy vậy, Speech và Inside vẫn là hai lớp yếu
nhất; các cặp cha–con như Vehicle/Car, Music/Musical instrument và
Animal/Bird/Fowl cũng cho thấy bài toán phân cấp còn khó.

## 6. Ảnh hưởng của SNR trên test

| SNR (dB) | Mẫu | Micro-F1 | Macro-F1 | mAP |
|---|---:|---:|---:|---:|
| [-5, 0) | 1.165 | 0,7014 | 0,7008 | 0,7299 |
| [0, 5) | 136 | 0,7068 | 0,7173 | 0,8267 |
| [5, 10) | 1.181 | 0,6714 | 0,6689 | 0,7157 |
| [10, 15) | 135 | 0,6551 | 0,6645 | 0,7317 |
| [15, 20] | 1.289 | 0,5488 | 0,5432 | 0,5739 |

SNR cao nghĩa là speech tương đối mạnh hơn noise mục tiêu; khoảng 15–20 dB là
nút thắt rõ nhất. Hai khoảng hẹp [0,5) và [10,15) có ít mẫu nên không nên diễn
giải chênh lệch nhỏ giữa chúng.

## 7. Hướng tiếp theo đề xuất

1. Giữ checkpoint này làm baseline và không dùng lại test để chọn cấu hình.
   Chia một validation nội bộ mới hoặc chạy cross-validation cho vòng nghiên cứu.
2. Thử **epoch-dependent random crop** và 2–3 crop pooling khi đánh giá. Clip dài
   tới gần 68 giây, nên một center crop 6 giây có thể bỏ lỡ sự kiện.
3. So sánh BCE hiện tại với Asymmetric Loss hoặc focal loss. Đây là ưu tiên cao
   vì `pos_weight` đang làm xác suất lệch cao và cần ngưỡng 0,66–0,95.
4. Thay mean pooling bằng attentive/statistics pooling để giữ sự kiện ngắn; chạy
   pilot trước trên Speech, Inside, Car và Musical instrument.
5. Thêm consistency loss hoặc head phân cấp cho các nhóm cha–con; đồng thời kiểm
   tra lại quy tắc gán nhãn AudioSet hierarchy.
6. Tăng cường mẫu SNR 15–20 dB bằng dynamic mixing, speech gain augmentation hoặc
   nhánh estimated-noise. Mỗi thay đổi cần được ablation riêng và chạy ít nhất ba
   seed trước khi kết luận.

## 8. Tái lập và artifact

```powershell
python benchmark_pretrained_21.py --batch-size 32 --workers 4
python finetune_beats_21_dedup.py --epochs 5 --head-only-epochs 1 `
  --trainable-blocks 4 --batch-size 32 --validation-batch-size 64 --workers 4
python evaluate_beats_21.py --batch-size 64 --workers 4
python test_beats_21.py --batch-size 64 --workers 4
```

- Zero-shot: `benchmark_results/pretrained_21_zero_shot.json`
- Lịch sử fine-tune: `benchmark_results/beats_21_last4.json`
- Validation/calibration: `benchmark_results/beats_21_last4_evaluation.json`
- Test: `benchmark_results/beats_21_last4_test.json`
- Checkpoint: `checkpoints/beats_21_last4.pt`
- Ngưỡng đã khóa: `checkpoints/beats_21_thresholds.json`

# BEATs 21 lớp: Epoch-Crop và Three-Crop Ablation

## Mục tiêu

Thí nghiệm kiểm tra hai giả thuyết: crop train cố định làm model chỉ thấy một
vùng của clip dài, và center-crop validation có thể bỏ lỡ sự kiện nằm ở đầu hoặc
cuối clip. Test được khóa hoàn toàn; mọi lựa chọn chỉ dựa trên validation mAP.

## Thiết kế thí nghiệm

- Baseline: crop train xác định cố định theo `seed + sample_index`; validation
  dùng center crop 6 giây.
- Epoch-crop: mỗi lần một clip dài được đọc trong train, offset 6 giây được lấy
  lại ngẫu nhiên. Kiến trúc, seed, batch, optimizer, learning rate, loss và số
  epoch giữ nguyên.
- Three-crop: tại validation lấy đoạn đầu, giữa và cuối, sau đó lấy trung bình
  xác suất. Không tối ưu threshold; mọi F1 dùng ngưỡng 0,5.
- BEATs giữ 8 block đầu frozen, fine-tune 4 block cuối; epoch 1 chỉ học head và
  epoch 2-5 mở encoder.

## Kết quả tổng thể

| Train crop | Validation view | Micro-F1@0,5 | Macro-F1@0,5 | mAP |
|---|---|---:|---:|---:|
| Cố định | Center | 0,5045 | 0,4968 | 0,6830 |
| Cố định | Three-crop mean | **0,5117** | 0,5051 | 0,6867 |
| Thay đổi theo epoch | Center | 0,5037 | 0,4965 | 0,6836 |
| Thay đổi theo epoch | Three-crop mean | 0,5111 | **0,5054** | **0,6881** |

So với baseline center, cấu hình kết hợp tăng 0,0051 mAP và 0,0086 macro-F1.
Tuy nhiên, so với baseline đã dùng three-crop, epoch-crop chỉ tăng 0,0014 mAP và
0,0003 macro-F1. Epoch-crop một mình chỉ tăng 0,0006 mAP. Do đó phần lớn lợi ích
đến từ three-crop inference, không phải thay đổi crop khi train.

Three-crop max đạt mAP 0,6864 nhưng macro-F1 chỉ 0,4722, thấp hơn mean
aggregation. Max làm tăng xác suất của false positive, nên bị loại.

## Thay đổi theo SNR

So sánh cấu hình epoch-crop + three-crop mean với baseline center:

| SNR (dB) | Mẫu | Baseline mAP | Mới mAP | Delta mAP | Baseline Macro-F1 | Mới Macro-F1 |
|---|---:|---:|---:|---:|---:|---:|
| [-5, 0) | 1.186 | 0,7577 | 0,7650 | +0,0073 | 0,5686 | 0,5751 |
| [0, 5) | 146 | 0,7504 | 0,7580 | +0,0075 | 0,5297 | 0,5467 |
| [5, 10) | 1.150 | 0,7102 | 0,7160 | +0,0058 | 0,5136 | 0,5213 |
| [10, 15) | 121 | 0,7712 | 0,7504 | -0,0208 | 0,5123 | 0,5105 |
| [15, 20] | 1.304 | 0,5832 | 0,5861 | +0,0029 | 0,4221 | 0,4329 |

Vùng khó 15-20 dB tăng 0,0029 mAP và 0,0107 macro-F1, nhưng mức tăng mAP còn
nhỏ. Nhóm 10-15 dB giảm 0,0208 mAP; nhóm này chỉ có 121 mẫu nên sai số cao hơn,
nhưng vẫn cần được theo dõi ở các seed tiếp theo.

## Thay đổi nổi bật theo lớp

| Lớp | Baseline AP | AP mới | Delta |
|---|---:|---:|---:|
| Fowl | 0,6322 | 0,6576 | +0,0253 |
| Bird | 0,6378 | 0,6519 | +0,0141 |
| Inside, small room | 0,3109 | 0,3249 | +0,0140 |
| Chicken, rooster | 0,7375 | 0,7496 | +0,0121 |
| Car | 0,5919 | 0,6017 | +0,0098 |
| Musical instrument | 0,5693 | 0,5781 | +0,0087 |
| Aircraft | 0,7143 | 0,7080 | -0,0063 |
| Drum | 0,7119 | 0,7081 | -0,0038 |

Three-crop có lợi cho một số lớp sự kiện cục bộ và các lớp yếu như Inside và
Car. Tuy nhiên, mức tăng không đồng đều và chưa giải quyết đáng kể Speech hoặc
SNR 15-20 dB.

## Kết luận và quyết định

Cấu hình mới chưa đạt cổng cải thiện đã đặt trước là +0,01 validation mAP. Vì
vậy checkpoint epoch-crop không thay thế baseline chính. Three-crop mean có thể
được dùng cho đánh giá offline khi chấp nhận chi phí suy luận gần 3 lần, nhưng
không phù hợp mặc định cho hệ thống thời gian thực.

Thí nghiệm tiếp theo nên thay loss hiện tại bằng Asymmetric Loss trong một
ablation độc lập. Mục tiêu là xử lý mất cân bằng và false positive mà không dựa
vào threshold theo lớp.

## Artifact và lệnh tái lập

- Dataset crop: `noise_pipeline/audio21_crops.py`
- Fine-tune: `finetune_beats_21_epochcrop.py`
- Multi-crop evaluator: `evaluate_beats_21_multicrop.py`
- Checkpoint: `checkpoints/beats_21_epochcrop_last4.pt`
- Train history: `benchmark_results/beats_21_epochcrop_last4.json`
- Multi-crop results:
  `benchmark_results/beats_21_epochcrop_last4_multicrop_validation.json`

```powershell
python finetune_beats_21_epochcrop.py --epochs 5 --head-only-epochs 1 `
  --trainable-blocks 4 --batch-size 32 --validation-batch-size 64 --workers 4 `
  --output checkpoints/beats_21_epochcrop_last4.pt `
  --results benchmark_results/beats_21_epochcrop_last4.json

python evaluate_beats_21_multicrop.py `
  --checkpoint checkpoints/beats_21_epochcrop_last4.pt `
  --batch-size 16 --workers 4 `
  --output benchmark_results/beats_21_epochcrop_last4_multicrop_validation.json
```

# Đánh giá BEATs nguyên bản trên bộ dữ liệu 21 lớp

## Phạm vi thí nghiệm

Thí nghiệm sử dụng checkpoint chính thức
`BEATs_iter3_plus_AS2M_finetuned_on_AS2M_cpt2.pt`. Mô hình gồm 12 Transformer
block, embedding 768 chiều, 12 attention head, FFN 3.072 chiều và predictor
AudioSet 527 lớp; tổng cộng 90.717.055 tham số.

Đây là đánh giá **zero-shot đối với bộ dữ liệu 21 lớp**:

- Không tạo optimizer và không thực hiện backward pass.
- Không cập nhật encoder hoặc classification head; số tham số trainable bằng 0.
- Không hiệu chỉnh threshold; toàn bộ lớp dùng ngưỡng cố định 0,5.
- Predictor AudioSet 527 lớp được giữ nguyên, chỉ chọn 21 đầu ra có AudioSet ID
  (`mid`) trùng với `selected_labels.csv`.
- Waveform mixture mono 16 kHz được center-crop hoặc pad về 6 giây.

Checkpoint SSL thuần của BEATs chỉ sinh embedding và không thể trực tiếp dự đoán
21 lớp. Vì vậy, checkpoint AudioSet chính thức là cấu hình hợp lệ gần nhất để
đánh giá phân loại mà không học bất kỳ tham số nào từ bộ dữ liệu 21 lớp.

## Kết quả tổng thể

| Split | Số mẫu | Micro-F1@0,5 | Macro-F1@0,5 | Micro-AP | mAP | Exact Match |
|---|---:|---:|---:|---:|---:|---:|
| Validation | 3.907 | 0,3627 | 0,3240 | 0,3410 | 0,5959 | 0,1116 |
| Test | 3.906 | **0,3565** | **0,3077** | **0,3412** | **0,5857** | **0,0978** |

Trên test, mô hình dự đoán trung bình 1,4268 nhãn/mẫu, gần với 1,3881 nhãn thật
mỗi mẫu. Chênh lệch mAP validation-test là 0,0102, cho thấy kết quả giữa hai
split tương đối nhất quán. F1 thấp hơn mAP vì xác suất của predictor AudioSet
chưa được hiệu chỉnh cho phân bố dữ liệu 21 lớp và thí nghiệm cố ý giữ nguyên
threshold 0,5.

## Kết quả theo SNR trên test

| Khoảng SNR (dB) | Số mẫu | Micro-F1@0,5 | Macro-F1@0,5 | mAP |
|---|---:|---:|---:|---:|
| [-5, 0) | 1.165 | 0,4344 | 0,4188 | 0,6757 |
| [0, 5) | 136 | 0,4011 | 0,3977 | **0,7593** |
| [5, 10) | 1.181 | 0,3744 | 0,3250 | 0,6388 |
| [10, 15) | 135 | 0,3508 | 0,2489 | 0,6447 |
| [15, 20] | 1.289 | 0,2651 | 0,1648 | 0,4621 |

Hiệu năng giảm rõ ở khoảng 15-20 dB, nơi speech mạnh hơn tương đối so với noise
mục tiêu. Hai khoảng [0,5) và [10,15) có ít mẫu, do đó chênh lệch nhỏ giữa các
khoảng này cần được diễn giải thận trọng.

## Kết quả từng lớp trên test

| Lớp | Support | F1@0,5 | AP |
|---|---:|---:|---:|
| Siren | 200 | 0,5556 | 0,8176 |
| Wind instrument, woodwind instrument | 191 | 0,1449 | 0,7630 |
| Water | 202 | 0,1455 | 0,7052 |
| Train | 165 | 0,4038 | 0,6993 |
| Rail transport | 154 | 0,3011 | 0,6780 |
| Drum | 164 | 0,2827 | 0,6762 |
| Fowl | 201 | 0,5860 | 0,6760 |
| Chicken, rooster | 155 | 0,6179 | 0,6625 |
| Animal | 393 | 0,4305 | 0,6532 |
| Music | 889 | 0,6468 | 0,6501 |
| Percussion | 142 | 0,1438 | 0,6297 |
| Domestic animals, pets | 221 | 0,4798 | 0,6221 |
| Vehicle | 525 | 0,3961 | 0,6220 |
| Bowed string instrument | 141 | 0,1074 | 0,5988 |
| Bird | 244 | 0,3189 | 0,5747 |
| Aircraft | 155 | 0,2990 | 0,5404 |
| Engine | 168 | 0,1421 | 0,5100 |
| Car | 233 | 0,1805 | 0,4579 |
| Musical instrument | 302 | 0,0755 | 0,3941 |
| Speech | 411 | 0,2037 | 0,1966 |
| Inside, small room | 166 | 0,0000 | 0,1726 |

Theo AP, các lớp tốt nhất là Siren, Wind instrument, Water và Train. Hai lớp yếu
nhất là Speech và Inside, small room. Inside có AP lớn hơn 0 nhưng F1 bằng 0 ở
ngưỡng 0,5, nghĩa là model vẫn có một phần khả năng xếp hạng nhưng điểm số không
vượt qua ngưỡng quyết định cố định.

## Tái lập

```powershell
python evaluate_original_beats_21.py `
  --batch-size 16 `
  --workers 4 `
  --output benchmark_results/beats_original_21_zero_shot.json
```

Mã chạy nằm tại `evaluate_original_beats_21.py`; kết quả đầy đủ, bao gồm metric
từng lớp và từng khoảng SNR, nằm tại
`benchmark_results/beats_original_21_zero_shot.json`.

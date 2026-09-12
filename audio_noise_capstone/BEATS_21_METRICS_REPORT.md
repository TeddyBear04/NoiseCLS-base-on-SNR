# Kết quả đánh giá BEATs trên bộ dữ liệu 21 lớp

## Thiết lập đánh giá

Bộ dữ liệu gồm 21 lớp âm thanh và được xử lý dưới dạng bài toán multi-label.
Mô hình sử dụng mixture waveform mono 16 kHz, được pad hoặc crop về 6 giây. BEATs
được fine-tune với classification head 21 đầu ra; 8 Transformer block đầu được
giữ frozen và 4 block cuối được cập nhật. Checkpoint được chọn theo validation
macro average precision (mAP).

Ngưỡng dự đoán của từng lớp được hiệu chỉnh trên validation rồi khóa trước khi
chạy test. Test chỉ được đánh giá một lần và không được dùng để chọn mô hình hoặc
siêu tham số.

## Kết quả tổng thể

| Cấu hình BEATs | Split | Ngưỡng | Micro-F1 | Macro-F1 | mAP | Exact Match |
|---|---|---|---:|---:|---:|---:|
| Pretrained, chưa fine-tune | Validation | 0,5 | 0,3627 | 0,3240 | 0,5959 | 0,1116 |
| Fine-tuned | Validation | 0,5 | 0,5045 | 0,4968 | 0,6830 | 0,1387 |
| Fine-tuned | Validation | Theo từng lớp | **0,6640** | **0,6659** | **0,6830** | **0,4262** |
| Fine-tuned | Test | Theo từng lớp, khóa từ validation | **0,6423** | **0,6413** | **0,6733** | **0,4030** |

Fine-tune làm validation mAP tăng từ 0,5959 lên 0,6830, tương ứng mức tăng tuyệt
đối 0,0871. Test mAP đạt 0,6733, thấp hơn validation 0,0097. Test macro-F1 thấp
hơn validation 0,0246, cho thấy mức suy giảm tổng quát hóa tương đối nhỏ.

Ở ngưỡng cố định 0,5, mô hình dự đoán trung bình 3,48 nhãn/mẫu trên validation,
trong khi ground truth chỉ có 1,39 nhãn/mẫu. Sau hiệu chỉnh ngưỡng từng lớp, số
nhãn dự đoán trung bình giảm còn 1,52 trên validation và 1,50 trên test.

## Kết quả theo SNR trên test

| Khoảng SNR (dB) | Số mẫu | Micro-F1 | Macro-F1 | mAP |
|---|---:|---:|---:|---:|
| [-5, 0) | 1.165 | 0,7014 | 0,7008 | 0,7299 |
| [0, 5) | 136 | **0,7068** | **0,7173** | **0,8267** |
| [5, 10) | 1.181 | 0,6714 | 0,6689 | 0,7157 |
| [10, 15) | 135 | 0,6551 | 0,6645 | 0,7317 |
| [15, 20] | 1.289 | 0,5488 | 0,5432 | 0,5739 |
| **Toàn bộ test** | **3.906** | **0,6423** | **0,6413** | **0,6733** |

Trong bộ dữ liệu này, SNR cao nghĩa là speech mạnh hơn tương đối so với noise mục
tiêu. Hiệu năng giảm rõ rệt tại khoảng 15–20 dB, nơi macro-F1 chỉ đạt 0,5432 và
mAP đạt 0,5739. Hai khoảng [0,5) và [10,15) có ít mẫu hơn đáng kể, vì vậy không
nên diễn giải quá mức các chênh lệch nhỏ giữa những khoảng này.

## Ý nghĩa metric

- **mAP:** đo khả năng xếp hạng xác suất của 21 lớp và không phụ thuộc ngưỡng;
  đây là metric chính để chọn checkpoint.
- **Macro-F1:** tính F1 riêng cho từng lớp rồi lấy trung bình, phù hợp để đánh giá
  mức cân bằng giữa lớp phổ biến và lớp hiếm.
- **Micro-F1:** gộp dự đoán của toàn bộ lớp trước khi tính F1, phản ánh hiệu quả
  tổng thể nhưng chịu ảnh hưởng nhiều hơn từ lớp có nhiều mẫu.
- **Exact Match:** tỷ lệ mẫu mà toàn bộ tập nhãn dự đoán trùng hoàn toàn với
  ground truth; đây là metric nghiêm ngặt đối với bài toán multi-label.

## Kết luận

BEATs fine-tuned đạt test mAP 0,6733 và macro-F1 0,6413 trên bộ dữ liệu 21 lớp.
Kết quả xác nhận fine-tune encoder và hiệu chỉnh ngưỡng từng lớp đều cần thiết.
Hướng cải thiện tiếp theo nên ưu tiên dữ liệu SNR 15–20 dB, sử dụng dynamic
mixing hoặc speech-gain augmentation, đồng thời chỉ lựa chọn cấu hình trên tập
validation.

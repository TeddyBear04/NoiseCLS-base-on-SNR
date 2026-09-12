# Đánh giá lại BEATs: bộ 21 lớp và 36 lớp

## Phạm vi và định nghĩa metric

Hai bộ dữ liệu có kiểu bài toán khác nhau nên không thể so sánh accuracy trực
tiếp. Bộ 21 lớp là **multi-label**: accuracy ở đây là *subset accuracy/exact
match*—một mẫu chỉ đúng khi toàn bộ tập nhãn đúng. Bộ 36 lớp là **single-label**:
accuracy là top-1 accuracy thông thường. Cả hai đánh giá dùng checkpoint BEATs
đã chọn trước đó; bộ 21 dùng threshold từng lớp đã khóa từ validation.

## Kết quả test chạy lại

| Bộ dữ liệu / checkpoint | Mẫu | Accuracy | Micro precision | Micro recall | Micro-F1 | Macro precision | Macro recall | Macro-F1 | mAP |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 21 lớp multi-label, `beats_21_last4.pt` | 3,906 | 0.4030† | 0.6192 | 0.6673 | 0.6423 | 0.6383 | 0.6531 | 0.6413 | 0.6733 |
| 36 lớp single-label, `beats_last4.pt` | 6,480 | 0.0278 | — | — | — | 0.0008 | 0.0278 | 0.0015 | — |

†Subset accuracy/exact match, không phải top-1 accuracy.

## Diễn giải

Kết quả bộ 21 tái lập đúng checkpoint đã báo cáo trước đó. Precision và recall
khá cân bằng sau calibration; mAP không phụ thuộc threshold.

Kết quả test bộ 36 là bất thường và không thể coi là hiệu năng hợp lệ: model dự
đoán toàn bộ 6,480 mẫu là `Acoustic guitar`, dẫn đến accuracy đúng bằng 1/36.
Trong khi đó checkpoint ghi nhận validation accuracy 0.6492 và macro-F1 0.6469.
Đã kiểm tra checkpoint có 36 nhãn đúng thứ tự, manifest test có đủ 36 lớp (180
mẫu/lớp), và đường dẫn audio test hợp lệ. Đây là dấu hiệu domain shift hoặc lỗi
trong quá trình tạo/chia test của `mix-dataset`, không chỉ là sai metric.

## Artifact

- 21 lớp: `benchmark_results/beats_21_last4_retest_metrics.json`
- 36 lớp: `benchmark_results/beats_36_last4_test_metrics.json`
- Evaluator 36 lớp: `evaluate_beats_36_test.py`

## Khuyến nghị cho bộ 36

Không dùng kết quả test 0.0278 để kết luận năng lực BEATs. Trước khi fine-tune
hay thay loss, cần audit data generation: so sánh distribution waveform/SNR,
nguồn speech/noise và loudness giữa train-validation-test; kiểm tra test có bị
khác preprocessing hoặc leakage nhãn. Sau audit, huấn luyện lại với split đã
xác nhận và chỉ đánh giá test một lần.

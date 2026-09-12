# Báo cáo thí nghiệm BEATs — bộ dữ liệu 21 lớp

## Giao thức chung

Tập dữ liệu gồm 18,231 train, 3,907 validation và 3,906 test mẫu PCM16 16 kHz.
Đây là bài toán multi-label (1.39 nhãn/mẫu). Metric chọn cấu hình là validation
macro average precision (mAP), không phụ thuộc threshold. Test chỉ được dùng
một lần cho checkpoint baseline đã khóa.

## Baseline và pretrained encoder

| Mô hình | Split | Macro mAP | Macro-F1 @ 0.5 |
| --- | --- | ---: | ---: |
| AST zero-shot | Validation | 0.4980 | 0.0941 |
| PANNs CNN14 zero-shot | Validation | 0.3745 | 0.1499 |
| BEATs AudioSet zero-shot | Validation | 0.5959 | 0.3240 |
| BEATs zero-shot | Test | 0.5857 | 0.3077 |
| BEATs fine-tune last 4 + weighted BCE | Validation | **0.6830** | 0.4968 |
| Cùng checkpoint, threshold đã khóa theo lớp | Test | **0.6733** | **0.6413** |

BEATs fine-tune dùng head khởi tạo từ predictor AudioSet, 1 epoch head-only,
sau đó mở 4 block cuối; BCE có `pos_weight` chặn ở 20. Sau calibration,
threshold per-class giảm số nhãn dự đoán trung bình từ 3.48 xuống khoảng 1.5.

## Ablation đầy đủ trên validation

| Thay đổi so với baseline | Macro mAP | Delta | Quyết định |
| --- | ---: | ---: | --- |
| Random crop mới mỗi epoch | 0.683558 | +0.000601 | Không dùng riêng |
| 3-crop mean inference | 0.686670 | +0.003714 | Dùng khi chấp nhận 3× inference |
| Random crop + 3-crop mean | **0.688080** | +0.005124 | Tốt nhất validation, cần xác nhận seed khác |
| Asymmetric Loss | 0.666767 | -0.016189 | Loại |
| Learnable attention pooling | 0.682806 | -0.000151 | Loại |
| BCE sau khử tham số optimizer bị lặp | 0.682956 | +0.000000 | Giữ sửa lỗi kỹ thuật |

ASL làm tăng micro-AP nhưng dự đoán dư nhãn và giảm macro-mAP. Attention
pooling không mang lại lợi ích đáng kể so với mean pooling. Sửa optimizer loại
bỏ việc cập nhật lặp bảng relative-attention bias dùng chung; kết quả baseline
không đổi đáng kể.

## Pilot có đối chứng (1,000 train / 1,000 validation, 3 epoch)

| Phương pháp | Macro mAP | Delta so với control | Quyết định |
| --- | ---: | ---: | --- |
| BCE control | 0.632243 | — | Mốc pilot |
| Oracle-noise teacher distillation, weight 0.1 | 0.632148 | -0.000095 | Loại |
| Oversample SNR >=15 dB, 2× | 0.631766 | -0.000477 | Loại |
| Cap `pos_weight=5` cho Speech/Inside/Car/Musical instrument | 0.631378 | -0.000865 | Loại |

Distillation tăng VRAM từ khoảng 6.35 lên 7.11 GB nhưng không tăng mAP.
Oversampling SNR cao chỉ tăng mAP nhóm `[15,20]` +0.000700, đổi lại metric
tổng giảm. Targeted weight làm tăng micro-F1 +0.004091 nhưng giảm mAP và AP
cả bốn lớp mục tiêu, nên không chạy full.

## Phân tích lỗi lớp yếu

Ở threshold 0.5, lỗi chính là false positive, không phải thiếu recall:
`Speech` 968 FP/98 FN; `Inside, small room` 659/34; `Car` 584/22; `Musical
instrument` 727/31. Car hay bị nhầm trong Vehicle/Engine; Musical instrument
trong Music/nhạc cụ con. Speech đặc biệt khó vì mixture luôn có clean speech,
trong khi Speech cũng có thể là nhãn noise mục tiêu.

## Kết luận và bước tiếp theo

Checkpoint triển khai hiện nên là `checkpoints/beats_21_last4.pt`; dùng
3-crop mean chỉ trong chế độ offline nếu latency cho phép. Không dùng các
checkpoint ASL, attention, teacher-distillation, high-SNR hay targeted-weight.
Nghiên cứu tiếp theo nên là đánh giá nhiều seed cho crop + 3-crop, hoặc thiết
kế head phân cấp/contrastive cho Vehicle–Car và Music–instrument; với Speech,
cần tách clean speech hoặc bổ sung metadata, thay vì chỉ đổi loss.

## Artifact

- Error analysis: `benchmark_results/beats_21_error_analysis.json`
- Kết quả từng run: `benchmark_results/beats_21_*.json`
- Báo cáo chi tiết crop, ASL và attention: `BEATS_21_*_ABLATION_REPORT.md`

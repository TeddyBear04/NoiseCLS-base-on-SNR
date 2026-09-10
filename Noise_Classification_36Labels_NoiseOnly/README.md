# Noise-only classification (36 labels)

Project này được tạo trực tiếp từ template `Noise_Classification_36Labels`.
Khác biệt duy nhất về bài toán là model chỉ nhận waveform trong folder `noise`,
không sử dụng file `mixture` hoặc `clean`.

## Mục tiêu

Đây là oracle-noise baseline để kiểm tra riêng năng lực phân loại 36 loại noise
của backbone. Kết quả có thể so sánh với thí nghiệm mixture để biết lỗi đến từ
classifier hay do speech che lấp noise.

## Dữ liệu được sử dụng

```text
36_labels/
├── labels.txt
├── train/
│   ├── manifest.csv
│   └── noise/*.wav
├── validation/
│   ├── manifest.csv
│   └── noise/*.wav
└── test/
    ├── manifest.csv
    └── noise/*.wav
```

Config bị khóa ở `signal_type="noise"`. Dynamic SNR bị khóa ở `false` vì thao
tác đó cần clean speech. Folder `clean` và `mixture` có thể không tồn tại mà
pipeline vẫn train/evaluate bình thường.

## Chạy

```bash
cd /marimo/Noise_Classification_36Labels_NoiseOnly
python -m pip install -r requirements.txt

python main.py --config config/train_config.json --check-data
python main.py --config config/train_config.json --device cuda
```

Đường dẫn dataset mặc định là `../36_labels`. Có thể override bằng biến môi
trường `NOISE_DATASET_PATH`.

## Kết quả

Kết quả nằm trong `checkpoint/Cnn14MobileV2/`:

- `summary.csv`: ưu tiên `test_top1_accuracy`, `test_macro_f1` và
  `test_balanced_accuracy`.
- `test_per_label.csv`: xem `top1_recall` của từng noise class.
- `classification_report_test.txt`: precision, recall và F1 theo lớp.
- `learning_curves.png`: kiểm tra overfitting.

Waveform augmentation được tắt cho baseline đầu tiên. Nếu bật lại
`augmentation.enabled`, loader chỉ augment noise và same-class Mixup; nó vẫn
không đọc clean hoặc mixture.

## Chọn backbone

Toàn bộ backbone nằm trong `MODEL_REGISTRY` (`models/factory.py`). Đổi model
chỉ cần sửa `model.backbone` trong `config/train_config.json`, không đụng code:

```json
"model": { "backbone": "ResNet18", "pretrained": true, "classes_num": 36 }
```

Checkpoint ghi vào `checkpoint/<backbone>/` nên các lần chạy không đè lên nhau.

Số tham số đo với `classes_num=36`:

| Backbone | Params | Pretrained | Ghi chú |
| --- | ---: | :---: | --- |
| `Cnn14MobileV2` | 0.51M | – | mặc định hiện tại của baseline |
| `Cnn14MobileV2_1P9M` | 1.99M | – | biến thể rộng hơn của Cnn14MobileV2 |
| `MobileNetV2` | 3.57M | – | inverted residual, nhẹ |
| `EfficientNetB0` | 4.05M | ✓ | compound scaling, hiệu quả/params tốt |
| `MobileNetV1` | 4.29M | – | depthwise separable baseline |
| `PANNS_Cnn6` | 4.59M | – | PANNs, 4 conv block |
| `PANNS_Cnn10` | 4.97M | – | PANNs, 8 conv block |
| `DenseNet121` | 6.98M | ✓ | feature reuse, giữ lại chi tiết băng hẹp |
| `EfficientNetB2` | 7.75M | ✓ | một bậc scaling trên B0 |
| `ResNet18` | 11.19M | ✓ | residual baseline tầm trung |
| `ResNet34` | 21.30M | ✓ | như trên nhưng sâu gấp đôi |
| `ResNet22` | 62.67M | – | PANNs ResNet, nặng |
| `PANNS_Cnn14` | 79.75M | – | PANNs đầy đủ, nặng nhất |

Cột `Pretrained` cho biết backbone có nhận `model.pretrained: true` hay không.
Các model torchvision nạp trọng số ImageNet: conv đầu được đổi từ 3 kênh xuống
1 kênh và lấy trung bình trọng số RGB, phần còn lại giữ nguyên. Backbone không
hỗ trợ sẽ bị bỏ qua kèm cảnh báo trong log chứ không lỗi.

Mọi backbone tuân theo cùng một hợp đồng của `BaseBackbone`: nhận feature
`[Batch, 1, Time, Mel]` từ `AudioFrontend` và trả về logits `[Batch, 36]`.
`tests/test_noise_pipeline.py::TestBackboneRegistry` quét toàn bộ registry để
đảm bảo điều đó, nên có thể đổi qua lại mà không sợ vỡ pipeline.

Nhóm torchvision (`ResNet18/34`, `DenseNet121`, `EfficientNetB0/B2`) dùng chung
`models/torchvision_adapter.py`: đổi conv đầu về 1 kênh, pooling kiểu PANNs
(trung bình theo tần số, rồi max + mean theo thời gian), và head tuyến tính.
Thêm backbone torchvision mới chỉ mất khoảng 20 dòng.

# Dual-CED-Mini: mixup p = 1, α = 1

Code để chạy lại dòng **Dual-CED-Mini, p = 1, α = 1** trong `final-report.xlsx`, sheet *Đánh giá Mixup*.
Kết quả của lượt gốc (molab, 2026-11-20) trên validation sạch 3.064 clip:

| Acc | Precision | Recall | F1 | mAP | Epoch tốt nhất / đã chạy |
|---|---|---|---|---|---|
| **61,78** | 60,29 | 59,82 | 58,76 | 64,52 | 10 / 15 |

## Chạy

Cần GPU CUDA và khoảng 40 GB đĩa (dataset + cache n̂ FRCRN + 2 teacher, tải từ HF lần đầu).

```bash
git clone -b Dual-CED-Mini https://github.com/TeddyBear04/NoiseCLS-base-on-SNR
cd NoiseCLS-base-on-SNR/SNR_Aware/Dual_CED_Mini
bash run.sh            # = bash run.sh mx1a1
```

`run.sh` làm 3 việc:

1. Cài gói trong `requirements.txt` và `timm==1.0.30` (`--no-deps`); torch dùng bản có sẵn của máy (lượt gốc: torch 2.11.0+cu130).
2. `download_data.py`: tải từ HF dataset `phitruong0606/mix-dataset-v2` (công khai), kiểm SHA-256:
   teacher `mix_frcrn_ld0p3_best_model.pt` (Low, nghe n̂), `mix_original_ld0p8_weights.pt` (High, nghe mixture),
   `cache-nhat.zip` (n̂ FRCRN đã tính sẵn), `mix-dataset-v2.zip`.
3. `dual_ablation.py mx1a1`: train 15 epoch rồi chấm validation. Trước khi train, script so config với
   `config/reference_mx1a1_config_used.json` (config đúng của lượt 61,78) và dừng nếu lệch.

Đường dẫn mặc định giống máy gốc; đổi bằng biến môi trường:

| Biến | Mặc định |
|---|---|
| `DUAL_DATASET` | `/datasets/mix-dataset-v2` |
| `DUAL_FRCRN` | `/datasets/frcrn` |
| `DUAL_TEACHER_DIR` | `/marimo/pretrained/beat-ft` |
| `DUAL_OUT` | `/marimo/dual_ablation` |
| `PY` | `/tmp/uv-venv/bin/python` nếu có, không thì `python` |

Thời gian trên RTX PRO 6000: khoảng 8-10 phút train (thêm vài phút tính logit teacher lần đầu), chưa tính tải dữ liệu.

## Kết quả nằm ở đâu

`$DUAL_OUT/mx1a1/`:

- `results_anchored.json`: **số để so với Excel** (`overall.acc`, `precision`, `recall`, `macro_f1`, `mAP`, và theo từng SNR).
- `results.json`: lịch sử train. Cột `val_acc` ở đây là cách chấm trong lúc train (đỉnh 60,12 ở epoch 10),
  **không phải** số trong Excel; số Excel là cách chấm cuối ở dưới.
- `val_predictions_anchored.npz`, `config_used.json`, `best_model.pt`.

Dòng cuối của log in `RESULT mx1a1 acc=... prec=... rec=... f1=... mAP=... epoch=...`.

## Phương pháp

- Student: hai CED-Mini (khởi tạo `mispeech/ced-mini`), **Low** nghe n̂ FRCRN, **High** nghe mixture gốc; router Soft MoE (cosine) theo SNR.
- Loss: CE + KD từ hai teacher BEATs (λ 0,65, T 2; KD fused 0,5 + KD theo expert 0,5) + router loss 0,2.
- Recipe chung của ablation (ce9): 15 epoch, early stop patience 5 theo val acc, seed 2027, batch 64, AdamW wd 0,01,
  lr encoder 3e-5, head + router 3e-4, warm-up 2 epoch rồi cosine về 0, clip 1,0, layer decay 1,0,
  không label smoothing / SpecAugment / SWA / TTA, bf16.
- Mixup: mọi batch đều trộn (p = 1), λ ~ Beta(1, 1); trộn cả tín hiệu, nhãn, phân phối teacher và SNR.
- Chấm validation: đầu vào High = 0,85·n̂ + 0,15·mixture; gộp hai expert bằng cổng c = clip(0,42 + 0,05(α_router − 0,5)), một view.

Variant khác của cùng bảng: `bash run.sh base`, `bash run.sh ld0p82`, `bash run.sh mx0p4a0p4`, ...
(`mx<p>a<α>`, dấu phẩy viết thành `p`).

## Lưu ý về độ lặp lại

Cùng seed, cùng config, nhưng GPU khác, bf16 và DataLoader 8 worker không cho kết quả giống từng bit;
chênh lệch vài phần mười điểm acc là bình thường. Nếu lệch nhiều hơn khoảng 1 điểm thì kiểm lại SHA-256 dữ liệu
và phiên bản torch / timm.

Nguồn code: MLNC (`truong-vo-phi/MLNC`, nhánh `kd-moe`, commit `64e0418`, `imp/dual-expert`), thêm tùy chọn
`layer_decay_embeddings` trong `src/runner.py` và script ablation `dual_ablation.py`.

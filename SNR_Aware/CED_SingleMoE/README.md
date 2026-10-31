# CED-Tiny single-backbone MoE + KD hai teacher BEATs

Bản chép từ `truong-vo-phi/MLNC` nhánh `kd-moe`, thư mục `imp/kd-moe-ced`, đã chỉnh theo khuôn
`main.py <stage> --config <json>` của các nhánh anh em. Tự chứa, không import thư mục khác.

```text
n_hat (FRCRN, 16 kHz, 4 s) -> RMS 0.07925 -> 6 đặc trưng âm học + LogMel 64x398
  -> patch 16x16 (96 token + CLS) -> block 0-9 dùng chung (GREP attention)
  -> router: CLS block 9 (nén 16) + 6 đặc trưng -> alpha
  -> block 10-11: SplitMoEFFN = 0.5 shared + 0.5 [(1-alpha) low + alpha high]
  -> head_low / head_high -> z = (1-alpha) z_low + alpha z_high
```

Teacher: BEATs 5 fold fine-tune trên nhiễu sạch (nhiễu oracle), logit out-of-fold
(`Keshio1/frcrn-kdmoe-teacher/beats_teacher_logits_clean.pt`, ghép theo `sample_id`). Student chỉ nghe n_hat; teacher
chỉ dùng lúc train, lúc suy luận không cần.

Loss: `(1-0.65) CE + 0.65 KD + 0.20 route`. KD = 0.65 KD đầu ra trộn + 0.35 KD theo expert
(expert low/high cùng học teacher, nhân trọng số `1-w` / `w`); `w = sigmoid((SNR-15)/2.5)`;
route = BCE(alpha, w). Mọi hằng số nằm trong `config/train_config.json`.

## Chạy (molab)

```bash
python -u main.py fetch  --config config/train_config.json   # dataset (HF) + logit teacher (HF Keshio1, public)
python -u main.py check  --config config/train_config.json   # in acc teacher trên nhiễu sạch: ~0.74 là đúng
python -u main.py smoke  --config config/train_config.json
nohup python -u main.py train --config config/train_config.json > train.log 2>&1 &
python -u main.py eval   --config config/train_config.json   # student TTA +-50 ms trên val
```

Kết quả (chỉ validation) nằm ở `output_dir`: `lich_su.csv`, `val_theo_snr.csv`, `thong_so.csv`,
`best_model.pt`.

## Khác bản gốc

- Gốc dùng hai teacher BEATs (Fold0, EMA) và trộn xác suất teacher lúc suy luận; ở đây một teacher 5 fold
  nhiễu sạch, chỉ chưng cất khi train, suy luận chỉ có student.
- Dữ liệu đọc qua `manifest.csv` gộp (cột `split`, `validation` = `eval`) hoặc `<split>/manifest.csv`;
  bộ đệm n_hat kiểm số hàng và, nếu có `<tên>.json`, kiểm cả thứ tự `sample_id`.
- Gốc không đọc `config.yaml` và hard-code vài trọng số: nay đều lấy từ JSON.
- `load_ced_tiny` báo lỗi khi tải trọng số thất bại (gốc âm thầm dùng trọng số ngẫu nhiên).
- Bỏ `champion_pipeline.py`, `train_kd_moe.py` và mã BEATs; thay bằng stage `eval` và `train`.
- Lệch tài liệu: Notion ghi bảng bias GREP 320 bucket log; code dùng bảng 7 (tần số) x 47 (thời gian).

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

Expert: `model.experts` (2 = low/high như bản gốc, 3 = low/mid/high), chia khoảng SNR theo `loss.snr_edges_db`
([15] hoặc [2.5, 12.5] = -5/0, 5/10, 15/20 dB). Block 10-11: `0.5 shared + 0.5 sum_k g_k FFN_k`, một head mỗi expert,
logit = `sum_k g_k z_k`.

Routing (`routing`): bản gốc dùng gate mềm của router cả lúc train. Chẩn đoán run 1 (`main.py diag`) cho thấy expert
không chuyên môn hoá: FFN low/high chỉ lệch ~2 %, ép toàn bộ sang expert low cho đúng accuracy của cả model, router chỉ
đạt alpha 0.41 ở 20 dB. Bản 3 expert route cứng theo khoảng SNR thật lúc train (Hash Layers, Roller et al., NeurIPS 2021)
với xác suất giảm tuyến tính 1 -> 0 qua các epoch (Scheduled Sampling, Bengio et al., NeurIPS 2015); suy luận chỉ dùng router.

Teacher (`teacher.mode`):
- `clean5fold`: BEATs 5 fold fine-tune trên nhiễu sạch, logit out-of-fold (`Keshio1/frcrn-kdmoe-teacher`), ghép theo
  `sample_id`; mọi expert học cùng teacher.
- `fold0_ema`: hai teacher của bản gốc, Fold0 và EMA, nghe cùng n_hat (RMS 0.07925) với student (`main.py logits`);
  `teacher.expert_teachers` gán teacher cho từng expert. Eval báo thêm phép trộn teacher lúc suy luận của bản gốc.

Loss: `(1-0.65) CE + 0.65 KD + 0.20 route`, KD = 0.65 KD logit trộn (đích = teacher các expert theo khoảng SNR)
+ 0.35 KD theo expert (expert k học teacher của nó, trọng số khoảng SNR), T = 2; route = CE(gates, nhãn khoảng SNR).

## Chạy (molab)

```bash
python -u main.py check  --config config/train_config.json   # in acc teacher trên nhiễu sạch: ~0.74 là đúng
python -u main.py smoke  --config config/train_config.json
nohup python -u main.py train --config config/train_config.json > train.log 2>&1 &
python -u main.py eval   --config config/train_config.json   # student TTA +-50 ms trên val
python -u main.py diag   --config config/train_config.json   # ép từng expert theo SNR: expert có chuyên môn hoá không
```

Config: `train_config.json` (run 1, 2 expert, teacher nhiễu sạch), `train_config_3exp.json` (3 expert, route oracle có lịch),
`train_config_fold0_ema.json` (như trên, teacher Fold0/EMA; chạy `logits` trước). Dữ liệu tải bằng notebook molab.

Kết quả (chỉ validation) nằm ở `output_dir`: `lich_su.csv`, `val_theo_snr.csv`, `val_tta_theo_snr.csv`,
`val_chan_doan_expert.csv`, `thong_so.csv`, `best_model.pt`.

## Khác bản gốc

- Thêm teacher 5 fold nhiễu sạch, số expert tuỳ chọn, route oracle có lịch và stage `diag`.
- Dữ liệu đọc qua `manifest.csv` gộp (cột `split`, `validation` = `eval`) hoặc `<split>/manifest.csv`;
  bộ đệm n_hat kiểm số hàng và, nếu có `<tên>.json`, kiểm cả thứ tự `sample_id`.
- Gốc không đọc `config.yaml` và hard-code vài trọng số: nay đều lấy từ JSON.
- `load_ced_tiny` báo lỗi khi tải trọng số thất bại (gốc âm thầm dùng trọng số ngẫu nhiên).
- Bỏ `champion_pipeline.py`, `train_kd_moe.py`; thay bằng stage `logits`, `train`, `eval`.
- Lệch tài liệu: Notion ghi bảng bias GREP 320 bucket log; code dùng bảng 7 (tần số) x 47 (thời gian).

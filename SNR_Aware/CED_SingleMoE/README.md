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
Router học (`routing.source: router`, `model.router_input: raw`): 6 đặc trưng tính trên n_hat thô (bản gốc tính sau chuẩn hoá
RMS nên log-RMS là hằng số; RMS n_hat thô tương quan -0.83 với SNR), log crest/kurtosis, BatchNorm.
`routing.source: frcrn_snr` (cách của FRCRN_KD_MoE): route one-hot theo khoảng của SNR ước lượng
10 log10(P(mixture - n_hat) / P(n_hat)) ở mức gốc (`frcrn_{split}.snr.npy`), cả train lẫn suy luận.

Teacher (`teacher.mode`):
- `clean5fold`: BEATs 5 fold fine-tune trên nhiễu sạch, logit out-of-fold (`Keshio1/frcrn-kdmoe-teacher`), ghép theo
  `sample_id`; mọi expert học cùng teacher.
- `fold0_ema`: hai teacher của bản gốc, Fold0 và EMA, nghe cùng n_hat (RMS 0.07925) với student (`main.py logits`);
  `teacher.expert_teachers` gán teacher cho từng expert. Eval báo thêm phép trộn teacher lúc suy luận của bản gốc.

Loss: `(1-0.65) CE + 0.65 KD + 0.20 route`, KD = 0.65 KD logit trộn (đích = teacher các expert theo khoảng SNR)
+ 0.35 KD theo expert (expert k học teacher của nó, trọng số khoảng SNR), T = 2; route = CE(gates, nhãn khoảng SNR).

GFD-Cosine (`feature_kd`, `src/feature_kd.py`, config `train_config_overlap_gfd.json` = patch chồng lấn + loss này):
`L = L_hiện_tại + weight * mean(1 - cos(LayerNorm(W s), t))`, s = mean token patch sau LayerNorm cuối CED (192),
t = mean token lớp l BEATs teacher 5 fold nghe nhiễu sạch (768, fold out-of-fold như logit); `taps` ghép block b của CED với lớp l (GFD-5: 6/9/12, mỗi cặp một projector, lấy trung bình). Gộp toàn clip nên không cần
khớp số token. Projector Linear(192, 768) + LayerNorm chỉ ở trainer, không vào `best_model.pt`. Gần PEFD (Chen et al.,
NeurIPS 2022: đặc trưng lớp cuối, projector phía student, 25 x (2 - 2 cos) ~ weight 50). Mixup trộn t cùng hoán vị.
`lich_su.csv` có `loss_feat` và `grad_ratio_feat` = |grad(weight L_feat)| / |grad(L_hiện_tại)| trên backbone.
t xuất bằng FRCRN_KD_MoE: `teacher_crossfit.py features --config config/unified/teacher_pooled_gfd.json`
(fold HF chép thành `output/teacher_unified/fold{f}.pt`, cần dataset 36_labels có nhiễu sạch) -> `teacher_hints_l{6,9,12}_g1x1.npy` + `.ids.json`.

## Chạy (molab)

```bash
python -u main.py check  --config config/train_config.json   # in acc teacher trên nhiễu sạch: ~0.74 là đúng
python -u main.py smoke  --config config/train_config.json
nohup python -u main.py train --config config/train_config.json > train.log 2>&1 &
python -u main.py eval   --config config/train_config.json   # student TTA +-50 ms trên val
python -u main.py diag   --config config/train_config.json   # ép từng expert theo SNR: expert có chuyên môn hoá không
# quét trọng số GFD-Cosine (giá trị JSON, khoá phải có sẵn trong config):
python -u main.py train --config config/train_config_overlap_gfd.json \
    --set feature_kd.weight=1.0 --set paths.output_dir=output/ced_singlemoe_v2_overlap_gfd_w1
```

Config: `train_config.json` (run 1, 2 expert, teacher nhiễu sạch), `train_config_3exp.json` (3 expert, route oracle có lịch),
`train_config_fold0_ema.json` (như trên, teacher Fold0/EMA; chạy `logits` trước); bản `_frcrnsnr` của hai config sau
route theo SNR ước lượng từ FRCRN (cần `frcrn_{split}.snr.npy` từ `build_frcrn_snr` của FRCRN_KD_MoE). Dữ liệu tải bằng notebook molab.

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

## Ablation final-report.xlsx: mixup p 0,8, α 0,8 (val acc 63,19)

Chạy lại ô tốt nhất của Single-CED (SplitFFN) trong sheet *Đánh giá Mixup* (lượt molab 2026-11-20, commit `6c80ca1`).
Validation sạch 3.064 clip (v2 eval):

| Acc | Precision | Recall | F1 | mAP | Epoch tốt nhất / đã chạy |
|---|---|---|---|---|---|
| **63,19** | 61,75 | 61,46 | 60,50 | 66,89 | 11 / 15 |

```bash
git clone -b Single-CED https://github.com/TeddyBear04/NoiseCLS-base-on-SNR
cd NoiseCLS-base-on-SNR/SNR_Aware
cp /đường/dẫn/frcrn_best.pt FRCRN_KD_MoE/model/frcrn_best.pt   # mkdir -p FRCRN_KD_MoE/model trước
bash CED_SingleMoE/run_ablation.sh            # = run_ablation.sh mx0p8a0p8
```

Cần GPU CUDA và khoảng 60 GB đĩa. `run_ablation.sh` làm 5 việc:

1. Cài gói (`timm==1.0.30` với `--no-deps`, `torchaudio` theo đúng bản torch; lượt gốc dùng torch 2.11.0+cu130).
2. `download_data.py`: tải từ HF (công khai, kiểm SHA-256) mix-dataset-v2 (`phitruong0606/mix-dataset-v2`),
   36_labels (`Keshio1/36_labels`, chỉ giải nén mixture + manifest của train / validation) và logit teacher
   `beats_teacher_logits_clean.pt` (`Keshio1/frcrn-kdmoe-teacher`, rev `a231940`).
3. `build_nhat.py`: dựng cache n̂ FRCRN fp32 trên 36_labels bằng code `../FRCRN_KD_MoE` và `frcrn_best.pt`
   (kiểm SHA-256 `b8fdeb00...`), khoảng 25 phút. Đã có cache thì bỏ qua.
4. Pytest recipe ce9 + smoke 1 epoch.
5. `ablation.py mx0p8a0p8` = `main.py train --config config/train_config_overlap_ce9_mini_ablation_s2027.json`
   với `--set train.mixup_prob=0.8 --set train.mixup_alpha=0.8`. Trước khi train, config được so với
   `config/reference_mx0p8a0p8_thong_so.csv` (thong_so.csv của lượt gốc) và dừng nếu lệch.

Kết quả: `output/ablation_s2027/mx0p8a0p8/val_theo_snr.csv` (dòng `overall`: top-1, precision, recall, F1 macro, mAP),
`lich_su.csv` (từng epoch), `thong_so.csv`, `val_du_doan_best.csv`. Log in `best val top1 ...` ở cuối.

Đường dẫn mặc định giống máy gốc, đổi bằng biến môi trường: `SF_DATA_V2` (`/marimo/dataset_v2/mix-dataset-v2`),
`SF_DATA_36L` (`/marimo/dataset/mix-dataset`), `SF_NHAT` (`/marimo/nhat_cache`),
`SF_TEACHER` (`/marimo/teacher/beats_teacher_logits_clean.pt`), `SF_FRCRN_CKPT`, `PY`.
Variant khác của cùng bảng: `bash run_ablation.sh base | ld0p82 | mx1a1 | ...`.

Cùng seed 2027 nhưng GPU khác, bf16 và DataLoader nhiều worker không cho kết quả giống từng bit; lệch vài phần mười
điểm là bình thường. Lệch hơn khoảng 1 điểm thì kiểm SHA-256 dữ liệu / checkpoint và bản torch / timm.

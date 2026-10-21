# KD-MoE + ReviewKD (ABF + HCL, 4 tầng student <-> token BEATs layer 3/6/9/12 trên n_hat)

Nhánh `kdmoe-reviewkd`, thư mục `SNR_Aware/KDMoE_ReviewKD/`. Thư mục tự chứa: không import thư mục nào khác của repo.

```text
mixture x -> FRCRN (đóng băng) -> n_hat -> RMS 0.07925 -> student (encoder pretrained) -> MaskGate -> pooling
          -> router + expert -> 36 lớp
nhiễu sạch n -> BEATs teacher cross-fit 5 fold (đóng băng) -> logit 36 lớp -> KD (Hinton) lên 36 logit student
```

## Phương pháp

Như KD-MoE gốc + ReviewKD: ABF gộp đặc trưng 4 tầng student từ sâu lên nông, HCL so từng tầng với token BEATs layer 3 / 6 / 9 / 12 (teacher nghe n_hat) trên lưới 8 x 24; trọng số HCL warm-up 4 epoch. ABF và HCL theo code gốc dvlab-research/ReviewKD.

- ReviewKD: Chen et al., CVPR 2021, https://arxiv.org/abs/2104.09044

Giao thức chung mọi nhánh KD-MoE: seed 2026, 50 epoch, patience 8, stage 1 8 epoch, EMA 0.999, chọn epoch trên
validation sạch: 3 064 mixture còn lại sau khi bỏ các clip có nhiễu nằm trong AudioSet train
(`config/audioset_train_noise_ytids.txt`), vì encoder pretrained AudioSet đã thấy các clip đó.

## Các run

| Run | Student | Config run | Config model |
|---|---|---|---|
| `panns_reviewkd` | PANNs CNN10 | `config/train_panns_reviewkd.json` | `config/kdmoe_panns_cnn10_reviewkd.yaml` |
| `mn10_reviewkd` | EfficientAT mn10 | `config/train_mn10_reviewkd.json` | `config/kdmoe_mn10_reviewkd.yaml` |

## Chuẩn bị

1. **Dataset** mixture 36 lớp: HuggingFace dataset `Keshio1/36_labels` (file `mix-dataset.zip`), giải nén
   sao cho có `<DATASET>/manifest.csv`, `labels.txt`, `train/`, `validation/`, `test/`.
   Trên molab notebook có sẵn ô tải về `/marimo/dataset/mix-dataset`.
2. **Checkpoint FRCRN** (không có trong git, xin trực tiếp): đặt tại `model/frcrn_best.pt`, sha256
   `b8fdeb00a76289fa48451578a7a40aa437cdb9954683d578197dd0f8a184b14a`.
3. Các file còn lại `run.sh` tự tải: BEATs cpt2 (HuggingFace) vào `pretrained/`, PANNs CNN10 (zenodo) và
   repo EfficientAT (GitHub, commit cố định `a425fdc`, clone vào `/marimo/ext/EfficientAT`).
4. Python có torch + `requirements.txt` (molab: `/tmp/uv-venv/bin/python`, `run.sh` tự chọn nếu có).

## Chạy

Từ thư mục này:

```bash
mkdir -p output && DATASET=/marimo/dataset/mix-dataset nohup bash run.sh > output/run.log 2>&1 &
tail -f output/chain.log            # tiến độ từng bước: START <bước> ... rc=<mã thoát>
RUNS="panns_reviewkd" bash run.sh          # chỉ chạy một số run (cách nhau bằng dấu cách)
```

`run.sh` chạy tuần tự (1 GPU) và bỏ qua bước đã xong (cache, teacher, run đã có `stage2/result.json`):

1. **cache**: FRCRN đóng băng tính `n_hat` fp32 cho train/validation/test (`output/frcrn_cache_fp32/`) và SNR năng lượng FRCRN mỗi clip (`frcrn_{split}.snr.npy`).
2. **teacher**: BEATs iter3+ AS2M cpt2 fine-tune 36 lớp trên nhiễu sạch, cross-fit 5 fold theo `noise_ytid` (`config/teacher.json`) -> logit out-of-fold `output/teacher_unified/beats_teacher_logits_clean.pt` + trọng số 5 fold. Nếu chưa có, `run.sh` tải teacher dùng chung từ HuggingFace `Keshio1/frcrn-kdmoe-teacher` (private, cần `hf auth login` bằng tài khoản được cấp quyền; kiểm `SHA256SUMS`), tải không được mới train lại (~30 phút).
3. **features teacher**: token BEATs layer 3/6/9/12 trên n_hat (ReviewKD), từ trọng số 5 fold đã lưu.
4. **train**: lần lượt từng run ở bảng trên (`main.py train --config ...`), 2 giai đoạn: stage 1 chỉ học L_front, stage 2 toàn bộ loss; kết quả `output/runs/<run>/train/`.
5. **evaluate**: `evaluate.py` chấm validation + test theo từng mức SNR, tập sạch/rò AudioSet -> `output/eval/metrics_by_snr.csv` + `predictions_*.csv` (cho McNemar).

Nhánh này cần đặc trưng bên trong teacher nên cần thêm trọng số 5 fold (`output/teacher_unified/beats_teacher_fold*_weights.pt`, có trên repo HF); đặc trưng được xuất lại từ 5 fold (~10 phút). Teacher trên HF là bản train ngày 2026-10-21 (cùng config, seed 2026), chính là teacher của các run mn10 + AT / FitNets / ReviewKD; các run còn lại trong bảng tham chiếu dùng một lần train teacher trước đó (cùng config).

Log từng bước nằm ở `output/logs/`. Muốn kiểm nhanh trước khi chạy thật:
`python main.py smoke --config config/train_panns_reviewkd.json` (32 mẫu mỗi split, 1 epoch mỗi giai đoạn; cần cache
và logit teacher của bước 1-2).

## Kết quả tham chiếu

Accuracy trên validation sạch, trọng số EMA, chạy ngày 2026-10-20/21 trên molab (1 GPU), seed 2026. Chạy lại cho số gần như vậy (GPU không tất định từng bit).

| Run | Acc val sạch |
|---|---|
| `panns_reviewkd` | 0.521 |
| `mn10_reviewkd` | 0.545 |

## Kiểm thử

```bash
python -m pytest SNR_Aware/KDMoE_ReviewKD/tests -q     # chạy từ gốc repo, CPU
```

## Cấu trúc

```text
run.sh                        cả chuỗi: cache -> teacher -> train các run -> evaluate
main.py, online_pipeline.py   train / smoke / check student KD-MoE
evaluate.py                   đánh giá val + test theo SNR, tập sạch / rò AudioSet, dự đoán cho McNemar
teacher_crossfit.py           teacher BEATs cross-fit 5 fold + xuất đặc trưng (tiến trình riêng: crossfit / features / atmaps)
efficientat.py, external_repo.py   encoder EfficientAT mn10 (import repo gốc cô lập)
reviewkd.py                   ReviewKD: ABF + HCL, ReviewKDMoE / ReviewKDTrainer / ReviewKDMixDataset
src/, scripts/                KD-MoE, FRCRN, dataset, trainer, cache n_hat
teacher/                      hạ tầng teacher BEATs (teacher/beats: code BEATs gốc, giấy phép kèm theo)
config/                       config các run của nhánh + teacher + group_map, membership, frcrn.yaml
model/ pretrained/ output/    checkpoint và kết quả (ngoài git)
```

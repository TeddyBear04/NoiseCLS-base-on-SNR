# KD-MoE + Attention Transfer (năng lượng token / attention rollout / khoảng cách Pearson)

Nhánh `kdmoe-at`, thư mục `SNR_Aware/KDMoE_AT/`. Thư mục tự chứa: không import thư mục nào khác của repo.

```text
mixture x -> FRCRN (đóng băng) -> n_hat -> RMS 0.07925 -> student (encoder pretrained) -> MaskGate -> pooling
          -> router + expert -> 36 lớp
nhiễu sạch n -> BEATs teacher cross-fit 5 fold (đóng băng) -> logit 36 lớp -> KD (Hinton) lên 36 logit student
```

## Phương pháp

Như KD-MoE gốc + Attention Transfer: bản đồ attention 2 tầng student (khớp về lưới patch BEATs 24 x 8, phần <= 8 kHz) bám theo bản đồ teacher layer 6 / 12. `*_at_energy`: năng lượng token; `*_at_rollout`: attention rollout; `*_at_pearson`: rollout + khoảng cách Pearson thay cho L2.

- AT: Zagoruyko & Komodakis, ICLR 2017, https://arxiv.org/abs/1612.03928
- Attention rollout: Abnar & Zuidema, ACL 2020, https://arxiv.org/abs/2005.00928
- DIST (Pearson): Huang et al., NeurIPS 2022, https://arxiv.org/abs/2205.10536

Giao thức chung mọi nhánh KD-MoE: seed 2026, 50 epoch, patience 8, stage 1 8 epoch, EMA 0.999, chọn epoch trên
validation sạch: 3 064 mixture còn lại sau khi bỏ các clip có nhiễu nằm trong AudioSet train
(`config/audioset_train_noise_ytids.txt`), vì encoder pretrained AudioSet đã thấy các clip đó.

## Các run

| Run | Student | Config run | Config model |
|---|---|---|---|
| `panns_at_energy` | PANNs CNN10 | `config/train_panns_at_energy.json` | `config/kdmoe_panns_cnn10_at.yaml` |
| `panns_at_rollout` | PANNs CNN10 | `config/train_panns_at_rollout.json` | `config/kdmoe_panns_cnn10_at.yaml` |
| `panns_at_pearson` | PANNs CNN10 | `config/train_panns_at_pearson.json` | `config/kdmoe_panns_cnn10_at_pearson.yaml` |
| `mn10_at_energy` | EfficientAT mn10 | `config/train_mn10_at_energy.json` | `config/kdmoe_mn10_at.yaml` |
| `mn10_at_rollout` | EfficientAT mn10 | `config/train_mn10_at_rollout.json` | `config/kdmoe_mn10_at.yaml` |
| `mn10_at_pearson` | EfficientAT mn10 | `config/train_mn10_at_pearson.json` | `config/kdmoe_mn10_at_pearson.yaml` |

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
RUNS="panns_at_energy" bash run.sh          # chỉ chạy một số run (cách nhau bằng dấu cách)
```

`run.sh` chạy tuần tự (1 GPU) và bỏ qua bước đã xong (cache, teacher, run đã có `stage2/result.json`):

1. **cache**: FRCRN đóng băng tính `n_hat` fp32 cho train/validation/test (`output/frcrn_cache_fp32/`) và SNR năng lượng FRCRN mỗi clip (`frcrn_{split}.snr.npy`).
2. **teacher**: BEATs iter3+ AS2M cpt2 fine-tune 36 lớp trên nhiễu sạch, cross-fit 5 fold theo `noise_ytid` (`config/teacher.json`) -> logit out-of-fold `output/teacher_unified/beats_teacher_logits_clean.pt` + trọng số 5 fold. Nếu chưa có, `run.sh` tải teacher dùng chung từ HuggingFace `Keshio1/frcrn-kdmoe-teacher` (private, cần `hf auth login` bằng tài khoản được cấp quyền; kiểm `SHA256SUMS`), tải không được mới train lại (~30 phút).
3. **atmaps teacher_at_energy**: bản đồ AT năng lượng token layer 6/12, từ trọng số 5 fold đã lưu.
4. **atmaps teacher_at_rollout**: bản đồ AT attention rollout layer 6/12, từ trọng số 5 fold đã lưu.
5. **train**: lần lượt từng run ở bảng trên (`main.py train --config ...`), 2 giai đoạn: stage 1 chỉ học L_front, stage 2 toàn bộ loss; kết quả `output/runs/<run>/train/`.
6. **evaluate**: `evaluate.py` chấm validation + test theo từng mức SNR, tập sạch/rò AudioSet -> `output/eval/metrics_by_snr.csv` + `predictions_*.csv` (cho McNemar).

Nhánh này cần đặc trưng bên trong teacher nên cần thêm trọng số 5 fold (`output/teacher_unified/beats_teacher_fold*_weights.pt`, có trên repo HF); đặc trưng được xuất lại từ 5 fold (~10 phút). Teacher trên HF là bản train ngày 2026-10-21 (cùng config, seed 2026), chính là teacher của các run mn10 + AT / FitNets / ReviewKD; các run còn lại trong bảng tham chiếu dùng một lần train teacher trước đó (cùng config).

Log từng bước nằm ở `output/logs/`. Muốn kiểm nhanh trước khi chạy thật:
`python main.py smoke --config config/train_panns_at_energy.json` (32 mẫu mỗi split, 1 epoch mỗi giai đoạn; cần cache
và logit teacher của bước 1-2).

## Kết quả tham chiếu

Accuracy trên validation sạch, trọng số EMA, chạy ngày 2026-10-20/21 trên molab (1 GPU), seed 2026. Chạy lại cho số gần như vậy (GPU không tất định từng bit).

| Run | Acc val sạch |
|---|---|
| `panns_at_energy` | 0.518 |
| `panns_at_rollout` | 0.515 |
| `panns_at_pearson` | 0.512 |
| `mn10_at_energy` | 0.542 |
| `mn10_at_rollout` | 0.544 |
| `mn10_at_pearson` | 0.538 |

## Kiểm thử

```bash
python -m pytest SNR_Aware/KDMoE_AT/tests -q     # chạy từ gốc repo, CPU
```

## Cấu trúc

```text
run.sh                        cả chuỗi: cache -> teacher -> train các run -> evaluate
main.py, online_pipeline.py   train / smoke / check student KD-MoE
evaluate.py                   đánh giá val + test theo SNR, tập sạch / rò AudioSet, dự đoán cho McNemar
teacher_crossfit.py           teacher BEATs cross-fit 5 fold + xuất đặc trưng (tiến trình riêng: crossfit / features / atmaps)
efficientat.py, external_repo.py   encoder EfficientAT mn10 (import repo gốc cô lập)
src/, scripts/                KD-MoE, FRCRN, dataset, trainer, cache n_hat
teacher/                      hạ tầng teacher BEATs (teacher/beats: code BEATs gốc, giấy phép kèm theo)
config/                       config các run của nhánh + teacher + group_map, membership, frcrn.yaml
model/ pretrained/ output/    checkpoint và kết quả (ngoài git)
```

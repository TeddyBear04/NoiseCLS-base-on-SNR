# FRCRN + BEATs teacher (logit cache) + KD-MoE

Nhánh này hiện thực pipeline trong sơ đồ đầu vào:

```text
noise sạch n -> BEATs fine-tune đóng băng -> logits 36 lớp -> gom 4 nhóm -> KD router

mixture x -> FRCRN đóng băng -> n_hat -> LogMel + SpecAugment
          -> 2 HybridBlock + MaskGate -> ViT -> FreqPoolGRU
          -> router 4 nhóm + 4 expert 36 lớp -> logits 36 lớp
```

Teacher chạy trực tiếp trên `batch["noise"]` trong lúc train/validation. Không tạo hoặc đọc
teacher-logit bank. Khi suy luận chỉ cần `FRCRN -> KD-MoE`; teacher và noise sạch không còn
tham gia.

## Loss

```text
L = L_CE + lambda_KD L_KD + lambda_front L_front + lambda_cons L_cons

L_CE    : cross entropy với label smoothing 0.1 trên logits 36 lớp
L_KD    : T^2 KL giữa phân phối teacher sau khi gom 36 -> 4 nhóm và router student
L_front : L1 bất đối xứng giữa log-mel dự đoán và log-mel noise thật
L_cons  : 1 - cosine của embedding hai clip cùng noise_ytid
```

KD chỉ truyền gradient vào router student. BEATs teacher và FRCRN đều ở `eval`, toàn bộ tham
số bị đóng băng. `PairedClipSampler` tạo cặp cùng `noise_ytid` cho `L_cons`.

## Checkpoint mặc định

`config/train_config.json` dùng FRCRN đã cung cấp:

```text
D:\language\Project IV\Capstone\MLNC\model\best.pt
```

Đường dẫn được lưu tương đối (`../../MLNC/model/best.pt`) để repo vẫn chạy được sau khi di
chuyển cả workspace. Teacher mặc định dùng checkpoint noise-only hiện có ở
`../Mid_Expert/artifacts/teacher_noise_best.pt` và BEATs pretrained hiện có trong workspace.

## Chạy

Từ thư mục này:

```powershell
python main.py check
python main.py smoke
python main.py train
```

- `check`: nạp thật FRCRN và teacher, xác nhận checkpoint/36 nhãn và trạng thái đóng băng.
- `smoke`: 32 mẫu mỗi split, một epoch cho mỗi giai đoạn.
- `train`: huấn luyện đầy đủ hai giai đoạn; kết quả nằm trong `output/train/stage{1,2}`.

Giai đoạn 1 chỉ học frontend bằng `L_front`. Giai đoạn 2 nạp `stage1/best.pt` rồi tối ưu toàn
bộ loss. Dataset cần `manifest.csv`, `labels.txt`, waveform mixture/noise và các cột
`sample_id`, `split`, `mixture_path`, `noise_path`, `noise_ytid`, `label_mids`,
`multi_hot_36`, `target_snr_db`.

## Cấu trúc (tự chứa, không import thư mục khác)

```text
main.py, online_pipeline.py   train/smoke/check student KD-MoE (FRCRN -> n_hat -> KD-MoE)
evaluate.py                   đánh giá val + test theo SNR, tập sạch/rò AudioSet, dự đoán cho McNemar
reviewkd.py, efficientat.py   ReviewKD; encoder EfficientAT mn10 (external_repo.py: import repo gốc cô lập)
teacher_crossfit.py           teacher BEATs cross-fit 5 fold + xuất đặc trưng (tiến trình riêng)
src/, scripts/                KD-MoE, FRCRN, dataset, trainer (bản chép từ MLNC; scripts/frcrn_cache.py)
teacher/                      hạ tầng teacher BEATs (bản chép từ SNR_Expert_SSLAM, chỉ BEATs; teacher/beats)
config/                       mọi config + group_map, membership, frcrn.yaml, danh sách clip rò AudioSet
model/  pretrained/  output/  checkpoint FRCRN + logit Mid_Expert; PANNs/BEATs/EfficientAT; kết quả (ngoài git)
```

Teacher luôn đọc từ logit đã cache (`paths.teacher_cache`); bản ViT đời đầu (`train_config.json`) dùng logit
Mid_Expert trong `model/teacher_midexpert.pt` thay cho teacher online.

## Teacher cross-fit (`teacher_crossfit.py`)

Teacher BEATs iter3+ AS2M cpt2 fine-tune 36 lớp trên nhiễu sạch, công thức BEATs bảng 4 cột ESC-50,
chia 5 fold theo `noise_ytid`: logit/đặc trưng của mỗi mẫu train lấy từ fold KHÔNG thấy nó; validation là
trung bình 5 fold; mỗi fold chọn epoch trên val sạch (bỏ clip nằm trong AudioSet train). Dùng hạ tầng của
`../SNR_Expert_SSLAM` (Corpus, BEATs, vòng train) và chạy riêng khỏi `main.py` (gói `models` trùng tên).

```powershell
python -u teacher_crossfit.py crossfit --config config/teacher_crossfit_reviewkd.json   # 5 fold, lưu trọng số
python -u teacher_crossfit.py features --config config/teacher_crossfit_reviewkd.json   # token layer + logit
```

Đầu ra ở `output/teacher_crossfit*/`: `teacher_crossfit*.pt` (logit đóng gói), `fold{f}.pt` (trọng số khi
`crossfit.save_weights`), `teacher_hints_l{L}[_nhat].npy` + `.ids.json` (token layer L, lưới 24 x 8 x 768).
`crossfit.hint_source: n_hat` cho teacher nghe n_hat (cache FRCRN fp32) thay vì nhiễu sạch.

## Các biến thể KD (student `main.py train --config ...`)

| Config | Student | KD từ teacher cross-fit |
|---|---|---|
| `train_config_panns_cnn10_teachercf.json` | PANNs CNN10 pretrain, 3 expert theo SNR | logit (Hinton 2015) |
| `..._teachercf_at.json` / `_rollout.json` | như trên | + Attention Transfer: năng lượng token / attention rollout |
| `..._teachercf_atpearson.json` | như trên | + AT bằng khoảng cách Pearson (DIST) |
| `..._teachercf_fitnets.json` | như trên | + FitNets: block 3 -> token layer 6 |
| `train_config_panns_cnn10_reviewkd_r{0,1}.json` | như trên (`reviewkd.py`) | logit; R1 thêm ReviewKD (R0 = đối chứng cùng teacher) |
| `train_config_mn10_teachercf.json` | EfficientAT mn10_as pretrained (`efficientat.py`), 3 expert theo SNR | logit (cùng teacher với R0) |
| `train_config_mn10_teachercf_rms.json` | như trên + `model.rms_normalize: 0.07925` (RMS n_hat sau FRCRN) | như trên |

ReviewKD (`reviewkd.py`, Chen et al. CVPR 2021, theo beat-reviewkd.pdf nhưng student là PANNs CNN10 pretrained):
ABF + HCL theo code gốc; block 1-4 CNN10 so với token BEATs layer 3/6/9/12 (teacher nghe n_hat), mỗi bản đồ
cắt phần <= 8 kHz / 3.84 s đầu rồi đưa về lưới 8 x 24. Thứ tự chạy: `teacher_crossfit.py crossfit` ->
`features` -> `main.py train` R0, R1.

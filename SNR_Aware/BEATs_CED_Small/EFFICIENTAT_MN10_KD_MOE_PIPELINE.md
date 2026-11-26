# Đặc tả tái lập: EfficientAT mn10 + KD-MoE

Tài liệu này mô tả **đúng run `Unified mn10`**, không phải các biến thể `mn10_at_*`,
`mn10_fitnets`, `mn10_reviewkd` hay `mn10_snrroute`. Mục tiêu là đủ chi tiết để một
người khác có thể viết lại pipeline mà không cần đoán các setting ngầm trong code.

Run tham chiếu được thực hiện ngày 20–21/10/2026, seed 2026, trên commit workspace
`b53f316` cộng các thay đổi chưa commit. Cấu hình gốc là:

- `config/unified/teacher.json`: teacher BEATs cross-fit.
- `config/unified/train_mn10.json`: đường dẫn và dữ liệu cho student.
- `config/unified/kdmoe_mn10.yaml`: kiến trúc, loss và tối ưu student.

> **Điểm nhận dạng quan trọng nhất:** student không nghe mixture trực tiếp. Mixture đi
> qua FRCRN đóng băng để lấy nhiễu ước lượng `n_hat`; `n_hat` được cache bằng fp32,
> chuẩn hóa RMS về `0.07925`, rồi mới đưa vào EfficientAT `mn10_as`. Teacher nghe
> **nhiễu sạch**, tạo logit 36 lớp bằng cross-fit 5 fold. KD được áp dụng lên **36 logit
> cuối của student**, không áp dụng lên router.

## 1. Pipeline tổng quát

### 1.1. Khi huấn luyện

```text
                                        ┌──────────────────────────────────┐
noise sạch n ──────────────────────────►│ BEATs teacher, 5-fold cross-fit  │
                                        │ input: n; output: 36 logits      │
                                        └────────────────┬─────────────────┘
                                                         │ cache OOF/mean fp16
                                                         ▼
mixture x ─► FRCRN frozen ─► n_hat ─► RMS normalize ─► EfficientAT log-mel
                                                       ─► mn10_as encoder
                                                       ─► channel mask gate
                                                       ─► pooled embedding e
                                                       ├► router: 3 SNR bands
                                                       └► 3 experts × 36 logits
                                                              │
                         clean label, true SNR band, teacher logits, clean noise
                                                              │
                                                              ▼
                        CE + KD36 + router-CE + front reconstruction + consistency
```

FRCRN trong checkpoint này dự đoán speech. Wrapper trả về nhiễu ước lượng:

```text
speech_hat = FRCRN_network(x)
n_hat      = x - speech_hat
```

### 1.2. Khi suy luận

```text
x -> FRCRN -> n_hat -> RMS normalize -> EfficientAT frontend + mn10
  -> mask gate -> embedding -> router + 3 experts -> weighted 36 logits -> argmax
```

Không cần noise sạch, teacher, `NoiseHead`, teacher cache hay nhãn SNR khi suy luận.
Router tự dự đoán band SNR từ embedding. Đây không phải biến thể `snrroute`, vốn gán
expert bằng SNR năng lượng FRCRN.

## 2. Dữ liệu và giao thức chia tập

### 2.1. Dữ liệu gốc của run

- Task: single-label noise classification, 36 lớp.
- Speech: VLSP 2020.
- Noise: các clip AudioSet thuộc 36 nhãn sau khi loại 4 nhãn `Bass drum`, `Cello`,
  `Double bass`, `Snare drum`.
- Audio: mono, IEEE float32 WAV, 16 kHz, đúng 4.0 s = 64.000 samples.
- SNR đích: `[-5, 0, 5, 10, 15, 20]` dB.
- Seed khi dựng dataset: 42.
- Mỗi cặp label–SNR có 200 mẫu: train 140, validation 30, test 30.

| Split | Tổng mẫu | Mẫu mỗi SNR | Noise clip khác nhau |
|---|---:|---:|---:|
| train | 30.240 | 5.040 | 2.692 |
| validation | 6.480 | 1.080 | 579 |
| test | 6.480 | 1.080 | 574 |

Mỗi label có số mẫu bằng nhau trong từng mức SNR. Mixture được dựng theo:

```text
P_s = mean(speech^2) + 1e-8
P_n = mean(noise^2)  + 1e-8
noise_scaled = noise * sqrt(P_s / (P_n * 10^(SNR/10)))
mixture = speech + noise_scaled
if max(abs(mixture)) > 1:
    mixture      /= peak
    noise_scaled /= peak
```

### 2.2. Cấu trúc thư mục bắt buộc

```text
dataset_root/
├── labels.txt
├── manifest.csv                 # toàn bộ 43.200 hàng; teacher dùng file này
├── train/
│   ├── manifest.csv             # student dùng file theo split
│   ├── clean/*.wav
│   ├── noise/*.wav
│   └── mixture/*.wav
├── validation/
│   └── ...
└── test/
    └── ...
```

Các cột tối thiểu của manifest:

```text
sample_id, split, clean_path, noise_path, mixture_path,
speech_id, noise_ytid, label_mids, label_names, multi_hot_36,
target_snr_db
```

`sample_id` phải duy nhất và ổn định vì nó là khóa nối giữa manifest, FRCRN cache và
teacher cache. `noise_ytid` phải ổn định vì nó được dùng để chia fold teacher và tạo
cặp consistency.

### 2.3. Thứ tự 36 lớp

Chỉ số class là số dòng, bắt đầu từ 0, trong `labels.txt`:

```text
0  Acoustic guitar
1  Applause
2  Bark
3  Burping, eructation
4  Bus
5  Chime
6  Clarinet
7  Computer keyboard
8  Cough
9  Cowbell
10 Drawer open or close
11 Electric piano
12 Fart
13 Finger snapping
14 Fireworks
15 Flute
16 Glockenspiel
17 Gong
18 Gunshot, gunfire
19 Harmonica
20 Hi-hat
21 Keys jangling
22 Knock
23 Laughter
24 Meow
25 Microwave oven
26 Saxophone
27 Scissors
28 Shatter
29 Squeak
30 Tambourine
31 Tearing
32 Telephone
33 Trumpet
34 Violin, fiddle
35 Writing
```

Không được đổi thứ tự này: head teacher, teacher logits, nhãn student và output 36
logit đều dùng cùng thứ tự.

### 2.4. Validation “clean” dùng để chọn epoch

BEATs và EfficientAT được pretrain trên AudioSet. Vì vậy các hàng validation có
`noise_ytid` nằm trong AudioSet train được đánh dấu là `leaked` và **không dùng để
chọn checkpoint**.

- Danh sách loại: `config/audioset_train_noise_ytids.txt`.
- Validation đầy đủ: 6.480 mẫu.
- Validation clean: 3.064 mẫu.
- Validation leaked bị loại: 3.416 mẫu.
- Train vẫn dùng đủ 30.240 mẫu.
- Báo cáo cuối vẫn chấm cả `all`, `clean` và `leaked`; test không bị lọc khi suy luận.

## 3. Teacher BEATs cross-fit 5 fold

### 3.1. Backbone và đầu vào

- Backbone: `BEATs_iter3_plus_AS2M_finetuned_on_AS2M_cpt2.pt`.
- Input teacher: cột `noise_path`, tức noise sạch 16 kHz, 4 s.
- Embedding: 768.
- Pooling: mean của patch tokens.
- Head: Linear 768 → 36.
- Khởi tạo head: lấy đúng 36 hàng tương ứng từ AudioSet predictor trong checkpoint.
- Dropout: 0.0.
- Encoder layer-drop: 0.1.
- Có 12 transformer blocks và cả 12 block được fine-tune.
- Patch embedding và các phần encoder ngoài block vẫn frozen; head 36 lớp được train.
- Loss teacher: cross entropy thường, không label smoothing, không KD.
- Teacher không dùng FiLM, attention mask, CRD hoặc SNR làm đầu vào.

### 3.2. Cách chia fold

Không chia ngẫu nhiên theo từng row. Tất cả segment có cùng `noise_ytid` phải nằm
trong cùng fold:

```python
ytids = sorted(unique_train_noise_ytids)
np.random.default_rng(2026).shuffle(ytids)
fold_of = {ytid: index % 5 for index, ytid in enumerate(ytids)}
```

Với dataset gốc:

| Fold | Số noise_ytid held-out | Row held-out | Row train teacher |
|---:|---:|---:|---:|
| 0 | 539 | 6.012 | 24.228 |
| 1 | 539 | 6.124 | 24.116 |
| 2 | 538 | 6.007 | 24.233 |
| 3 | 538 | 5.962 | 24.278 |
| 4 | 538 | 6.135 | 24.105 |

Mỗi fold train trên bốn fold còn lại, sau đó dự đoán phần held-out. Vì vậy logit
teacher của train là out-of-fold (OOF): teacher sinh logit cho một row chưa từng học
row/noise clip đó. Logit validation là trung bình **logit thô** của cả 5 teacher,
không phải trung bình probability.

### 3.3. Tối ưu teacher

| Setting | Giá trị |
|---|---:|
| Seed | 2026 |
| Epoch tối đa | 30 mỗi fold |
| Patience | 5 |
| Batch train / validation | 64 / 64 |
| Gradient accumulation | 1 |
| Optimizer | AdamW |
| Encoder LR / head LR | `3e-5` / `3e-5` |
| Betas | `(0.9, 0.98)` |
| Weight decay | `0.01` |
| Layer-wise LR decay | `0.2` |
| Gradient clip norm | `5.0` |
| Schedule | linear warm-up 10%, sau đó cosine về 0 |
| Precision | CUDA bf16 autocast |
| TF32 | bật trong teacher process |
| Augment | circular waveform roll + 1 time mask và 1 freq mask, width ngẫu nhiên tối đa 20% trục |
| Mixup | không |
| Tiêu chí chọn epoch | macro-F1 trên 3.064 validation clean |
| Ngưỡng cải thiện | `1e-4` |

Layer-wise LR của block `i` từ dưới lên là:

```text
lr_i = 3e-5 * 0.2^(11-i), i = 0..11
```

Tham số relative-position dùng chung nhiều block chỉ xuất hiện trong optimizer group
của block cao nhất sử dụng nó.

### 3.4. Artifact teacher

File `teacher_unified.pt` có cấu trúc logic:

```python
{
  "source": "BEATs crossfit 5 folds, ...",
  "precision": "fp16 logits",
  "label_order": "labels.txt",
  "train": {
    "mix_id": [...],            # 30.240 sample_id
    "logits36": float16[N, 36], # OOF
    "snr_db": float32[N],
    "accuracy": ...,
    "accuracy_clean": ...
  },
  "validation": {
    "mix_id": [...],            # 6.480 sample_id
    "logits36": float16[N, 36], # mean logits của 5 fold
    "snr_db": float32[N],
    ...
  }
}
```

Kết quả teacher tham chiếu:

| Fold | Best epoch | Clean-val accuracy | Clean-val macro-F1 |
|---:|---:|---:|---:|
| 0 | 1 | 0.7405 | 0.7118 |
| 1 | 1 | 0.7409 | 0.7162 |
| 2 | 1 | 0.7389 | 0.7095 |
| 3 | 1 | 0.7363 | 0.7092 |
| 4 | 1 | 0.7425 | 0.7138 |

Accuracy của gói cuối: train OOF 0.7645; train OOF clean 0.7304; validation all
0.7765; validation clean 0.7431.

> Sau khi hoàn tất cross-fit, student dùng `output/teacher_unified/teacher_unified.pt`.
> Không dùng `teacher_unified_nhat.pt`: file thứ hai là teacher chạy trên `n_hat` để
> phục vụ các thí nghiệm feature distillation, không phải teacher của run `Unified mn10`.

## 4. FRCRN và cache `n_hat`

### 4.1. Kiến trúc/checkpoint

FRCRN có các setting:

```yaml
model:
  win_len: 640          # mặc định trong dataclass
  win_inc: 320
  fft_len: 640
  target: speech
  target_level_db: -25.0
```

Network dùng convolutional STFT, hai complex UNet nối tiếp và complex ratio mask:

```text
mask = tanh(UNet1(X)) + tanh(UNet2(UNet1(X)))
speech_hat = iSTFT(X * complex_mask)
n_hat = mixture - speech_hat
```

Checkpoint phải tương thích hoàn toàn với model, không thiếu hoặc thừa key. Tất cả
tham số FRCRN ở `eval()` và `requires_grad=False`.

### 4.2. Quy tắc tạo cache

- Đọc đủ 4 s, không random crop.
- Batch cache: 64; workers: 4.
- Chạy FRCRN trong fp32 thật.
- Tắt cả cuDNN TF32 và CUDA matmul TF32 trong lúc tạo cache.
- Lưu `float32 [N, 64000]` vào `frcrn_{split}.npy`.
- Lưu danh sách `sample_id` cùng thứ tự vào `frcrn_{split}.json`.
- Train student đọc bằng NumPy memmap trong từng worker.
- Với run này, `.snr.npy` có thể được tạo nhưng chỉ cần cho biến thể `snrroute`; router
  học của `Unified mn10` không dùng `snr_est`.

Cache train và validation được tạo trước khi train; cache test được tạo lúc evaluate.
Việc cache trước tránh FRCRN chạy trong autocast bf16 và đảm bảo một `n_hat` cố định
cho mỗi clip.

## 5. Student EfficientAT mn10 + KD-MoE

### 5.1. EfficientAT frontend

Input ban đầu là `n_hat` 16 kHz. Với mỗi clip:

1. Tính RMS trên waveform fp32.
2. Nhân `n_hat` với `0.07925 / (RMS + 1e-8)`.
3. Nếu đang train với noise sạch làm target front loss, nhân noise sạch bằng **cùng
   hệ số**, không chuẩn hóa noise độc lập.
4. Resample từ 16 kHz lên 32 kHz bằng `torchaudio.functional.resample` trong fp32.
5. Dùng `EfficientAT.models.preprocess.AugmentMelSTFT` với:

```yaml
sample_rate: 32000
n_mels: 128
win_length: 800
hopsize: 320
n_fft: 1024
freqm: 0
timem: 0
fmin: 0.0
fmax: null
fmin_aug_range: 10
fmax_aug_range: 2000
preemphasis: [-0.97, 1.0]
log_epsilon: 1.0e-5
normalization: "(log_mel + 4.5) / 5"
```

Khi train, frontend EfficientAT ngẫu nhiên hóa biên fmin/fmax theo implementation gốc;
khi eval dùng biên cố định. SpecAugment riêng của KD-MoE bị tắt hoàn toàn (`0` mask).
Mixture estimate và noise target được ghép batch rồi đi chung một lần qua mel frontend,
nên nhận cùng fmin/fmax ngẫu nhiên.

Output mel được center-crop còn đúng 384 frames. Với hop 320 ở 32 kHz, đây là 3.84 s.
Shape vào CNN là `[B, 1, 128, 384]`.

### 5.2. Encoder và embedding

- Repo: `https://github.com/fschmid56/EfficientAT.git`.
- Commit cố định: `a425fdc`.
- Variant pretrained: `mn10_as`.
- MobileNetV3 width multiplier: 1.0.
- Pretrain: AudioSet, model gốc 527 lớp.
- Dùng `head_type="mlp"`.
- Chỉ nạp `features.*` và lớp `classifier.2` từ checkpoint pretrained.
- Bỏ lớp classifier 527 AudioSet.

Luồng tensor:

```text
[B, 1, 128, 384]
  -> mn10 features, 17 stage, stride tổng 32
  -> H = [B, 960, 4, 12]
  -> permute tokens [B, 4, 12, 960]
  -> MaskGate
  -> global mean trên lưới 4x12
  -> Linear(960, 1280) + Hardswish + Dropout(p=0.2)
  -> embedding e [B, 1280]
```

Nếu viết lại, phải lấy dropout và lớp Linear từ đúng MLP head của EfficientAT tại
commit đã pin, thay vì tự dựng một head có initialization khác.

### 5.3. Mask gate và front reconstruction head

Gate hoạt động trên từng vị trí tần số–thời gian và từng channel:

```text
g = 0.05 + 0.95 * sigmoid(W H + b)
H_gated = H * g
```

Bias của Linear gate khởi tạo bằng 2.0. Không có softmax qua token/channel.

`NoiseHead` chiếu mỗi token gated từ 960 → `32*32 = 1024`, rồi pixel-shuffle thủ
công lưới `4 x 12` để dựng lại log-mel `[B, 1, 128, 384]`. Head này chỉ cần khi có
noise sạch để tính front loss; inference không chạy head này.

### 5.4. Router và experts

True SNR band dùng cho router supervision:

```text
group 0: SNR < 5 dB       -> {-5, 0}
group 1: 5 <= SNR < 15    -> {5, 10}
group 2: SNR >= 15 dB     -> {15, 20}
```

Router:

```text
Linear(1280, 64) -> GELU -> Linear(64, 3)
```

Bias lớp cuối router khởi tạo 0.

Ba expert độc lập, mỗi expert xuất đủ 36 lớp:

```text
Linear(1280, 256) -> GELU -> Dropout(0.1) -> Linear(256, 36)
```

Expert `k` khởi tạo hai Linear bằng generator CPU seed `k` (`0, 1, 2`), uniform
theo bound `1/sqrt(in_features)`. Đây là initialization riêng của code gốc.

Ghép output:

```text
p_route = softmax(router_logits, dim=3 experts)
student_logits36 = sum_k p_route[k] * expert_logits[k]
```

`inference_mode: weighted`, nên cả train lẫn eval đều dùng weighted mixture; không
chuyển sang top-1 expert ở inference.

## 6. Hàm loss chính xác

### 6.1. Stage 2

```text
L_total = L_CE
        + 1.0 * L_KD36
        + 1.0 * L_route
        + 1.0 * L_front
        + 0.1 * L_cons
```

Không có AT, FitNets hoặc ReviewKD trong run này.

#### Classification CE

```text
L_CE = CrossEntropy(student_logits36, label, label_smoothing=0.1)
```

#### Hinton KD trên 36 lớp

Với `T=2`:

```text
p_t = softmax(teacher_logits36 / T)
log_p_s = log_softmax(student_logits36 / T)
L_KD36 = mean_batch[T^2 * sum_c p_t[c] * (log(p_t[c]) - log_p_s[c])]
```

Teacher logits được `detach`, và mọi phép tính loss dùng fp32. `kd_weight_by_snr` tắt.
KD tác động lên final 36 logits, do đó gradient đi qua cả experts, router và encoder.
Teacher không cung cấp target trực tiếp cho router.

#### Router CE

```text
L_route = CrossEntropy(router_logits3, true_snr_band)
```

Weight `1.0`. True band lấy từ `target_snr_db` trong manifest.

#### Asymmetric front loss

Gọi `Y_hat` là log-mel do NoiseHead dựng lại và `Y` là log-mel noise sạch sau cùng
RMS scale và cùng frontend. Với `d = Y_hat - Y`:

```text
weight(d) = 2.0 if d > 0 else 1.0
L_front = mean(weight(d) * abs(d))
```

Phần dự đoán thừa bị phạt gấp đôi.

#### Consistency loss

Sampler cố đưa hai mixture có cùng `noise_ytid` vào batch. Chuẩn hóa L2 embedding
1280 chiều, rồi với mỗi nhóm có ít nhất hai mẫu:

```text
L_pair = mean_{i<j}(1 - cosine(e_i, e_j))
L_cons = mean_over_noise_groups(L_pair)
```

Nhóm chỉ có một mẫu không đóng góp. Weight trong total loss là `0.1`.

### 6.2. Stage 1

Stage 1 tối ưu **chỉ**:

```text
L_stage1 = L_front
```

Với EfficientAT, các prefix sau bị freeze: `encoder`, `post_fc`, `router`, `experts`
(cùng các module ViT không tồn tại trong model này). Vì vậy stage 1 thực tế chỉ học
`MaskGate` và `NoiseHead`. Sau stage 1, nạp raw weights từ `stage1/best.pt`, mở toàn
bộ tham số và bắt đầu optimizer/scheduler/EMA mới cho stage 2.

## 7. Batching, optimizer, schedule và checkpoint

### 7.1. Train sampler

`paired_units_per_batch = 32`. Một unit là:

- 2 row cùng `noise_ytid`, nếu clip còn ít nhất hai row; hoặc
- 1 row lẻ.

Mỗi epoch, row trong từng noise clip được shuffle và ghép thành các cặp rời nhau.
Các unit được shuffle trong từng label rồi xen kẽ label để cân bằng batch.

Với dataset gốc:

- 15.790 unit/epoch.
- 494 optimizer steps/epoch.
- Batch vật lý thay đổi từ 32 đến tối đa 64 row, tùy số unit đôi.
- `train.batch_size: 64` trong YAML **không điều khiển train loader** khi batch sampler
  này được dùng; nó chỉ là config danh nghĩa. Eval batch thật là 64.
- Không có gradient accumulation ở student: mỗi physical batch là một optimizer step.

### 7.2. Optimizer chung cho cả hai stage

| Setting | Giá trị |
|---|---:|
| Optimizer | AdamW |
| Peak LR | `6e-5` |
| Betas | `(0.9, 0.999)` |
| Weight decay | `0.05` |
| No-decay | parameter có tên chứa `bias`, `ln`, `norm`, `pos`, `a_att`, `a_gru` |
| Warm-up | 5% tổng step của stage |
| Schedule | cosine từ peak LR về `1e-6` |
| Gradient clip norm | `1.0` |
| Forward autocast | bf16 trên CUDA |
| Mel/resample | ép fp32 |
| GradScaler | không dùng |
| EMA | decay `0.999`, update sau mỗi optimizer step |
| Resume | ghi mỗi epoch |

Stage 1 có 8 epoch và schedule riêng `8 * 494` step. Stage 2 có tối đa 50 epoch và
schedule riêng `50 * 494` step, kể cả khi early-stop trước đó.

### 7.3. Epoch selection

- Stage 1 chọn theo `front_score = -L_front` trên validation clean.
- Stage 2 chọn theo `macro_acc` trên validation clean.
- `macro_acc` là trung bình của 6 balanced accuracies, mỗi giá trị tính riêng tại
  một SNR trong `[-5, 0, 5, 10, 15, 20]`.
- Patience stage 2: 8 epoch không cải thiện.
- Validation luôn chạy bằng **EMA weights**.
- Khi EMA score tốt nhất, `best.pt` lại lưu **raw model weights tại epoch đó**, không
  lưu EMA. Đây là hành vi gốc cần giữ nếu muốn khớp artifact.
- `resume.pt["ema"]` là EMA ở epoch cuối đã chạy, không nhất thiết là EMA ở best epoch.

Run tham chiếu:

- Stage 1 chạy đủ 8 epoch.
- Stage 2 best epoch = 9.
- Stage 2 early-stop sau epoch 17.
- Báo cáo `best_raw` dùng raw weights tại epoch 9.
- Báo cáo `ema_last` dùng EMA tại epoch 17.

### 7.4. Seed

- Teacher cross-fit seed: 2026.
- Student runtime seed: 2026; giá trị này ghi đè `train.seed: 42` trong YAML.
- Seed Python `random`, NumPy, torch CPU và mọi CUDA device.
- Paired sampler dùng seed 2026 và cộng số epoch.
- Code gốc không bật `torch.use_deterministic_algorithms`; do đó cùng seed chưa đảm
  bảo bitwise-identical giữa GPU/CUDA/PyTorch khác nhau.

## 8. Cấu hình hợp nhất để viết lại

Đây là bản canonical, gom các file gốc vào một chỗ. Nó là đặc tả, không phải file mà
`main.py` hiện tại đọc trực tiếp.

```yaml
experiment:
  name: unified_mn10
  task: single_label_noise_classification
  num_classes: 36
  runtime_seed: 2026

data:
  sample_rate: 16000
  clip_seconds: 4.0
  train_split: train
  validation_split: validation
  test_split: test
  snr_levels_db: [-5, 0, 5, 10, 15, 20]
  num_workers: 4
  paired_units_per_batch: 32
  validation_exclude_noise_ytids: config/audioset_train_noise_ytids.txt

teacher:
  backbone: beats
  checkpoint: BEATs_iter3_plus_AS2M_finetuned_on_AS2M_cpt2.pt
  input_column: noise_path
  embedding_dim: 768
  head_init: audioset_rows_for_36_labels
  pooling: mean_patches
  dropout: 0.0
  layerdrop: 0.1
  folds: 5
  group_folds_by: noise_ytid
  fold_seed: 2026
  trainable_transformer_blocks: 12
  epochs: 30
  patience: 5
  batch_size: 64
  eval_batch_size: 64
  accumulation_steps: 1
  optimizer: AdamW
  encoder_lr: 3.0e-5
  head_lr: 3.0e-5
  layer_decay: 0.2
  betas: [0.9, 0.98]
  weight_decay: 0.01
  gradient_clip_norm: 5.0
  schedule: warmup_cosine_to_zero
  warmup_fraction: 0.1
  precision: bf16
  augment:
    circular_roll: true
    spec_ratio: 0.2
    mixup: false
  loss: cross_entropy
  select_metric: macro_f1
  select_subset: validation_clean
  min_improvement: 1.0e-4
  output_logits_precision: fp16
  train_logits: out_of_fold
  validation_logits: mean_of_5_fold_logits

frcrn:
  frozen: true
  checkpoint: model/frcrn_best.pt
  predicts: speech
  student_input: mixture_minus_speech_estimate
  win_len: 640
  win_inc: 320
  fft_len: 640
  target_level_db: -25.0
  cache_dtype: float32
  cache_batch_size: 64
  allow_tf32_during_cache: false

student:
  type: efficientat_kdmoe
  repo_url: https://github.com/fschmid56/EfficientAT.git
  repo_commit: a425fdc
  variant: mn10_as
  width_mult: 1.0
  pretrained_num_classes: 527
  input_rms: 0.07925
  frontend:
    internal_sample_rate: 32000
    n_mels: 128
    win_length: 800
    hopsize: 320
    n_fft: 1024
    fmin: 0.0
    fmax: null
    fmin_aug_range: 10
    fmax_aug_range: 2000
    freqm: 0
    timem: 0
    center_crop_frames: 384
  encoder_output: [960, 4, 12]
  embedding_dim: 1280
  embedding_dropout: 0.2
  gate:
    enabled: true
    eps: 0.05
    bias_init: 2.0
  noise_head:
    token_dim: 960
    patch_size: 32
    output_shape: [1, 128, 384]
  router:
    input_dim: 1280
    hidden_dim: 64
    num_groups: 3
    route_by: true_snr_band
    band_edges_db: [5.0, 15.0]
  experts:
    count: 3
    input_dim: 1280
    hidden_dim: 256
    output_dim_each: 36
    dropout: 0.1
  inference_mode: weighted

loss:
  ce:
    weight: 1.0
    label_smoothing: 0.1
  kd36:
    weight: 1.0
    temperature: 2.0
    teacher_input: clean_noise
    target: final_student_logits_36
    weight_by_snr: false
  route_ce:
    weight: 1.0
    target: true_snr_band
  front:
    weight: 1.0
    kind: asymmetric_l1
    overestimate_multiplier: 2.0
  consistency:
    weight: 0.1
    kind: one_minus_cosine
    pair_key: noise_ytid
  attention_transfer_weight: 0.0
  fitnets_weight: 0.0
  reviewkd_weight: 0.0

training:
  stages:
    - name: front_pretrain
      epochs: 8
      trainable: [gate, noise_head]
      loss: front_only
      selection: negative_front_loss_on_clean_validation_using_ema
    - name: full_finetune
      epochs: 50
      patience: 8
      trainable: all
      loss: total
      selection: macro_acc_on_clean_validation_using_ema
  optimizer: AdamW
  lr: 6.0e-5
  min_lr: 1.0e-6
  warmup_ratio: 0.05
  schedule: warmup_cosine
  betas: [0.9, 0.999]
  weight_decay: 0.05
  gradient_clip_norm: 1.0
  amp: bf16
  ema_decay: 0.999
  eval_batch_size: 64
  resume_every_epochs: 1
```

## 9. Hai config student nguyên bản

### 9.1. Entrypoint JSON

Chỉnh `dataset` theo máy mới nếu không giữ nguyên workspace layout.

```json
{
  "paths": {
    "dataset": "../../.agents/36_labels",
    "frcrn_checkpoint": "model/frcrn_best.pt",
    "frcrn_config": "config/frcrn.yaml",
    "student_config": "config/unified/kdmoe_mn10.yaml",
    "group_map": "config/group_map.json",
    "membership": "config/membership.npy",
    "teacher_cache": "output/teacher_unified/teacher_unified.pt",
    "frcrn_cache_dir": "output/frcrn_cache_fp32",
    "output": "output/unified/mn10",
    "validation_exclude_noise_ytids": "config/audioset_train_noise_ytids.txt"
  },
  "data": {
    "train_split": "train",
    "validation_split": "validation",
    "clip_seconds": 4.0,
    "num_workers": 4,
    "paired_units_per_batch": 32
  },
  "runtime": {
    "device": "auto",
    "smoke_samples": 32,
    "seed": 2026
  }
}
```

### 9.2. Model/loss/train YAML

```yaml
model:
  rms_normalize: 0.07925
  encoder: panns_cnn10       # chỉ chọn code path CNN; lớp thật bị EfficientAT thay thế
  panns_checkpoint: ""
  use_gate: true
  gate_eps: 0.05
  gate_bias_init: 2.0
  patch_size: 32
  freq_bins: 128
  num_frames: 384
  num_classes: 36
  route_by: snr
  snr_band_edges: [5.0, 15.0]
  num_groups: 3
  expert_hidden: 256
  router_hidden: 64
  dropout: 0.1
  inference_mode: weighted
  specaug_freq_masks: 0
  specaug_freq_width: 0
  specaug_time_masks: 0
  specaug_time_width: 0

frontend:
  sample_rate: 16000

loss:
  label_smoothing: 0.1
  kd_temperature: 2.0
  kd_weight: 1.0
  kd_target: logits
  kd_weight_by_snr: false
  route_ce_weight: 1.0
  front_weight: 1.0
  front_lam_over: 2.0
  cons_weight: 0.1

train:
  lr: 6.0e-5
  weight_decay: 0.05
  betas: [0.9, 0.999]
  warmup_ratio: 0.05
  min_lr: 1.0e-6
  grad_clip: 1.0
  batch_size: 64
  eval_batch_size: 64
  epochs: 50
  patience: 8
  stage1_epochs: 8
  ema_decay: 0.999
  amp: bf16
  seed: 42                 # bị runtime.seed=2026 ghi đè
  resume_every: 1

student:
  type: efficientat
  variant: mn10_as
  width_mult: 1.0
  repo_url: https://github.com/fschmid56/EfficientAT.git
  commit: a425fdc
  repo_dir: /marimo/ext/EfficientAT
```

`group_map.json` và `membership.npy` vẫn được entrypoint yêu cầu. Tuy nhiên với
`route_by: snr` và `kd_target: logits`, class-group membership không tham gia công
thức loss của run này; group target được dựng trực tiếp từ SNR.

## 10. Cài đặt và lệnh chạy

### 10.1. Môi trường

Project yêu cầu Python >= 3.10 và các package tối thiểu:

```text
numpy>=1.26
scipy>=1.11
scikit-learn>=1.3
torch>=2.2
torchaudio>=2.2
soundfile>=0.12
pyyaml>=6.0
requests>=2.31
tqdm>=4.66
```

Ngoài ra cài requirements của EfficientAT tại commit `a425fdc`. Run gốc ghi nhận:

```text
torch: 2.11.0+cu130
GPU: NVIDIA RTX PRO 6000 Blackwell Server Edition
```

Các version base trong repository là lower bounds, không phải lockfile. Muốn so khớp
số tuyệt đối tốt nhất nên dùng đúng PyTorch/CUDA trên.

### 10.2. Chạy từ đầu: chỉ có data + FRCRN checkpoint + BEATs checkpoint

Người tái lập **không cần có sẵn** `teacher.pt`, `teacher_unified.pt`,
`teacher_{split}.npz`, hay `frcrn_{split}.npy`. Các artifact này là output trung gian
và phải được tạo theo thứ tự dưới đây:

```text
data + BEATs.pt
  -> teacher_crossfit.py crossfit
  -> output/teacher_unified/teacher_unified.pt
  -> main.py train (tự unpack thành teacher_{train,validation}.npz)

data + FRCRN.pt
  -> main.py train (tự tạo output/frcrn_cache_fp32/frcrn_{train,validation}.npy)
  -> student EfficientAT mn10 KD-MoE
```

Hai file checkpoint ban đầu cần đặt ở:

```text
FRCRN_KD_MoE/model/frcrn_best.pt
FRCRN_KD_MoE/pretrained/BEATs_iter3_plus_AS2M_finetuned_on_AS2M_cpt2.pt
dataset_root/...
```

Tên file có thể là `FRCRN.pt` và `BEATs.pt` trên máy người tái lập, nhưng phải sửa
`frcrn_checkpoint` trong `train_mn10.json` và `checkpoint_candidates` trong
`teacher.json` để trỏ đúng file. Không thay checkpoint bằng một checkpoint cùng tên
nhưng khác trọng số nếu mục tiêu là so khớp run tham chiếu.

Sửa hai đường dẫn máy-specific:

- `config/unified/teacher.json -> dataset.path`.
- `config/unified/kdmoe_mn10.yaml -> student.repo_dir`.

`repo_dir` có thể chưa tồn tại; code sẽ clone repo và checkout commit lần đầu. Việc
tải checkpoint `mn10_as` cũng cần network lần đầu. Vì vậy ngoài data + FRCRN + BEATs,
người chạy cần cho phép tải EfficientAT repo/weights hoặc nhận sẵn bản clone và
pretrained weight đã pin đúng commit.

### 10.3. Chạy teacher từ đầu

Từ thư mục `SNR_Aware/FRCRN_KD_MoE`:

```powershell
python -u teacher_crossfit.py crossfit --config config/unified/teacher.json
```

Kết quả cần có:

```text
output/teacher_unified/fold0.pt ... fold4.pt
output/teacher_unified/teacher_unified.pt
output/teacher_unified/teacher_crossfit_summary.json
```

Không cần chạy subcommand `features` cho baseline mn10 này. Khi bắt đầu chỉ với hai
checkpoint gốc, đây là bước bắt buộc để sinh `teacher_unified.pt`. Chỉ được bỏ qua
nếu đã nhận một `teacher_unified.pt` đúng checksum ở phần 12.

### 10.4. Kiểm tra, smoke test và train student

```powershell
python main.py check --config config/unified/train_mn10.json
python main.py smoke --config config/unified/train_mn10.json
python main.py train --config config/unified/train_mn10.json
```

`smoke` ghi vào output riêng và dùng 32 mẫu/split, 1 epoch/stage; nó không phải kết quả
báo cáo. `train` tự unpack teacher cache, dựng/kiểm tra FRCRN cache, train stage 1 rồi
stage 2 và hỗ trợ resume theo từng stage.

Output chính:

```text
output/unified/mn10/train/
├── teacher_cache/teacher_train.npz
├── teacher_cache/teacher_validation.npz
├── stage1/best.pt
├── stage1/resume.pt
├── stage1/result.json
├── stage2/best.pt
├── stage2/resume.pt
└── stage2/result.json
```

### 10.5. Evaluate validation và test

```powershell
python -u evaluate.py `
  --dataset <dataset_root> `
  --out output/eval/unified_mn10 `
  --only "Unified mn10"
```

Script chấm cả `best_raw` và `ema_last`, trên validation/test, theo từng SNR và theo
`all/clean/leaked`. Batch eval là 64, bf16 autocast; probabilities là softmax của 36
logit. Các metric gồm accuracy, top-3, macro precision/recall/F1, mAP và router accuracy.

## 11. Kết quả tham chiếu để sanity-check

### 11.1. `best_raw`, test all

| SNR | N | Accuracy | Top-3 | Macro-F1 | mAP | Router acc. |
|---:|---:|---:|---:|---:|---:|---:|
| all | 6.480 | 0.6019 | 0.7713 | 0.5954 | 0.6597 | 0.6867 |
| -5 | 1.080 | 0.6528 | 0.8241 | 0.6454 | 0.7285 | 0.7972 |
| 0 | 1.080 | 0.6306 | 0.8185 | 0.6198 | 0.7118 | 0.6583 |
| 5 | 1.080 | 0.6046 | 0.7759 | 0.5964 | 0.6796 | 0.5602 |
| 10 | 1.080 | 0.6231 | 0.7796 | 0.6136 | 0.6862 | 0.5556 |
| 15 | 1.080 | 0.5833 | 0.7444 | 0.5777 | 0.6276 | 0.6833 |
| 20 | 1.080 | 0.5167 | 0.6852 | 0.5139 | 0.5648 | 0.8657 |

Validation all của `best_raw`: accuracy 0.6144, macro-F1 0.6143, mAP 0.6778.
Validation clean của `best_raw`: N=3.064, accuracy 0.5568, macro-F1 0.5217,
mAP 0.5867. Các số này được tính lại sau train bằng raw checkpoint; không phải score
EMA dùng nội bộ để chọn epoch.

### 11.2. Dung sai kỳ vọng

Nếu viết lại nhưng không dùng đúng GPU/PyTorch/CUDA, không nên kỳ vọng bitwise match.
Một bản tái lập hợp lý phải đồng thời thỏa:

- Teacher 5 fold chọn rất sớm và clean-val accuracy xấp xỉ 0.737–0.742.
- Student stage 2 dừng quanh vùng epoch tham chiếu, không diverge/NaN.
- Test accuracy toàn bộ quanh 0.60 và xu hướng giảm khi SNR tăng.
- Router có pattern đặc trưng: tốt ở hai band biên, khó hơn quanh 5–10 dB.

Sai khác lớn thường đến từ dùng mixture thay `n_hat`, dùng teacher trên `n_hat`, quên
RMS 0.07925, chọn checkpoint trên 6.480 validation thay vì 3.064 clean, hoặc dùng
top-1 routing thay weighted routing.

## 12. Checksum artifact gốc

SHA-256 dùng để xác nhận đang cầm đúng artifact:

```text
b8fdeb00a76289fa48451578a7a40aa437cdb9954683d578197dd0f8a184b14a  model/frcrn_best.pt
e5815275a04b6885e7b8af63d120b29bffae2cd2225cf4915e1ec6d819d3022c  BEATs_iter3_plus_AS2M_finetuned_on_AS2M_cpt2.pt
852c31e4231f3faf71d26d0f0cbcf93b8df409728c236e52ead7d9c540385427  output/teacher_unified/teacher_unified.pt
6883401d221e29887ef6c8680ac933cbabc128de98131cef10cbd49eece94394  config/unified/train_mn10.json
fb85e5a4cf9c097c95bbe439a624b82fb837b22583c86eac0a7c693fe9f670a1  config/unified/kdmoe_mn10.yaml
565bfcb2adb52b6a335eaceed3339f27691019a0b77aa0273842485ec9e8bd4e  config/audioset_train_noise_ytids.txt
989e8e19eded73f43068baf28eee301cbfda40874db57a91b82fbe46cea896ef  dataset/labels.txt
3a6484a51f5ed69f5293f3c256acc9961b7867070cecd2f9977685e3c86650f4  dataset/manifest.csv
f324d42177ca8fdd1adce15c664bc1259607c89f03ff8990ed30db561cbb8d08  dataset/train/manifest.csv
9ed2bc028472196b426d4d49b32eebc10a0edee7498e9e88e18294797e71a25f  dataset/validation/manifest.csv
d33b4c9efab304cd9e88b61adecf7b206b7bd41e2b9d340b59d105ebefb464cd  dataset/test/manifest.csv
```

## 13. Checklist chống sai lệch

- [ ] Đúng 36 lớp và đúng thứ tự `labels.txt`.
- [ ] Audio mono 16 kHz, 4 s; sáu mức SNR đúng như trên.
- [ ] Teacher nghe noise sạch, không nghe mixture hoặc `n_hat`.
- [ ] Teacher train OOF theo `noise_ytid`, validation logits là mean của 5 fold.
- [ ] Tự sinh `teacher_unified.pt` từ BEATs cross-fit (hoặc kiểm tra đúng checksum
  nếu được bàn giao sẵn); không dùng `teacher_unified_nhat.pt`.
- [ ] FRCRN frozen và trả `mixture - speech_hat`.
- [ ] FRCRN cache fp32, TF32 tắt, keyed bằng `sample_id`.
- [ ] Student normalize RMS của `n_hat` về 0.07925.
- [ ] EfficientAT pin commit `a425fdc`, variant `mn10_as`, width 1.0.
- [ ] Resample 16 → 32 kHz; crop giữa 384 mel frames.
- [ ] Không bật SpecAugment mask của student; vẫn giữ fmin/fmax augmentation gốc.
- [ ] Ba expert theo true SNR band; router là MLP học được.
- [ ] Weighted routing ở inference.
- [ ] KD trên final 36 logits, `T=2`, weight 1; không KD router.
- [ ] Stage 1 chỉ train gate + NoiseHead bằng front loss trong 8 epoch.
- [ ] Stage 2 mở toàn bộ model, tối đa 50 epoch, patience 8.
- [ ] Train sampler theo cặp `noise_ytid`, 32 unit/batch.
- [ ] Chọn epoch trên 3.064 validation clean bằng EMA `macro_acc`.
- [ ] Phân biệt `best_raw` và `ema_last` khi báo cáo.

## 14. Mapping từ đặc tả sang code hiện tại

| Thành phần | File nguồn |
|---|---|
| Entrypoint train/check/smoke | `main.py`, `online_pipeline.py` |
| Teacher cross-fit | `teacher_crossfit.py` |
| BEATs frontend/backbone | `teacher/models/beats_backbone.py`, `teacher/beats/` |
| FRCRN cache | `scripts/frcrn_cache.py` |
| Dataset và paired sampler | `scripts/load_dataset.py` |
| EfficientAT adapter | `efficientat.py`, `external_repo.py` |
| KD-MoE forward | `src/model/kdmoe/model.py` |
| Gate/NoiseHead | `src/model/kdmoe/gate.py` |
| Router/experts | `src/model/kdmoe/router.py`, `experts.py` |
| Loss | `src/loss/kdmoe.py` |
| Hai-stage trainer, EMA, selection | `src/train/kdmoe_trainer.py`, `src/train/base.py` |
| Đánh giá | `evaluate.py` |

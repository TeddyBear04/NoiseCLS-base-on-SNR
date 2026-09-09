# Noise Classification 36 Labels + Local SNR

This is an isolated implementation of the final BlackFeather-inspired method.
The original `Noise_Classification_36Labels` directory is unchanged.

## Proposed method

```text
                       mixture x = s + n
                              |
              +---------------+----------------+
              |                                |
              v                                v
       Mixture Encoder                 Demucs Noise Extractor
              |                            (supervised, E-theta)
              |                                |
              |                                v
              |                     estimated noise n_hat
              |                         |             |
              |                         v             v
              |                   Noise Encoder   s_hat=x-n_hat
              |                         |             |
              +-----------+-------------+-------------+
                          |             |
                          v             v
                    Gated Fusion   Local-SNR Head
                          |             |
                          v             v
                36-class Classifier  SNR(t)
```

## Noise extractor

`E-theta` is selected by `model.extractor_type`:

| value | architecture | parameters |
|---|---|---|
| `demucs` (default) | waveform U-Net of Defossez et al. 2020, "Real Time Speech Enhancement in the Waveform Domain" - the 16 kHz mono Demucs - with sinc resampling, GLU encoder/decoder blocks and a BiLSTM bottleneck | 6,027,585 |
| `mask` | complex ratio mask over the STFT | 6,210 |

Demucs is supervised on the noise stem, so it predicts `n_hat` directly.
Predicting speech instead and taking `n_hat = x - s_hat` would carry the whole
error of `s_hat` into the noise, which is ruinous at +15 and +20 dB where the
noise holds a few percent of the mixture energy. Both variants map
`[batch, samples]` to `[batch, samples]`, so a run can ablate the extractor
alone under the same split, seed and loss.

The classifier uses noise-dominant gated fusion, so the mixture branch cannot
simply replace the extracted-noise representation. Local SNR combines an
explicit energy-ratio estimate with a learned bounded correction:

```text
SNR_hat(t) = 10 log10(P_t(s_hat) / P_t(n_hat)) + correction(z_x,t, z_n,t)
```

No DPRNN, environment classifier, or Naive Bayes stage is part of the proposed
method. Those require multi-source stems or acoustic-scene labels that the
current dataset does not provide.

## Objective

```text
L = lambda_cls * CE
  + lambda_sep * (alpha * MR-STFT + beta * relative-L1 - gamma * SI-SDR)
  + lambda_snr * masked-Huber
```

Relative L1 prevents weak noise at high speech-to-noise ratios from receiving
negligible gradients. Local-SNR loss ignores speech-inactive windows.

## Data contract

Each sample returned by the loader contains:

```text
waveform            mixture input
clean_waveform      aligned clean-speech component
noise_waveform      aligned supervised extraction target
target              one-hot 36-class label
local_snr_db        physical SNR targets for 0.5-second windows
local_snr_mask      speech-activity mask for SNR regression
```

Online mixing applies one common peak gain to both components. Therefore the
following invariant holds after augmentation:

```text
waveform == clean_waveform + noise_waveform
```

The local target is computed from actual windowed stem energies, not from the
random gain-control envelope.

## Three training stages

1. `extractor`: train only the complex-mask noise extractor with supervised
   separation loss.
2. `heads`: freeze the extractor and train both encoders, gated fusion,
   classifier, and Local-SNR correction head. Oracle noise is used stochastically
   as teacher forcing.
3. `joint`: fine-tune the complete model end-to-end with a reduced learning
   rate.

Configuration lives in `config/train_config.json`. The default schedule is
30 + 10 + 40 epochs. Each stage carries its own `EarlyStopping`, driven by the
same score that selects its checkpoint: negative separation loss for
`extractor`, macro-F1 for `heads` and `joint`. The classifier has no meaningful
score during `extractor` - it is not trained or even reached in that stage - so
separation loss is the only signal available there.

Weight decay reaches the optimiser as AdamW's decoupled `weight_decay` and is
never added to the loss. It is split three ways: `regularization.l2_lambda` for
classifier and encoder weights, `regularization.extractor_l2_lambda` (zero by
default) for the noise extractor, and zero for every bias and normalisation
parameter. Shrinking extractor weights shrinks the noise it predicts, which is
the opposite of what the high-SNR bands need.

## Run

From this directory:

```powershell
..\.venv\Scripts\python.exe main.py --check-data
..\.venv\Scripts\python.exe main.py --device cuda
```

The dataset defaults to the sibling directory `../36_labels`. It can be
overridden without editing the config:

```powershell
$env:NOISE_DATASET_PATH = "D:\path\to\36_labels"
..\.venv\Scripts\python.exe main.py --device cuda
```

Checkpoints are written under:

```text
checkpoint_local_snr/BlackFeatherLocalSNR/
  stage1_extractor.pt
  stage2_heads.pt
  stage3_joint.pt
  history.jsonl
  summary.json
```

Inference returns both the 36-class posterior and a time-indexed Local-SNR
curve:

```powershell
..\.venv\Scripts\python.exe inference.py example.wav --device cuda
```

## Evaluation

The trainer reports:

- top-1, top-3, balanced accuracy, mAP, and macro/micro F1;
- classification metrics for every SNR band;
- noise SI-SDR;
- Local-SNR MAE and RMSE in dB.

For the research ablation, call the model with `fusion_mode="mixture"`,
`"noise"`, or `"fusion"` and compare all three under the same split and seed.

## Verification

```powershell
..\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

The tests cover component-consistent mixing, physical Local-SNR targets,
forward/backward passes, all fusion ablations, and stage freezing.

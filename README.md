# BEATs mixture/noise fusion for 36-label single-label noise classification

This project adds a noise-separation-and-amplification branch alongside the
original mixture branch, then fuses both BEATs embeddings before classifying:

```text
manifest.csv + labels.txt
        │
        ▼
  Read mixture.wav → mono, 16 kHz, 4 s
        │
        ├────────────────────────────┐
        ▼                            ▼
  ORIGINAL BRANCH               NOISE BRANCH
  (full mixture)                (separated + amplified noise)
        │                            │
  log-Mel fbank (128 bins)      Noise separator (STFT mask) → n_hat
        │                            │
  Conv2D patch embed            Amplify n_hat to target RMS
        │                            │
  BEATs Transformer Encoder     log-Mel fbank (128 bins)
        │                            │
  Mean pooling                  Conv2D patch embed
        │                            │
  z_mix ∈ R^768                 BEATs Transformer Encoder (shared weights)
        │                            │
        │                       Mean pooling
        │                            │
        │                       z_noise ∈ R^768
        │                            │
        └──────────┬─────────────────┘
                    ▼
     Feature fusion: z_mix + Linear(1536, 768)(Concat(z_mix, g·z_noise))  [zero-init]
                     g = sigmoid(a·r + b), r = energy(n_hat)/energy(mixture) in dB
                    │
                    ▼
     Classification head: Linear(768, 36) → 36 logits
                    │
          ┌─────────┴─────────┐
          ▼                   ▼
  CrossEntropyLoss (train)  Argmax (inference) → predicted label

Training only:
  oracle_noise.wav ─┐
  n_hat ────────────┴─→ Separation loss (negative SI-SDR)
  Separation loss + CrossEntropyLoss ─→ Total loss L = L_class + λ·L_sep
```

The BEATs encoder is shared between both branches (mixture and noise pass
through the same transformer weights, in one doubled batch). The noise
separator is a small dilated-convolution mask network operating on the
mixture STFT; its output is loudness-normalized ("amplified") to a fixed RMS
before going through BEATs, since raw noise estimates — especially at high
SNR, where noise is barely audible — would otherwise be too quiet for the
encoder (which was pretrained on normally-loud AudioSet clips) to extract a
useful embedding from.

## Required layout

```text
C_Noise_Separation_Fusion/
├── config/train_config.json
├── ../../36_labels/                  # dataset (outside this repo checkout)
│   ├── manifest.csv
│   ├── labels.txt
│   └── WAV files referenced by mixture_path / noise_path
├── checkpoint/pretrained/BEATs_iter3_plus_AS2M_finetuned_on_AS2M_cpt2.pt
└── checkpoint/                       # files are created automatically
```

Edit the single [train_config.json](config/train_config.json) to set dataset
path, batch sizes, epochs, learning rates, and output paths.

## Colab / Molab commands

Run the four stages in order — each depends on the previous checkpoint:

```bash
pip install -r requirements.txt
python main.py separator36 --config config/train_config.json   # pre-train the noise separator (SI-SDR)
python main.py head36 --config config/train_config.json        # train fusion head on frozen BEATs + separator
python main.py finetune36 --config config/train_config.json    # jointly fine-tune last blocks + separator + head
python main.py test36 --config config/train_config.json
```

`separator36` trains only the separator against `oracle_noise.wav` (available
in training data only) with a scale-invariant SDR loss. `head36` then caches
frozen BEATs embeddings for both branches and trains the fusion head.
`finetune36` unfreezes the last few BEATs blocks and (by default) continues
training the separator jointly, minimizing
`L = L_class + λ·L_sep` where `L_sep` is the negative SI-SDR against
`oracle_noise.wav`. Set `separator_lr: 0` in `train_config.json` to keep the
separator frozen during this stage instead.

## Fusion design

- The fusion projection is zero-initialised with a skip from `z_mix`, so before
  training the model is exactly the mixture-only head of `audio_noise_capstone`;
  its weight decay pulls it back toward that baseline.
- `g` gates the noise branch by the separator's own energy ratio `r`: noise-
  dominated clips (r near 0 dB) trust `n_hat`, high-SNR clips (very negative r,
  where `n_hat` is mostly leaked speech) are down-weighted.
- `max_gain_db` (default 20) caps how much a quiet `n_hat` is amplified. It can be
  changed in `head_training` without retraining the separator; the embedding
  cache is keyed on it.

## Fair comparison with audio_noise_capstone

`finetuning` uses the same batch size (32), accumulation (1), epochs (12) and
patience (4) as the mixture-only run. Compare over several seeds; `seed` only
changes training randomness, so the embedding cache is shared between seeds:

```bash
for SEED in 2026 2027 2028; do
  OUT=checkpoint/seed_$SEED
  python main.py head36 --config config/train_config.json --seed $SEED     --output $OUT/beats_fusion_head_36.pt --results $OUT/beats_fusion_head_36_summary.json
  python main.py finetune36 --config config/train_config.json --seed $SEED     --head-checkpoint $OUT/beats_fusion_head_36.pt     --output $OUT/audio_best_36_fusion.pt --results $OUT/summary_36_fusion.json
  python main.py test36 --config config/train_config.json     --checkpoint $OUT/audio_best_36_fusion.pt     --output $OUT/test_metrics_36_fusion.json --output-dir $OUT
done
```

## Outputs

Each run writes directly to `checkpoint/`:

```text
noise_separator.pt
noise_separator_summary.json
separator_history.csv
separator_snr_metrics.csv
beats_fusion_head_36.pt
beats_fusion_head_36_summary.json
audio_best_36_fusion.pt
summary_36_fusion.json
history.csv
labels.json
summary.json
summary.csv
test_metrics_36_fusion.json
confusion_matrix.csv
confusion_matrix_labeled.csv
per_class_metrics.csv
top_confusions.csv
confusion_matrix.png
```

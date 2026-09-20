# BEATs experts for the SNR-aware 36-label pipeline

This project trains the BEATs branches of the SNR-aware pipeline. It is a copy of
`BEATs/audio_noise_capstone`, which stays in place as the reference project.

```text
16 kHz mixture waveform → 128-bin log-Mel fbank → BEATs encoder
→ mean pooling → Linear(768, 36) → CrossEntropyLoss → argmax
```

The workflow trains a 36-class head on frozen embeddings first, then fine-tunes the
final BEATs transformer blocks. Each clip has exactly one label.

## The three configs

| Config | Branch | SNR levels used | Checkpoints |
|---|---|---|---|
| `config/train_config_low_snr.json` | Expert Low | −5, 0 dB | `checkpoint/low_snr/` |
| `config/train_config_mid_snr.json` | Expert Mid | 5, 10 dB | `checkpoint/mid_snr/` |
| `config/train_config.json` | Fallback baseline | all levels | `checkpoint/` |

Each branch keeps its own checkpoints. They are never shared.

## Required layout

```text
BEATs_Experts/
├── config/train_config{,_low_snr,_mid_snr}.json
└── checkpoint/
    ├── pretrained/BEATs_iter3_plus_AS2M_finetuned_on_AS2M_cpt2.pt
    ├── audio_best_36.pt          baseline, used by the router's fallback branch
    ├── beats_head_36.pt          head that goes with the baseline
    ├── low_snr/                  created by the low-SNR run
    └── mid_snr/                  created by the mid-SNR run
```

The dataset lives outside this folder, at the `dataset_splitter.dataset_path` set in
each config (`/marimo/dataset/mix-dataset` by default).

Checkpoints are excluded by `.gitignore`, so the three files above must be downloaded
or copied onto the training machine before any run.

## Commands

Run from this directory. Replace `low` with `mid` for the 5–10 dB expert.

```bash
pip install -r requirements.txt
python tasks/run_36.py head36     --config config/train_config_low_snr.json
python tasks/run_36.py finetune36 --config config/train_config_low_snr.json
python tasks/run_36.py test36     --config config/train_config_low_snr.json
```

`head36` extracts frozen BEATs embeddings into the cache directory named in the config,
then trains the head on them. `finetune36` unfreezes the last transformer blocks.
`test36` prints and writes the eight test metrics for that branch's SNR levels only.

## Outputs

Each run writes into its branch's checkpoint directory:

```text
beats_head_36.pt          audio_best_36.pt
beats_head_36_summary.json summary_36.json
test_metrics_36.json      history.csv
labels.json               confusion_matrix.csv
confusion_matrix.png      per_class_metrics.csv
top_confusions.csv
```

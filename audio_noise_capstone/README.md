# BEATs fine-tuning for 36-label single-label noise classification

This project uses the BEATs fine-tuning pipeline:

```text
16 kHz mixture waveform → 128-bin log-Mel fbank → BEATs encoder
→ mean pooling → Linear(768, 36) → CrossEntropyLoss → argmax
```

The workflow trains a 36-class head first, then fine-tunes the final four
BEATs transformer blocks. Each clip has exactly one label.

## Required layout

```text
audio_noise_capstone/
├── config/train_config.json
├── dataset/mix-dataset/
│   ├── manifest.csv
│   ├── labels.txt
│   └── WAV files referenced by mixture_path
├── checkpoint/pretrained/BEATs_iter3_plus_AS2M_finetuned_on_AS2M_cpt2.pt
└── checkpoint/                       # files are created automatically
```

Edit the single [train_config.json](config/train_config.json) to set dataset
path, batch sizes, epochs, learning rates, and output paths.

## Colab / Molab commands

```bash
pip install -r requirements.txt
python main.py head36 --config config/train_config.json
python main.py finetune36 --config config/train_config.json
python main.py test36 --config config/train_config.json
```

## Outputs

Each run writes directly to `checkpoint/`:

```text
head.pt
audio_best.pt
history.csv
labels.json
summary.json
summary.csv
test_metrics.json
confusion_matrix.csv
confusion_matrix.png
```

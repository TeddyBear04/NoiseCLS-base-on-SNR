# DPCRN noise-target classification (36 labels)

This is a Colab-ready PyTorch adaptation of DPCRN for the high-SNR noise-classification expert. It follows the layout and manifest contract of `BEATs/audio_noise_capstone`, but replaces BEATs with a DPCRN-style complex-mask network.

The model is trained from scratch. Its direct separation target is `noise_path`, not clean speech:

```text
mixture → STFT → CNN encoder → dual-path RNN → complex noise mask
        → estimated noise → noise embedding → 36-class classifier
```

## Layout

```text
dpcrn_noise_target_36/
├── config/train_config.json
├── dataset/mix-dataset/          # same layout as audio_noise_capstone
│   ├── manifest.csv
│   ├── labels.txt
│   └── WAV files referenced by mixture_path and noise_path
├── checkpoint/
├── models/
├── tasks/
└── utils/
```

## Colab commands

```bash
pip install -r requirements.txt
python main.py train36 --config config/train_config.json
python main.py test36 --config config/train_config.json
python main.py predict path/to/mixture.wav --checkpoint checkpoint/dpcrn_noise_target_best.pt
```

`snr_min_db` and `snr_max_db` select the target SNR band. The default is 15–20 dB. The data must include supervised `noise_path` stems; without those stems the noise-mask loss cannot be trained.

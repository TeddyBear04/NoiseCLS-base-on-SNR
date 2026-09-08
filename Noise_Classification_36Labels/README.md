# Noise classification at 16 kHz

This directory is an independent adaptation of `U_FFIA27K_audio`. It is
configured for the 36-label `36_labels` dataset and still reads the older
21-label layout when pointed at it. The original fish-feeding source directory
is unchanged.

## What this pipeline does

- Reads the authoritative per-split manifests; it never re-splits the noise
  sources.
- Rebuilds WAV paths from `sample_id`, so stale absolute paths stored in the
  manifests are ignored.
- Maps the original AudioSet label indices to contiguous model outputs using the
  dataset's label catalog.
- Trains a multi-label model with `BCEWithLogitsLoss` and optional class weights.
- Uses 16 kHz audio, random 4-second train crops, and deterministic 4-second
  sliding windows with a 2-second hop for validation, test, and inference.
- Optionally creates controlled time-varying SNR mixtures from the `clean` and
  noise stems during training.
- Reports top-1/top-3/balanced accuracy, mAP, macro/micro F1, per-label metrics,
  and metrics for each SNR band.

## Expected dataset layout

```text
36_labels/
├── labels.txt              # one display name per line; line order = label index
├── train/
│   ├── manifest.csv        # sample_id, multi_hot_36, target_snr_db, duration_seconds, ...
│   ├── mixture/
│   ├── clean/
│   └── noise/
├── validation/
└── test/
```

`labels.txt` has 36 lines and the manifest column is `multi_hot_36`, so
`model.classes_num` is 36. Each clip carries exactly one label, and the splits
are perfectly balanced: 840 train, 180 validation, and 180 test clips per label,
spread evenly over the six target SNRs.

### Supported manifest schemas

The loader detects the label encoding from the manifest header:

| Catalog file          | Manifest column   | Meaning                                    |
| --------------------- | ----------------- | ------------------------------------------ |
| `labels.txt`          | `multi_hot_<N>`   | JSON list of 0/1 flags, one per label      |
| `selected_labels.csv` | `label_indices`   | JSON list of original AudioSet indices     |

To read the older `21_labels_dataset`, override the directory and catalog names
in `dataset_splitter`: `train_directory: "train_single"`, `validation_directory:
"validation_single"`, `test_directory: "test_single"`, `noise_directory:
"oracle_noise"`, `selected_labels_file: "selected_labels.csv"`, and
`model.classes_num: 21`.

## Run on Marimo

Place the dataset at `/marimo/36_labels`. If it is elsewhere, edit
`dataset_splitter.dataset_path` in `config/train_config.json` or set the
`NOISE_DATASET_PATH` environment variable.

Install dependencies in a Marimo terminal:

```bash
cd /marimo/Capstone_2026_Fish_Feeding_Intensity/Noise_Classification_21Labels
python -m pip install -r requirements.txt
```

Start training from a terminal:

```bash
python main.py --config config/train_config.json --check-data
python main.py --config config/train_config.json --device cuda
```

Or run it from a Marimo cell:

```python
from pathlib import Path
import sys

project = Path("/marimo/Capstone_2026_Fish_Feeding_Intensity/Noise_Classification_21Labels")
sys.path.insert(0, str(project))

from main import run_training

result = run_training(
    str(project / "config" / "train_config.json"),
    device_name="cuda",
)
result["checkpoint_path"]
```

## Where results are written

`ckpt_dir` is `checkpoint_36_labels`, so runs on this dataset never touch
`checkpoint/`, which keeps the earlier 21-label results (`Cnn14MobileV2`,
`EfficientNetB0`, `PANNS_Cnn14`, `ResNet22`). Those checkpoints have a 21-unit
head and cannot be loaded into a 36-label model; both directions are guarded:

- Training refuses to start if `ckpt_dir/<backbone>/labels.json` records a
  different label set, so an old report is never silently overwritten.
- `inference.py` reports the label-count mismatch by name instead of failing
  with a raw `state_dict` size error.

The best model and reports are written to:

```text
checkpoint_36_labels/<backbone>/
├── audio_best.pt
├── history.csv
├── learning_curves.png
├── summary.csv
├── summary.json
├── test_per_label.csv
├── test_snr_metrics.csv
├── validation_snr_metrics.csv
├── snr_metrics.png
└── classification_report_test.txt
```

## Sliding-window inference

```bash
python inference.py /marimo/example.wav \
  --config config/train_config.json \
  --checkpoint checkpoint_36_labels/Cnn14MobileV2/audio_best.pt \
  --device cuda
```

This creates a CSV containing the probability of every label in every time
window, plus a JSON clip-level summary.

## Accuracy metrics

`36_labels` gives every clip exactly one label, so the headline number is
ordinary single-label accuracy. The model is still trained with a multi-label
head, which means several accuracy definitions coexist. All of them are reported
in the per-epoch log, `history.csv`, `summary.csv`/`summary.json`,
`classification_report_test.txt`, and per SNR band:

- `top1_accuracy` — the clip counts as correct when the highest-scoring of the
  36 outputs is its true label. This is the accuracy to quote for a single-label
  task, and it does not depend on `threshold`. Logged as `top1`.
- `top3_accuracy` — the true label is among the three highest-scoring outputs.
  Also threshold-free; useful for showing how close a wrong prediction was.
- `balanced_accuracy` — `top1_accuracy` computed per label and then averaged, so
  every label weighs the same. On the balanced `36_labels` splits it tracks
  `top1_accuracy` closely; a gap between them means the errors are concentrated
  in a few labels.
- `hamming_accuracy` — fraction of the 36 label decisions that are correct,
  averaged over every clip. Because only one label is positive, predicting
  all-zero already scores about 0.97, so this number is nearly uninformative
  here and is kept only for comparison with the multi-label runs.
- `subset_accuracy` — fraction of clips where the thresholded label set matches
  exactly. It punishes both an empty prediction and two labels above the
  threshold, so it reads far lower than `top1_accuracy`. Logged as
  `exact-match`.

`test_per_label.csv` and `validation_per_label_best.csv` carry two per-label
columns: `top1_recall` (accuracy restricted to the clips of that label, whose
mean is `balanced_accuracy`) and `accuracy` (element-wise correctness, whose
mean is `hamming_accuracy`).

`hamming_accuracy` and `subset_accuracy` move when you tune `threshold`.
`top1_accuracy`, `top3_accuracy`, `balanced_accuracy`, `mAP`, and `auc` do not.

## Results by SNR band

Every evaluation is also broken down by the clip's `target_snr_db`, which is the
main way to see how the model degrades as noise increases. Each band reports
`samples`, `top1_accuracy`, `top3_accuracy`, `balanced_accuracy`, `mAP`,
`macro_auc`, `macro_f1`, `micro_f1`, `precision_macro`, `recall_macro`,
`hamming_accuracy`, and `subset_accuracy`.

Output lands in three places:

- `test_snr_metrics.csv` and `validation_snr_metrics.csv` — one row per band.
- `snr_metrics.png` — grouped bar chart of top-1 accuracy / mAP / macro F1 /
  micro F1.
- `summary.json` — same numbers under `test_snr_metrics` and
  `validation_snr_metrics`.

The end of training also prints the table:

The bands are configurable via `snr_bands` in `config/train_config.json`.
Bounds are inclusive on both ends. `36_labels` uses six discrete target SNRs
(-5, 0, 5, 10, 15, 20 dB) spread evenly over the splits, so the defaults are six
single-value bands that cover every clip exactly once (verified on all three
splits: 30240 train, 6480 validation, and 6480 test clips, with none left over).
If you change the bands and leave clips outside all of them, a warning naming
the uncovered count is logged rather than silently dropping them.

## Important configuration knobs

- `batch_size`: start with 32; reduce to 16 or 8 if CUDA runs out of memory.
- `num_workers`: `-1` auto-selects up to eight workers. Use `0` if the Marimo
  runtime has multiprocessing issues.
- `cache_audio`: keep `false` for a dataset of this size.
- `dynamic_snr_enabled`: enables clean/noise on-the-fly mixing.
- `dynamic_snr_probability`: fraction of train crops receiving dynamic SNR.
- `threshold`: initial multi-label decision threshold. Tune it on validation
  predictions after the first training run.
- `use_pos_weight`: leave `false` on `36_labels`. That dataset is single-label
  and perfectly balanced (840 train clips per label), so neg/pos is 35 for every
  label; a positive weight would not correct any imbalance and would only push
  the model to over-predict.
- `monitor`: metric that drives best-checkpoint selection and early stopping.
  One of `macro_f1`, `mAP`, `accuracy` (an alias of `top1_accuracy`),
  `top1_accuracy`, `balanced_accuracy`, `hamming_accuracy`, `subset_accuracy`,
  `loss`. `macro_f1` is the default; `top1_accuracy` is a reasonable choice
  when accuracy is the number you report. Avoid `subset_accuracy`: it stays
  flat at 0.0 early in training, so early stopping would fire before the model
  learns anything.
- `profile_model`: disabled by default because FLOP profiling adds startup time.

The manifest stores weak clip-level labels, not event timestamps. Per-window
outputs therefore indicate model confidence over time but are not supervised
on exact noise onset/offset boundaries.

# Noise classification baseline

The loader uses local paths derived from `sample_id`, so stale absolute paths in
the CSV manifests do not need to be edited.

## Environment

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

## Quick test

```powershell
python train.py --smoke-test --batch-size 2
```

The training output includes micro precision/recall/F1, macro F1 and strict
exact-match accuracy at a default probability threshold of 0.5.

## Inference

```powershell
python infer.py 21_labels_dataset/test_single/mixture/test_single_00000000.wav
```

Use `--threshold 0.3` or `--top-k 5` to adjust the displayed multi-label output.

## Training

Medium-sized learning check:

```powershell
python train.py --limit 1000 --epochs 3 --batch-size 8
```

Full training:

```powershell
python train.py --epochs 30 --batch-size 8 --workers 4
```

The default loss is class-balanced `BCEWithLogitsLoss + 0.3 * MSE`. BCE is used
because each record can contain more than one of the 21 selected AudioSet labels.
The positive-class weights are calculated from the selected training subset and
clamped to 20 so rare classes cannot dominate optimization.

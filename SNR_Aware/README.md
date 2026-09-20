# SNR-aware 36-label noise classification

Classify 36 noise types in speech+noise mixtures. The difficulty changes with SNR, so
instead of one model for every noise level the system routes each mixture to an expert
trained for its SNR range.

```text
mixture = speech + noise
    → SNR gate           estimates which SNR range the mixture is in
    → router             picks the expert for that range
    → expert             predicts one of 36 noise labels
```

## Branches

| Branch | SNR range | Model | Why |
|---|---|---|---|
| Expert Low | −5 to 0 dB | BEATs Low | Noise dominates, speech is buried |
| Expert Mid | 5 to 10 dB | BEATs Mid | Mixture is balanced |
| Expert High | 15 to 20 dB | DPCRN noise-target | Speech dominates, so the noise must be separated before it can be classified |
| Fallback | 0–5, 10–15 dB, or the gate is unsure | BEATs baseline | Covers the ranges that have no expert of their own |

## The high-SNR branch is the unusual one

DPCRN here does **not** enhance speech. It learns the inverse problem: pull the noise
out of the mixture, then classify what it pulled out.

```text
mixture → complex STFT → CNN encoder → Dual-Path RNN → complex noise mask
        → estimated noise → log-magnitude embedding → 36-class classifier

loss = CrossEntropy(noise class) + 0.5 × SeparationLoss(estimated noise, noise target)
```

`noise_path` in the manifest is the separation ground truth.

## Layout

```text
SNR_Aware/
├── pipeline_config_4_branches.json   routing contract read by the router
├── BEATs_Experts/                    Expert Low, Expert Mid, and the fallback baseline
├── DPCRN_Noise_Target/               Expert High
├── gate/                             SNR gate
└── router/                           routing and end-to-end evaluation
```

`BEATs_Experts/` is a copy of `BEATs/audio_noise_capstone`, which stays in place as the
reference project.

## Dataset

```text
/marimo/dataset/mix-dataset/
├── manifest.csv
├── labels.txt
├── mixture audio files
└── noise target audio files
```

`manifest.csv` needs at least `split`, `mixture_path`, `noise_path`, `label_names` and
`target_snr_db`.

`target_snr_db` holds exactly six discrete levels — −5, 0, 5, 10, 15 and 20 dB — evenly
balanced across all three splits (5040 clips per level in train, 1080 in validation and
in test; 43,200 clips in total).

That balance is what shapes the gate. Training only ever sees the three expert ranges, so
the gate classifies into three classes — one per expert — rather than into five bins, two
of which would have no data to learn from.

The gate carries a second output alongside those three classes: a regression head that
estimates the SNR in dB. It is what makes the fallback branch's range condition real. A
three-class head can only say "low, mid or high"; it cannot say whether a mixture sits at
3 dB. The regression head can, so the router checks both conditions:

```text
fallback if   confidence < 0.70
         or   estimated SNR falls in (0, 5) or (10, 15)
```

On this dataset the range condition rarely fires, since every clip sits on one of the six
levels. It is there for real audio, where SNR is continuous. The evaluation report counts
the two fallback paths separately so you can see which one is doing the work.

## Running it

### 1. Train the BEATs experts

```bash
cd BEATs_Experts
python tasks/run_36.py head36     --config config/train_config_low_snr.json
python tasks/run_36.py finetune36 --config config/train_config_low_snr.json
python tasks/run_36.py test36     --config config/train_config_low_snr.json
```

Replace `low` with `mid` for the 5–10 dB expert. Each branch writes to its own
checkpoint directory and is never shared with the other.

### 2. Train the DPCRN high-SNR expert

```bash
cd DPCRN_Noise_Target
pip install -r requirements.txt
python -u main.py train36 --config config/train_config.json
python -u main.py test36  --config config/train_config.json
```

Best checkpoint: `checkpoint/dpcrn_noise_target_best.pt`. Use `python -u` so the
per-epoch lines appear in a redirected log immediately instead of sitting in the buffer.

### 3. Train the gate, then evaluate end to end

Not built yet. See
[the design spec](../docs/superpowers/specs/2026-09-20-snr-aware-gate-router-design.md).

## Evaluation

Two reports, and the gap between them is the point:

- **Oracle-SNR** routes each clip by its true `target_snr_db`. This measures each expert
  on its own, with no gate errors mixed in.
- **End-to-end** routes each clip by the gate's prediction. This is what deployment looks
  like.

A third, **baseline-only**, runs without routing at all. If the end-to-end numbers do not
beat it, the four-branch architecture is not earning its complexity.

Each report gives accuracy, macro precision, macro recall, macro-F1, micro-F1, mAP,
balanced accuracy and macro-AUC, overall and per SNR level.

# BEATs 21-label: Attention Pooling Ablation

## Objective

Test whether a learned temporal attention pooling head improves over the
baseline's mean pooling of BEATs frame features. The test split was not used.

## Configuration

- Same 21-label BEATs protocol as the weighted-BCE baseline: 6 s random train
  crop, 6 s centre validation crop, 18,231/3,907 samples, last four encoder
  blocks unfrozen after one head-only epoch.
- AdamW, batch size 32, head LR `5e-5`, encoder LR `2e-6`, five epochs.
- Each frame receives a learned scalar attention score; its softmax-weighted
  feature vector is classified by the pretrained-initialized linear head.
- Attention scores are zero-initialized, so the initial model is exactly
  uniform/mean pooling. This makes the comparison controlled.

## Result

| Validation metric | Mean pooling | Attention pooling | Delta |
| --- | ---: | ---: | ---: |
| Macro mAP | 0.682956 | 0.682806 | -0.000151 |
| Micro AP | 0.677941 | 0.678459 | +0.000518 |
| Micro F1 @ 0.5 | 0.504542 | 0.505604 | +0.001062 |
| Macro F1 @ 0.5 | 0.496816 | 0.497856 | +0.001040 |
| Exact match @ 0.5 | 0.138725 | 0.140773 | +0.002048 |

Attention was slightly ahead through epochs 1–4, but its final epoch mAP was
0.000151 lower. Its best class AP gains were `Vehicle` (+0.002612), `Speech`
(+0.002083), and `Musical instrument` (+0.001770); the largest losses were
`Percussion` (-0.005988), `Domestic animals, pets` (-0.003345), and `Aircraft`
(-0.002619). mAP decreased in every SNR band, with the largest loss in
`[10,15)` (-0.002325).

## Decision

Reject attention pooling as the default model: the change adds parameters and
inference work but does not improve the primary macro-mAP metric. Do not test
this checkpoint or run a multi-crop evaluation. Retain mean pooling and focus
future work on the data/noise setup rather than this head change.

## Implementation note

`finetune_beats_21.py` now exposes `--pooling mean|attention`, and
`evaluate_beats_21_multicrop.py` loads either checkpoint format. The default
remains `mean`, preserving the established workflow.

# BEATs 21-label: Asymmetric Loss Ablation

## Objective

Test whether Asymmetric Loss (ASL) improves the 21-label multi-label BEATs
fine-tuning baseline. This is a validation-only experiment; the test split was
not accessed.

## Configuration

- Encoder: BEATs pretrained; unfreeze last 4 transformer blocks after one
  head-only epoch.
- Input and split: 6 s random training crop; 6 s centre validation crop;
  18,231 training and 3,907 validation samples.
- Optimizer: AdamW, head LR `5e-5`, encoder LR `2e-6`, batch size 32.
- ASL: `gamma_negative=4`, `gamma_positive=1`, negative-probability clip
  `0.05`. ASL replaces the baseline's class-weighted BCE and therefore does
  not use `pos_weight`.
- Selection metric: validation macro average precision (mAP), fixed threshold
  0.5 for F1 diagnostics.

## Result

| Validation metric | Weighted BCE baseline | ASL | Delta |
| --- | ---: | ---: | ---: |
| Macro mAP | 0.682956 | 0.666767 | -0.016189 |
| Micro AP | 0.677941 | 0.694806 | +0.016864 |
| Micro F1 @ 0.5 | 0.504542 | 0.480277 | -0.024265 |
| Macro F1 @ 0.5 | 0.496816 | 0.493593 | -0.003223 |
| Exact match @ 0.5 | 0.138725 | 0.089071 | -0.049654 |
| Predicted labels/sample | 3.4848 | 3.7822 | +0.2974 |

ASL's best checkpoint was epoch 5. It increased global micro AP but degraded
macro mAP, especially for less frequent or ambiguous labels. The only class
with an AP improvement was `Car` (+0.003745). The largest drops were `Domestic
animals, pets` (-0.043558), `Aircraft` (-0.040643), and `Musical instrument`
(-0.035089). By SNR, ASL improved only `[0,5)` (+0.013329 mAP); it dropped
0.030543 mAP in `[15,20]`.

## Decision

Reject default ASL for the current objective and do not evaluate this checkpoint
on the test split. Its negative focusing is too aggressive relative to the
dataset's existing class-weighted BCE, producing too many positive labels at a
fixed 0.5 threshold. Keep `checkpoints/beats_21_last4.pt` as the current
single-crop baseline. The next experiment should retain BCE and improve the
temporal pooling head rather than replace the loss.

## Reproduction

```powershell
.\.venv\Scripts\python.exe finetune_beats_21_asl.py --epochs 5 --head-only-epochs 1 --trainable-blocks 4 --batch-size 32 --validation-batch-size 64 --workers 4 --head-lr 5e-5 --encoder-lr 2e-6 --patience 2 --output checkpoints\beats_21_asl_last4.pt --results benchmark_results\beats_21_asl_last4.json
```

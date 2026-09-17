# Selective refinement predictability

Targeted heavy trace: 16 fixed GSM8K requests, 27,381 adjacent still-masked token pairs. The existing 128-request aggregate trace remains the systems reference; this trace only supplies missing token confidence/hidden fields.

| Metric | Value |
|---|---:|
| confidence Pearson | 0.6172 |
| confidence Spearman | 0.5083 |
| within-block confidence-rank Spearman | 0.3091 |
| top-25/50/75 overlap | 0.563 / 0.653 / 0.822 |
| expert Jaccard mean | 0.574 |
| EP4/EP8 destination Jaccard mean | 0.854 / 0.749 |
| held-out confidence AUROC | 0.794 |

Only previous-step observations are used. Critical-rank persistence was not substituted for token-level predictability.

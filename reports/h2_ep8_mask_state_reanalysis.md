# H2 Mask-State Specialization: EP8 Re-analysis

## Verdict

**HOLD / NO METHOD.** EP8 makes expert-ID volatility more visible at the physical-rank level, but the full-workload systems consequence remains far below the 5% promotion gate.

## Same-position transition

| Metric | EP4 | EP8 |
|---|---:|---:|
| Destination-rank-set Jaccard P50 | 0.750 | **0.600** |
| Rank-set transition rate | 66.71% | **87.10%** |
| Samples | 263,606 | 263,606 |

The expert-set Jaccard remains the prior 0.231 P50. Splitting 256 experts into 32-expert EP8 buckets lowers destination Jaccard from 0.75 to 0.60 and raises transition incidence from 66.7% to 87.1%.

## EP8 state-only geometry

| State | Max/mean P50/P95 | CV P50/P95 | Remote bytes P50/P95 | Fanout P50 |
|---|---:|---:|---:|---:|
| CURRENT_BLOCK_MASKED | 2.556 / 3.875 | 0.853 / 1.314 | 204.0 / 428.0 KiB | 4.000 |
| CURRENT_BLOCK_NEWLY_ACCEPTED | 3.000 / 5.000 | 1.118 / 1.581 | 16.0 / 92.0 KiB | 4.000 |
| CURRENT_BLOCK_DECODED | 2.188 / 3.375 | 0.670 / 1.118 | 204.0 / 396.0 KiB | 4.000 |

State-only imbalance is large, but these 1–32 current-block rows are embedded in 1,472–3,616 physical rows. State-only predicted times also fall outside parts of the measured EP8 compute envelope and are diagnostic only.

## Full physical workload composition effect

| Target | Max/mean P50/P95/P99 | Critical expert P95/P99 | Remote bytes P95/P99 | Routed stage P50/P95/P99/max |
|---|---:|---:|---:|---:|
| EP4 | 0.18% / 1.07% / 1.86% | 0.00% / 5.06% | 0.36% / 0.64% | 0.01% / 0.08% / 3.81% / 10.45% |
| EP8 | 0.22% / 1.33% / 2.31% | 0.00% / 2.73% | 0.27% / 0.46% | 0.02% / 0.21% / 2.03% / 11.31% |

The counterfactual replaces current MASKED route sets with a deterministic cycle of same-invocation DECODED route sets, while retaining the full prefix/prior-block workload. EP8 raises the P95 stage effect from 0.082% to **0.211%**, still two orders of magnitude below the 5% gate. P99 is 2.03%, and only isolated maxima reach 11.31%.

No H2 method should be created. EP8 amplifies rank identity transitions, not a reproducible full-stage bottleneck.

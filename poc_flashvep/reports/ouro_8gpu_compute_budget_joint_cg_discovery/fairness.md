# Completed-output-token fairness

This ablation changes fairness only. It does not run a C/G optimizer.
All five native paths cross stock and completion-aware fairness on each device.
Atomic, staged G4, G2, and Split use C1024. Native Mixed uses C512.
The queue contains eight model-process blocks and 160 measured observer cohorts.
Each prompt-shape contrast has four independent GPU/process blocks and two input cohorts per block.
The two source cohorts repeat across GPUs. This limits workload generalization.

All prompt work passed the Full-R4 accounting assertion.
All 480 paired requests had the same generated token count and executed decode-loop work.
Output token IDs matched in 426/480 pairs.
The output differences do not establish an accuracy regression or prove equivalence.
The new policy still requires full-answer quality evaluation.

| Path | Prompt/N | Stock mean cohort pooled ITL P99, ms | Completion-fair mean cohort pooled ITL P99, ms | Paired change, ms [95% CI] |
| --- | --- | ---: | ---: | ---: |
| atomic | 2033/4 | 180.917 | 132.735 | -48.182 [-52.127, -44.054] |
| staged4 | 2033/4 | 183.629 | 148.318 | -35.311 [-50.803, -10.892] |
| grouped | 2033/4 | 143.821 | 80.354 | -63.467 [-75.976, -50.957] |
| split | 2033/4 | 124.110 | 82.056 | -42.054 [-67.382, 5.517] |
| mixed | 2033/4 | 128.383 | 80.443 | -47.939 [-100.827, -20.909] |
| atomic | 8177/8 | 694.272 | 406.420 | -287.852 [-509.293, -47.275] |
| staged4 | 8177/8 | 451.971 | 464.774 | 12.803 [-121.042, 146.647] |
| grouped | 8177/8 | 473.514 | 439.753 | -33.761 [-86.141, 41.332] |
| split | 8177/8 | 329.582 | 347.653 | 18.071 [-221.847, 346.456] |
| mixed | 8177/8 | 157.213 | 175.261 | 18.047 [-139.385, 197.405] |

The interval resamples process/GPU block aggregates, not individual tokens.
The table reports means of within-cohort pooled quantiles. It is not a global pooled-token quantile.
All these timings use the observer/shared-host lane, not a quiet clean-serving headline.
A fairness effect must not be attributed to Joint C/G selection.
P95, TTFT, and output-rate effects remain in `fairness_paired_effects.csv`.
A lower P99 does not imply a lower P95, faster TTFT, or sustainable capacity.

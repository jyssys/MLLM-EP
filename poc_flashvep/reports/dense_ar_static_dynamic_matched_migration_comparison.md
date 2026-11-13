# Static and Dynamic: matched migration and ablation

## Result

`TRACE_ANALYSIS`: Revised improved TTFT against a shuffled, matched-work control.
It did not outperform the validation-selected Static Last8 baseline at Input8000/Output16/QPS4.8.
It also did not establish Native-relative Decode 5% non-inferiority in that cell.
Thus, matched-work evidence alone does not establish the requested final GO.

All rows below use three independent N128 starts.
Arrival seeds are2026111381/82/83.
Each comparison uses the same fresh128 prompt IDs and actual arrival trace.
The model revision is `b968826d9c46dd6066d109eabc6255188de91218`.
GPU0/1,42 GiB KV per GPU, BF16, and exact Full36 execution remain fixed.

## Start-mean results

| Arm | TTFT p95 (ms) | ITL p99 (ms) | TPOT p95 (ms) | Balanced goodput (requests/s) |
|---|---:|---:|---:|---:|
| Native | 10937.125 | 26.981 | 25.418 | 1.145 |
| Controller Zero | 11243.116 | 27.611 | 26.272 | 1.107 |
| Static Last12,31.25% | 8043.383 | 26.786 | 26.018 | 1.575 |
| Static Last12,50% | 6045.453 | 27.571 | 26.322 | 2.576 |
| Best Static Last8,100% | 4550.111 | 28.979 | 27.389 | 3.973 |
| Matched Profile | 6046.387 | 27.997 | 26.711 | 2.505 |
| Previous IDLE | 6368.529 | 26.886 | 25.891 | 2.411 |
| Previous DEADLINE | 6269.233 | 28.633 | 27.418 | 2.377 |
| Previous COST | 6086.327 | 27.626 | 26.609 | 2.572 |
| Revised | 5316.998 | 28.263 | 27.027 | 3.296 |
| Continuous Colocated DP2 | 3026.389 | 35.836 | 29.323 | 3.505 |

These rates include finite-cohort drain.
They are not sustainable serving capacities.
Each latency value is the mean of three start-level quantiles.
It is not a pooled token quantile.

## Exact aggregate matching

The matched control derives cut assignments from each Revised start's realized groups.
A fixed independent shuffle assigns those cuts to different request indices.
No per-request latency, prompt value, or favorable outcome selects the assignment.

| Seed | Migrated requests in each arm | Migrated layers in each arm | Realized match |
|---|---:|---:|---|
| 2026111381 | 108 | 896 | PASS |
| 2026111382 | 104 | 880 | PASS |
| 2026111383 | 96 | 800 | PASS |

This control uses post-hoc aggregate information.
It is not a deployable policy or a newly selected held-out winner.
It controls total migrated request count and layer work.
It does not preserve the actual group profile after arrival and runtime collection.

| Seed | Revised migrated groups | Matched migrated groups | Exact realized group profile |
|---|---:|---:|---|
|2026111381 |27 |30 | NO |
|2026111382 |26 |28 | NO |
|2026111383 |24 |30 | NO |

The shuffled control executes additional small source batches.
Therefore, the comparison includes batching effects as well as timing effects.
It does not isolate a pure state-predictor contribution.
An exact realized-group control remains `NOT_RUN`.
The frozen control configuration's intended profile description must not override this observed audit.

Revised/Matched TTFT p95 was 0.87937, with 95% CI [0.85359,0.90329].
The reduction was 12.06%.
ITL p99 was 1.00952[0.99201,1.02822].
TPOT p95 was 1.01184[0.99886,1.02851].
Balanced goodput was 1.31563[1.21648,1.48526].

These intervals use paired independent starts.
All27 empirical bootstrap draws preserve seed pairing.
No request or token resampling inflates the independent sample count.

## Strong Static and Colocated comparison

Revised/Best Static TTFT p95 was 1.16854[1.10985,1.25473].
Revised was 16.85% slower on that metric.
Balanced goodput was 0.82959[0.72449,0.91070].
Therefore, a matched-profile improvement does not demonstrate superiority over Strong Static.

Revised/Colocated TTFT p95 was 1.75688[1.27189,2.75758].
Revised had lower ITL p99, at 0.78868[0.70380,0.86034].
Balanced goodput was 0.94019[0.68075,1.28504].
This cell does not establish a Balanced goodput advantage over Colocated.
Other frozen SLO regimes remain separate analyses.

## Interpretation

Controller Zero exposes the cost of coordinator grouping without migration.
All final arms share the generic deferred completion-log improvement.
That improvement is not a Dynamic Layer contribution.

Previous policies retain their5 ms margins for ablation only.
Revised uses the separately frozen empirical uncertainty model.
No held-out result changes the selected settings or frozen SLOs.

The evidence supports a difference beyond total migrated work alone.
Batch fragmentation prevents attribution to state-aware selection alone.
It does not support the complete requested Dynamic superiority claim.
The [paired evidence](dense_ar_decode_safe_elastic_layer_pd_20261113/analysis/matched_migration_and_ablation_results.json) preserves all ratios and controls.
The [geometry erratum](dense_ar_decode_safe_elastic_layer_pd_20261113/analysis/matched_control_geometry_erratum.json) records the actual batch profiles.
The [main report](dense_ar_decode_safe_elastic_layer_pd_final_summary.md) states the tested-envelope verdict.

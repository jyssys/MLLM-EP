# Native fixed-R4 cohorts: TP and DP4 controls

All eight jobs pass both runner and GPU-guard success checks.
The CPU analysis contains 320 clean trials across 64 cells.
Each cell contains five controlled repetitions.
No incomplete job contributes timing data.
The coordinator ran the GPU jobs.
This worker ran only the analyzer and CPU tests.

TP4 has the lowest median among TP1/2/4 in all 16 matched cells.
Its API-wall median is lower than DP4's parent-wall median in 14 cells.
The remaining two differences are below 0.32%, with overlapping observed trial ranges.
These are generic calibration controls, not a candidate gain or a general placement optimum.

## Matched work and timing boundaries

Every request uses fixed R4, BF16, 128 controlled output tokens, and the same frozen non-EOS token ID, 378.
The analyzer checks protocol hashes, checkpoint revisions, prompt-token signatures, engine settings, and output denominators.
It excludes warm-up and separate trace timing.

TP time covers one native `generate` API call, including cohort submission and completion.
DP4 time covers the parent's first release send through all four matching completion receipts.
DP4 includes release skew, completion transport, controller work, and worker post-call assertions and serialization.
It excludes prompt preparation and readiness waits before release.
The two timing boundaries differ.
No subtraction converts them into a common unobserved boundary.

TP1/TP2 reserve one/two GPUs.
TP4 and DP4 reserve four GPUs.
DP4 retains all four loaded replicas, including idle replicas at C1/C2.
C is the submitted cohort size, not a constant native batch size.

## Median closed-cohort wall time

All entries are milliseconds.
TP columns use API wall.
The DP4 column uses observed parent wall.

| Base | N | C | TP1 | TP2 | TP4 | DP4 |
|---|---:|---:|---:|---:|---:|---:|
| 1.4B | 1024 | 1 | 1089.76 | 929.61 | 813.43 | 1046.91 |
| 1.4B | 1024 | 2 | 1166.76 | 1001.47 | 899.90 | 1046.60 |
| 1.4B | 1024 | 4 | 1265.09 | 1054.85 | 945.34 | 1091.82 |
| 1.4B | 1024 | 8 | 1493.11 | 1200.88 | 1014.67 | 1206.25 |
| 1.4B | 4096 | 1 | 1218.46 | 1011.08 | 863.74 | 1214.87 |
| 1.4B | 4096 | 2 | 1449.46 | 1166.61 | 993.63 | 1213.40 |
| 1.4B | 4096 | 4 | 1907.49 | 1426.70 | 1144.34 | 1216.29 |
| 1.4B | 4096 | 8 | 2781.51 | 1924.30 | 1450.21 | 1446.82 |
| 2.6B | 1024 | 1 | 2062.76 | 1826.40 | 1628.60 | 2097.56 |
| 2.6B | 1024 | 2 | 2220.94 | 1964.82 | 1797.89 | 2097.84 |
| 2.6B | 1024 | 4 | 2469.34 | 2074.72 | 1893.50 | 2120.22 |
| 2.6B | 1024 | 8 | 2904.01 | 2369.12 | 1999.82 | 2304.72 |
| 2.6B | 4096 | 1 | 2403.27 | 1990.85 | 1718.98 | 2398.70 |
| 2.6B | 4096 | 2 | 2867.94 | 2298.62 | 1984.75 | 2397.72 |
| 2.6B | 4096 | 4 | 3788.88 | 2818.43 | 2261.57 | 2401.67 |
| 2.6B | 4096 | 8 | 5562.20 | 3826.57 | 2877.71 | 2868.59 |

Output TPS uses `C*128 / observed_wall_seconds`.
The [summary CSV](../analysis/native_tp_dp_cohorts_v1/summary.csv) retains TPS, medians, and observed min/max values.
The [matched comparison CSV](../analysis/native_tp_dp_cohorts_v1/matched_comparisons.csv) labels its TP4/DP4 median ratios with `boundaries_match=false`.

## DP4 local timing and charged residual

Each trial retains every replica's observed local `generate` duration.
Idle replicas have no `generate` call and retain zero local duration.
The [worker CSV](../analysis/native_tp_dp_cohorts_v1/dp_workers.csv) contains 320 worker records.
The [worker summary](../analysis/native_tp_dp_cohorts_v1/dp_worker_summary.csv) contains 64 per-cell/per-replica summaries.

For each trial, the analyzer subtracts the longest active local interval from the observed parent interval.
It then takes the median of these paired residuals within the cell.
It does not subtract independently aggregated medians.
It never reconstructs DP4 parent time from worker maxima.

Cell medians range from 0.357926 ms to 0.687291 ms across 16 cells.
The residual ranges from 0.317805 ms to 0.812733 ms across all 80 DP4 trials.
This residual includes IPC, controller work, release skew, and worker glue.
It is not pure IPC time or an isolated causal overhead measurement.

## Reproduction and limits

The [JSON](../analysis/native_tp_dp_cohorts_v1/summary.json) retains all trial metrics, availability states, and 368 input hashes.
Eight SVG figures show wall-time and output-TPS curves for both sizes and contexts.
Their legends preserve API versus parent timing.
Example figures: [1.4B N4096 wall](../analysis/native_tp_dp_cohorts_v1/1.4B_N4096_median_wall_ms.svg), [2.6B N4096 TPS](../analysis/native_tp_dp_cohorts_v1/2.6B_N4096_median_output_TPS.svg).

Run the [separate analyzer](../scripts/analyze_native_tp_dp_cohorts.py) from the campaign repository:

```bash
python -m poc_ouro_parallelism_deep_discovery.scripts.analyze_native_tp_dp_cohorts \
  --output poc_ouro_parallelism_deep_discovery/analysis/NEW_TP_DP_ANALYSIS
```

The analyzer refuses an existing output directory.
Eleven new CPU tests pass.
The combined analyzer, DP protocol, and TP protocol regression passes all 32 tests.
A second analysis produces 13 byte-identical CSV/SVG files.
Its JSON matches except for the creation timestamp.

The five repetitions are calibration repeats, not independent natural populations.
C1 uses the first frozen pattern.
C2/C4/C8 alternate two repeated patterns.
There is no population confidence interval, held-out result, quality result, CDB factorial result, or steady-state serving claim.
Do not compare these constant-token values with the earlier changing-teacher workload as a system speedup.

Worker start: 2026-10-19 14:34:00 UTC.
Completed-data snapshot: 2026-10-19 14:38:07 UTC.
Reproduction checkpoint: 2026-10-19 14:39:56 UTC.
The validation artifact records the final verification cutoff.
Concurrent worker time is not additional campaign elapsed time.

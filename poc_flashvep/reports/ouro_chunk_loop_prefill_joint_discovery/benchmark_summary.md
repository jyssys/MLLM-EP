# Benchmark summary

All values below come from physical GPU4. No model-parallel or DP scaling result is included.
The main confirmation uses FA4, resident engines, equal calibration warm-up, and frozen natural generation.
The output cap is 128 tokens. EOS is enabled. These are bounded serving requests, not full reasoning accuracy tests.

## Corrected warm confirmation

Each cell summarizes paired cohorts across three process starts. The table reports means of cohort statistics.
It does not label a mean of cohort p95 values as a pooled-token p95.

| Prompt bin / concurrency | Policy | C / G | Cohorts | Long TTFT ms | ITL p95 ms | ITL p99 ms | Full cohort s | Output tok/s | SLO-qualified req/s | Peak allocated GiB |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2048 / 4 | atomic | 1024 / 4 | 10 | 235.083 | 26.753 | 79.200 | 4.280 | 128.583 | 0.251 | 14.898 |
| 2048 / 4 | split | 1024 / 1 | 10 | 290.587 | 27.512 | 108.761 | 3.716 | 137.963 | 0.269 | 15.008 |
| 2048 / 4 | mixed | 128 / 1 | 10 | 1477.618 | 90.165 | 100.124 | 4.902 | 104.515 | 0.393 | 15.056 |
| 2048 / 4 | mixed | 1024 / 1 | 10 | 337.428 | 25.724 | 121.210 | 3.639 | 140.826 | 0.275 | 15.200 |
| 2048 / 4 | grouped | 1024 / 2 | 10 | 258.736 | 26.174 | 138.205 | 3.644 | 140.615 | 0.275 | 15.293 |
| 8192 / 8 | atomic | 1024 / 4 | 10 | 924.384 | 38.510 | 364.162 | 5.925 | 172.820 | 0.169 | 15.368 |
| 8192 / 8 | split | 1024 / 1 | 10 | 1452.585 | 137.166 | 173.306 | 6.839 | 151.987 | 0.148 | 15.462 |
| 8192 / 8 | mixed | 128 / 1 | 10 | 7783.709 | 125.899 | 132.466 | 12.622 | 84.519 | 0.000 | 15.489 |
| 8192 / 8 | mixed | 1024 / 1 | 10 | 1648.820 | 122.176 | 128.611 | 6.517 | 157.341 | 0.123 | 15.591 |
| 8192 / 8 | grouped | 1024 / 2 | 10 | 1116.074 | 139.216 | 205.288 | 6.289 | 163.030 | 0.159 | 15.653 |

Full-cohort and throughput ratios can retain new-shape JIT effects. They are not equal-work architecture speedups.
SLO-qualified req/s uses a drained burst cohort. It is not steady-state open-loop goodput.
Thresholds are TTFT <= 5 s and maximum observed request ITL <= 100 ms. Zero values remain zero.

## Paired contrasts

Positive improvement means lower latency or higher rate. Speedup below 1.000× means a slowdown.
The 95% intervals resample paired cohorts with process-start blocks. They never resample tokens independently.

| Prompt / concurrency | Baseline | Metric | G2 speedup | Improvement % [95% CI] |
| --- | --- | --- | ---: | --- |
| 2048 / 4 | atomic C1024 | long_TTFT_s | 0.909× | -10.06 [-18.56, -1.07] |
| 2048 / 4 | atomic C1024 | ITL_p95_s | 1.022× | 2.16 [-0.55, 6.25] |
| 2048 / 4 | atomic C1024 | ITL_p99_s | 0.573× | -74.50 [-409.75, 6.78] |
| 2048 / 4 | atomic C1024 | output_tokens_s | 1.094× | 9.36 [-0.60, 27.92] |
| 2048 / 4 | mixed C1024 | long_TTFT_s | 1.304× | 23.32 [21.76, 25.07] |
| 2048 / 4 | mixed C1024 | ITL_p95_s | 0.983× | -1.75 [-3.59, 1.56] |
| 2048 / 4 | mixed C1024 | ITL_p99_s | 0.877× | -14.02 [-17.57, -10.98] |
| 2048 / 4 | mixed C1024 | output_tokens_s | 0.999× | -0.15 [-0.92, 1.02] |
| 8192 / 8 | atomic C1024 | long_TTFT_s | 0.828× | -20.74 [-36.74, -10.12] |
| 8192 / 8 | atomic C1024 | ITL_p95_s | 0.277× | -261.51 [-288.88, -243.74] |
| 8192 / 8 | atomic C1024 | ITL_p99_s | 1.774× | 43.63 [34.61, 49.12] |
| 8192 / 8 | atomic C1024 | output_tokens_s | 0.943× | -5.66 [-9.45, -2.70] |
| 8192 / 8 | mixed C1024 | long_TTFT_s | 1.477× | 32.31 [24.63, 37.60] |
| 8192 / 8 | mixed C1024 | ITL_p95_s | 0.878× | -13.95 [-20.41, -8.14] |
| 8192 / 8 | mixed C1024 | ITL_p99_s | 0.626× | -59.62 [-79.22, -41.31] |
| 8192 / 8 | mixed C1024 | output_tokens_s | 1.036× | 3.62 [0.66, 6.39] |

## Complete tables and limitations

- `results_raw.csv` contains every completed natural-serving cell, including calibration, warm-up, and rejected cold-order data.
- `paired_comparisons.csv` contains all registered baselines and confidence intervals.
- `experiment_matrix.csv` marks conditional Adaptive and DP work as NOT_RUN.
- `natural_output_parity.csv` preserves batch-dependent output differences.
- `full_R4_work_validation.csv` checks issued prompt work against four times the actual prompt length.
- No full GSM8K/MATH accuracy, HTTP serving, steady-state Poisson goodput, or adaptive-policy superiority is claimed.

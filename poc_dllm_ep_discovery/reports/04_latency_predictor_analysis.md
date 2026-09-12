# Latency predictor analysis

## Method

ExtraTrees regressors are evaluated leave-one-dataset-out: train on GSM8K/test on HumanEval and vice versa. The split tests workload transfer rather than random interpolation. Models are deliberately descriptive, not proposed as the contribution.

- M0: local token-expert pair count + layer.
- M1: M0 + active experts, p50/p90/max rows per expert, <=1 and <=4 tiny fractions.
- M2: M1 + global rank-load CV, token fanout and remote fraction.
- Robustness: repeat after removing observations above `(dataset, layer, rank)` p99 for each stage target.

## Results

| target/filter | M0 RMSE | M1 RMSE | M2 RMSE | M0→M1 | M1→M2 |
|---|---:|---:|---:|---:|---:|
| expert/raw | 0.05744 ms | 0.03534 | 0.03523 | **-38.5%** | -0.32% |
| expert/p99-trim | 0.05347 ms | 0.02929 | 0.02915 | **-45.2%** | -0.47% |
| dispatch/p99-trim | 0.09550 ms | 0.09459 | 0.09416 | -0.95% | -0.45% |
| combine/p99-trim | 0.07211 ms | 0.06690 | 0.06511 | -7.2% | -2.7% |

For robust expert timing, M1 has R² 0.9767 and 96.1% of predictions within 10%; M0 has R² 0.9223 and 86.5% within 10%. By contrast, the available route/shape variables explain little of dispatch timing, reinforcing that communication/runtime state is a distinct phenomenon rather than removable expert fragmentation.

Spearman correlations for expert time are 0.924 with pair count, 0.945 with active experts, 0.844 with p50 rows/expert and -0.843 with tiny<=4 fraction. Raw correlations are not causal; the leave-dataset-out improvement and fixed-M controls are the stronger evidence.

## Interpretation

The result falsifies a pair-count-only cost model, but M1's gain is primarily explanatory. The runtime already groups current-wave rows by expert in one fused MoE invocation. Changing shape across independent waves needs waiting, reordering or a new kernel, whose achievable E2E mass is bounded separately and is small.

Evidence: `PREDICTOR_RESULTS.csv`, `PREDICTOR_CORRELATIONS.csv`, `MATCHED_SAME_WORK_PAIRS.csv`.


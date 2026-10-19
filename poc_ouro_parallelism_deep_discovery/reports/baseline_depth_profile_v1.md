# Descriptive original-gate depth profile

## Scope

This analysis uses completed adaptive baseline requests, not a candidate runtime.
It adds no predictor, placement policy, scheduler, or GPU experiment.
The [analysis](../analysis/baseline_depth_profile_v1/summary.json) checks all admitted raw-record hashes.
It contains 128 request traces across four model/domain cells.
Each cell contains 32 requests.
The two model sizes use the same 64 distinct problem IDs.

The unchanged operating point is q=0.5, minimum R2, maximum R4, and `last_exited`.
Prefill executes full R4.
The following distributions exclude prefill and its first generated token.
Depth events come from the CDB reference runtime.
They do not establish adaptive native-vLLM TP4 behavior.

## Observed distributions

Each token-weighted fraction divides the corresponding depth count by recorded decode positions.
Request-weighted fractions first normalize each request, then average those fractions.
These denominators answer different descriptive questions.

| Base / domain | Recorded decode positions | Token-weighted R2 / R3 / R4 | Request-weighted R2 / R3 / R4 |
|---|---:|---|---|
| 1.4B / GSM8K | 3221 | 0.024216 / 0.891959 / 0.083825 | 0.023728 / 0.895837 / 0.080435 |
| 1.4B / MATH-500 | 60672 | 0.235611 / 0.719657 / 0.044732 | 0.135262 / 0.774377 / 0.090361 |
| 2.6B / GSM8K | 3305 | 0.007564 / 0.960061 / 0.032375 | 0.008005 / 0.964031 / 0.027963 |
| 2.6B / MATH-500 | 96482 | 0.100288 / 0.869458 / 0.030254 | 0.096749 / 0.874149 / 0.029102 |

R3 dominates these observed cells.
The 1.4B MATH cell has a larger token-weighted R2 fraction than its request-weighted fraction.
Long traces therefore influence aggregate pass work differently from an average request.
This observation does not identify an online admission rule or a distributed latency gain.

## Relative-position changes

The script divides each completed trace into four contiguous quarters.
Quarter boundaries use its final trace length.
Each table entry is the request-mean executed depth within that quarter.

| Base / domain | Quarter 1 | Quarter 2 | Quarter 3 | Quarter 4 | Mean first/final-quarter JSD, nats | Request-bootstrap 95% CI |
|---|---:|---:|---:|---:|---:|---|
| 1.4B / GSM8K | 3.048653 | 3.087437 | 3.071859 | 3.019363 | 0.024871 | [0.017399, 0.032833] |
| 1.4B / MATH-500 | 3.028767 | 2.962399 | 2.898991 | 2.930700 | 0.020727 | [0.015245, 0.026948] |
| 2.6B / GSM8K | 3.003081 | 3.034126 | 3.035792 | 3.006766 | 0.014587 | [0.010062, 0.019406] |
| 2.6B / MATH-500 | 2.967296 | 2.941936 | 2.930081 | 2.890151 | 0.012245 | [0.007584, 0.017526] |

JSD means Jensen–Shannon divergence with natural logarithms.
The script compares the R2/R3/R4 distributions of the first and final quarters within each request.
It then averages those request values.
Intervals use 4000 request-resampling bootstrap samples and seed 2026101911.
Tokens and quarters are not independent request samples.
The [editable plot](../analysis/baseline_depth_profile_v1/depth_by_relative_position.svg) shows the quarter means.
The JSON retains their separate request-bootstrap intervals.

## Architectural consequence and limits

These distributions can inform a declared workload model.
They do not supply future online gate decisions.
Final-length normalization also uses future information.
Neither quarter identity nor the final quarter distribution is a causal scheduler input here.

R3 dominance does not create three concurrent token fronts for one request.
The current token still follows the inspected recurrent dependency chain.
An implementation must charge gate evaluation, publication, KV validity, communication, and contention.
This analysis does not measure those costs.

The [depth-affinity screen](depth_affinity_capacity_screen.md) remains a separate unit-cost ownership model.
Its new-cohort data are in the [completed extension](../analysis/depth_affinity_extension_complete_v1/screen.json).
Its static partitions use observed histograms in hindsight.
Neither analysis establishes physical throughput or a speedup against native TP4.

No candidate changes status because of this descriptive supplement.
The original candidate held-out cohort remains untouched.

# Dense AR Layer PD: SLO calibration and goodput definitions

## Calibration evidence

The model is Qwen3-8B at revision `b968826d9c46dd6066d109eabc6255188de91218`.
Input length is 8000 tokens.
Calibration used 16 distinct prompts per start.
Each output cap used three independent low-load starts.
The total was 15 starts.
Requests ran serially to avoid a hidden Decode queue.

Evidence level: `MEASURED` for the calibration latencies.
The SLO factors and floors are research design choices.
They are not production-service standards.

The campaign froze SLOs before Dynamic GPU evaluation.
The immutable SHA256 is:

`c8aa94213e4f4ce54821d01ba985a8e9e88a20566a3e8d1f3cc8060d1b3926d9`

See the [threshold file](dense_ar_dynamic_layer_pd_crossover_20261113/configs/frozen_SLO_regimes.json).
See the [calibration inputs](dense_ar_dynamic_layer_pd_crossover_20261113/analysis/SLO_calibration_inputs.json).

## Frozen thresholds

Each cell gives TTFT / ITL / request TPOT in milliseconds.

| Output cap | TTFT-priority | Balanced | Decode-priority |
|---:|---:|---:|---:|
| 16 | 2000 / 54 / 52 | 5000 / 34 / 33 | 10000 / 30 / 29 |
| 32 | 2000 / 55 / 53 | 5000 / 34 / 33 | 10000 / 30 / 29 |
| 64 | 2000 / 55 / 53 | 5000 / 35 / 33 | 10000 / 31 / 29 |
| 128 | 2000 / 54 / 53 | 5000 / 34 / 33 | 10000 / 30 / 29 |
| 200 | 2000 / 53 / 53 | 5000 / 34 / 33 | 10000 / 30 / 29 |

The specification fixed TTFT floors of 2000/5000/10000ms before candidate measurements.
It used factors of 1.25/2/4 for TTFT.
It used factors of 2/1.25/1.10 for ITL and TPOT.
Each threshold is the larger applicable calibration product or floor.
The calculation rounds upward to whole milliseconds.

The previous 830.647ms TTFT, 49.872ms TPOT, and 57.164ms ITL thresholds remain a historical anchor.
They do not replace these frozen regimes.
Native-relative ITL p99 and TPOT p95 must also remain within 1.05×.

## Metric contract

TTFT starts at the scheduled open-loop arrival.
It ends when the coordinator receives the first user token.
There is no HTTP boundary in this benchmark.
Marker-free Service TTFT is `NOT_MEASURED`.

Request TPOT is first-to-last delivery time divided by the number of later tokens.
ITL is the interval between consecutive user-token deliveries.
ITL includes the first interval.
The analysis reports first ITL and steady ITL separately.
TBT uses this same delivery-interval definition.
It is not a second independent observation.

A request passes Decode SLO when both conditions hold:

1. Its average TPOT does not exceed the TPOT limit.
2. Its maximum ITL does not exceed the ITL limit.

Joint pass also requires TTFT within its limit.
Request SLO-goodput counts joint-passing requests per observation second.
Output-token SLO-goodput counts their output tokens per observation second.
The observation spans the first scheduled arrival through the last completion.
These are finite-cohort drain metrics.
They do not establish sustainable serving capacity.

Offered QPS is the Poisson rate parameter.
Scheduled trace QPS is the realized density of its finite scheduled offsets.
Observed arrival QPS uses coordinator observation timestamps when those timestamps exist.
Older discovery logs lack that boundary.
Their observed arrival QPS is `NOT_MEASURED`.
Earlier derived discovery snapshots called scheduled trace density “realized arrival QPS”.
The final analysis separates these fields.
Compute submission QPS uses the actual input RPC timestamps.
It includes intentional admission and Source queue effects.

## Statistical contract

The analysis uses paired independent server starts.
Requests and tokens do not substitute for independent starts.
It reports ratios of start-level metric means.
It enumerates the empirical paired bootstrap samples for the reported percentile interval.

A Decode point ratio below 1.05 is insufficient.
Both 95% confidence upper bounds must not exceed 1.05.
Absolute SLO pass is a separate result.
Maximum ITL and threshold exceedances remain visible even when the percentile gate passes.

The [final summary](dense_ar_dynamic_layer_pd_crossover_final_summary.md) gives the held-out results.
The [paired statistics](dense_ar_dynamic_layer_pd_crossover_20261113/analysis/heldout_paired_statistics.json) preserve every evaluated regime.

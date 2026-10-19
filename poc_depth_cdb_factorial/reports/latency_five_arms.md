# Measured five-arm latency and absolute scaling

Each table uses one fixed burst cohort.
C denotes total eligible requests across the system.
A0 uses physical GPU4. The other configurations use physical GPUs 4/5/6/7.

The model, online gate, KV policy, local capacity, and request IDs match across configurations.
Times are median cohort makespans over five paired repeats.
Speedup is the matching A0 median makespan divided by the method median makespan.
This is multi-request makespan scaling, not single-request latency speedup.

Hindsight uses a precomputed future-informed placement. Its reference-trace acquisition cost is outside the serving makespan.
The 95% intervals resample paired whole-cohort repeat IDs.

They are conditional on the frozen request cohorts and do not include new-query population uncertainty.

## C4

| Arm | Makespan, s | Speedup vs A0 [95% CI] | Median TTFT, ms | Mean TPOT, ms | P95 E2E, ms | Peak allocated/GPU, GiB |
|---|---|---|---|---|---|---|
| A0 | 5.351 | 1.000× [1.000, 1.000] | 142.54 | 20.42 | 5350.72 | 26.763 |
| A1 | 4.365 | 1.226× [1.133, 1.471] | 140.72 | 15.38 | 4288.20 | 26.755 |
| A2 | 4.172 | 1.282× [1.265, 1.478] | 145.25 | 14.56 | 4091.67 | 26.755 |
| A3 | 4.194 | 1.276× [1.233, 1.463] | 139.26 | 14.72 | 4159.42 | 26.755 |
| A4 | 4.332 | 1.235× [1.226, 1.396] | 140.65 | 15.40 | 4260.32 | 26.755 |

## C8

| Arm | Makespan, s | Speedup vs A0 [95% CI] | Median TTFT, ms | Mean TPOT, ms | P95 E2E, ms | Peak allocated/GPU, GiB |
|---|---|---|---|---|---|---|
| A0 | 5.923 | 1.000× [1.000, 1.000] | 142.60 | 22.54 | 5923.45 | 26.787 |
| A1 | 4.703 | 1.260× [1.147, 1.436] | 140.94 | 17.46 | 4702.59 | 26.772 |
| A2 | 4.623 | 1.281× [1.199, 1.402] | 139.83 | 17.19 | 4623.10 | 26.760 |
| A3 | 7.937 | 0.746× [0.739, 0.883] | 174.25 | 22.93 | 7673.86 | 26.772 |
| A4 | 8.145 | 0.727× [0.719, 0.793] | 173.25 | 24.03 | 8063.37 | 26.760 |

## C16

| Arm | Makespan, s | Speedup vs A0 [95% CI] | Median TTFT, ms | Mean TPOT, ms | P95 E2E, ms | Peak allocated/GPU, GiB |
|---|---|---|---|---|---|---|
| A0 | 7.036 | 1.000× [1.000, 1.000] | 148.61 | 27.01 | 7035.60 | 26.836 |
| A1 | 5.687 | 1.237× [1.198, 1.383] | 179.67 | 20.02 | 5687.01 | 26.804 |
| A2 | 5.528 | 1.273× [1.258, 1.399] | 179.40 | 20.04 | 5528.27 | 26.775 |
| A3 | 13.552 | 0.519× [0.483, 0.571] | 145.13 | 36.03 | 12133.46 | 26.804 |
| A4 | 13.349 | 0.527× [0.491, 0.579] | 144.05 | 37.45 | 12861.36 | 26.775 |

## C32

| Arm | Makespan, s | Speedup vs A0 [95% CI] | Median TTFT, ms | Mean TPOT, ms | P95 E2E, ms | Peak allocated/GPU, GiB |
|---|---|---|---|---|---|---|
| A0 | 13.091 | 1.000× [1.000, 1.000] | 3468.42 | 25.20 | 13090.52 | 26.836 |
| A1 | 6.296 | 2.079× [2.026, 2.277] | 175.73 | 22.40 | 6295.57 | 26.870 |
| A2 | 6.314 | 2.073× [2.053, 2.326] | 175.15 | 22.65 | 6314.45 | 26.816 |
| A3 | 26.391 | 0.496× [0.493, 0.543] | 175.67 | 61.32 | 22835.18 | 26.870 |
| A4 | 22.786 | 0.574× [0.547, 0.615] | 175.81 | 71.60 | 21984.13 | 26.816 |

## C64

| Arm | Makespan, s | Speedup vs A0 [95% CI] | Median TTFT, ms | Mean TPOT, ms | P95 E2E, ms | Peak allocated/GPU, GiB |
|---|---|---|---|---|---|---|
| A0 | 27.266 | 1.000× [1.000, 1.000] | 9783.92 | 26.22 | 27266.30 | 26.836 |
| A1 | 7.304 | 3.733× [3.456, 4.059] | 183.60 | 26.29 | 7303.95 | 26.947 |
| A2 | 7.323 | 3.724× [3.493, 4.075] | 187.37 | 26.47 | 7322.63 | 26.845 |
| A3 | 39.914 | 0.683× [0.640, 0.723] | 175.36 | 98.31 | 36530.86 | 26.947 |
| A4 | 38.451 | 0.709× [0.688, 0.726] | 189.74 | 97.14 | 35688.94 | 26.845 |

`metrics_raw.csv` contains requests/s, output tokens/s, ITL, goodput, queue time, actual output length, and loop work.
The goodput rule permits max(10 seconds, 200 milliseconds per emitted token).
This diagnostic is not a production latency guarantee.

Peak allocated memory comes from PyTorch, not an instantaneous whole-device NVML maximum.
Each memory cell is the median of the per-repeat maximum rank peak.
Each latency cell also uses the median of its per-repeat statistic.
The raw data also retain peak reserved memory and exact KV tensor bytes.

Loop-token counts describe scheduled inference work, not extra setup kernels during lazy graph capture.
The separate capture counters retain those setup events.

## Observer validity

25/25 trace/clean comparisons preserve complete token and depth traces.
The median observed trace-minus-clean time difference in those comparisons is 1.621%.
Each observer comparison has one paired run.
It cannot isolate observer cost from run-to-run variation. The causal overhead estimate remains `PARTIAL`.

Output-changing comparisons do not provide a validated observer-overhead estimate.
Trace CUDA-event intervals never replace clean makespan in the primary tables.

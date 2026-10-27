# Profiling and critical-path evidence

IN_PROGRESS: this is a preview, not a completed PoC.

## Separate heavy observer

| Model | EP | Rank | Observer window ms | Non-NCCL kernel union ms | NCCL union ms | No recorded activity ms | D2H events |
| --- | --- | --- | --- | --- | --- | --- | --- |
| ifm_ouro | 4 | 0 | 473.792 | 33.355 | 67.748 | 369.177 | 560 |
| ifm_ouro | 4 | 1 | 471.326 | 32.901 | 90.893 | 344.036 | 560 |
| ifm_ouro | 4 | 2 | 470.981 | 32.495 | 83.339 | 351.638 | 560 |
| ifm_ouro | 4 | 3 | 470.979 | 31.974 | 28.948 | 406.585 | 560 |
| ifm_ouro | 8 | 0 | 477.925 | 29.734 | 81.180 | 363.497 | 560 |
| ifm_ouro | 8 | 1 | 474.631 | 28.948 | 74.957 | 367.264 | 560 |
| ifm_ouro | 8 | 2 | 475.864 | 29.035 | 95.809 | 347.550 | 560 |
| ifm_ouro | 8 | 3 | 478.719 | 28.525 | 100.141 | 346.635 | 560 |
| ifm_ouro | 8 | 4 | 472.250 | 28.768 | 93.905 | 346.085 | 560 |
| ifm_ouro | 8 | 5 | 472.408 | 28.849 | 80.518 | 359.537 | 560 |
| ifm_ouro | 8 | 6 | 473.248 | 28.758 | 51.503 | 389.466 | 560 |
| ifm_ouro | 8 | 7 | 473.049 | 28.326 | 82.034 | 359.282 | 560 |
| ifm_huginn | 4 | 0 | 514.923 | 34.001 | 64.007 | 413.404 | 560 |
| ifm_huginn | 4 | 1 | 514.766 | 32.481 | 48.810 | 429.978 | 560 |
| ifm_huginn | 4 | 2 | 517.426 | 32.808 | 106.948 | 374.156 | 560 |
| ifm_huginn | 4 | 3 | 514.139 | 31.892 | 107.609 | 371.191 | 560 |
| foil1 | 4 | 0 | 76.183 | 6.948 | 3.497 | 65.183 | 90 |
| foil1 | 4 | 1 | 77.037 | 7.043 | 8.370 | 61.068 | 90 |
| foil1 | 4 | 2 | 75.713 | 6.994 | 7.670 | 60.496 | 90 |
| foil1 | 4 | 3 | 75.027 | 6.893 | 4.546 | 63.039 | 90 |

These CUPTI intervals come from B1, prefix-128 diagnostic forwards.
They are not the clean serving interval and not a complete generated-request E2E breakdown.
Kernel unions avoid summing overlapping services.
NCCL activity includes waits and cannot identify pure network transfer time.
No recorded GPU activity excludes visible kernels, copies, and memset.
It does not measure SM occupancy or prove hardware idle.

The bridge performs one count All-Gather, two All-to-All calls, and one restoration All-Reduce per MoE invocation.
IFM executes 112 MoE invocations per forward. Foil executes 18.
Count conversion and Python split preparation cause CPU/GPU coordination.
The original local layout also creates metadata and gather/scatter work.
Those mechanisms are generic runtime costs, not demonstrated loop-specific overhead.

## Whole-E2E percentages

`e2e_breakdown.md` uses complete rollouts, not extrapolated diagnostic forwards.
Its event spans include submission gaps and stream waiting.
The observer checks token parity, overhead, and event-to-host alignment.
Failed timing gates remain visible.
Fine and coarse observers do not enter one pooled component distribution.

## Rank imbalance and clocks

`rank_imbalance.csv` records actual routing counts and API event-span dispersion.
Max-minus-mean event duration is not an exact rank-wait measurement.
`collective_arrival_skew.csv` uses same-host profiler submission timestamps.
Submission skew is neither isolated network latency nor exact GPU waiting.
Editable SVG timelines retain the distinction between compute, NCCL, and memory copies.

## Unavailable counters

Nsight Compute hardware-counter measurements are unavailable in this campaign.
Actual HBM read traffic, effective HBM bandwidth, tensor-core occupancy, and L2 hit rates remain NOT_AVAILABLE.
Checkpoint weight size is not a measurement of HBM traffic.
Persistent KV memory is N/A because these official adapter paths retain no cache.

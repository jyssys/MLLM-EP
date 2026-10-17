# Profiling

## Clean exact-reference sweep

The table shows means of per-group medians.
Units are milliseconds.
Each cached result averages 32 actual fixed-input forwards.
It is not natural generation latency.

| Context | B1 prefill / cached step | B2 prefill / cached step | B4 prefill / cached step | B8 prefill / cached step |
|---|---|---|---|---|
| 1,024 | 85.037 / 71.950 | 88.773 / 67.427 | 158.497 / 67.697 | 301.736 / 67.777 |
| 4,096 | 177.383 / 63.147 | 334.871 / 62.932 | 643.032 / 62.942 | 1,259.782 / 64.594 |
| 8,192 | 376.285 / 66.704 | 723.521 / 66.114 | 1,421.282 / 67.301 | 2,840.988 / 102.870 |
| 16,384 | 880.956 / 65.498 | 1,726.245 / 66.547 | 3,432.235 / 102.489 | CAPACITY_SAFETY_SKIP |

The following B1 intervals use the 95% request-group bootstrap.

| Context | Prefill interval, ms | Cached-step interval, ms |
|---|---|---|
| 1K | [75.619,103.538] | [67.630,80.356] |
| 4K | [177.285,177.470] | [63.069,63.221] |
| 8K | [376.109,376.447] | [66.532,66.864] |
| 16K | [880.651,881.281] | [65.185,65.863] |

The 1K interval retains the initial request group's startup variation.
The narrow later intervals reflect eight related calibration questions, not broad workload coverage.

Raw trials and memory measurements remain in `outputs/profile_P*_v1`.
`metrics/analysis_final_v1/profile_summary.csv` contains all groups and confidence intervals.

## Matched light component lane

These are separate single-family CUDA-event runs.
Their intervals can contain enqueue gaps.
Percentages use that lane's matched clean wall denominator.
Do not add the percentages as an exclusive E2E partition.

| Phase / context | Matched clean forward | Attention core | MLP | KV update | Output projection |
|---|---:|---:|---:|---:|---:|
| Decode / 1K | 66.997 | 4.869, 7.27% | 10.012, 14.94% | 3.728, 5.56% | 3.254, 4.86% |
| Decode / 16K | 89.089 | 6.544, 7.35% | 12.303, 13.81% | 20.786, 23.33% | 1.094, 1.23% |
| Prefill / 16K | 879.832 | 325.764, 37.03% | 168.858, 19.19% | 0.313, 0.04% | 17.615, 2.00% |

All listed component families pass the 10% observer gate.
They use one calibration request with five technical repetitions.
The 1K prefill attention, KV, and MLP families fail that gate.
Their raw results remain available but do not support headline shares.

The long-context eager reference exposes substantial KV append activity.
The official cache appends K and V with `torch.cat`.
This behavior copies the growing historical cache.
The optimized native runtime already uses paged KV writes.
The study does not claim reference-cache overhead as a new recurrent architecture opportunity.

## Recurrence diagnostic

| R | Executed layers | Unique weights, GiB | KV at B1/1K, GiB | Median prefill, ms |
|---|---:|---:|---:|---:|
| 1 | 24 | 2.672249 | 0.1875 | 20.544 |
| 2 | 48 | 2.672249 | 0.3750 | 38.667 |
| 4 | 96 | 2.672249 | 0.7500 | 76.273 |

Tied weight capacity stays constant while executed work and KV storage increase.
This establishes a storage/work asymmetry.
It does not establish a new removable critical-path cost.
R1 and R2 change model semantics and have no quality-equivalence claim.

## Optimized native hardware diagnostics

Every row is `REFERENCE_PARTIAL` under the frozen cross-backend numerical gate.
These are actual clean B1 measurements with 32 forced output tokens.
E2E includes prefill and returned fixed-output completion.

| Context | TP1 TTFT / TPOT / E2E | TP2 TTFT / TPOT / E2E | TP4 TTFT / TPOT / E2E |
|---|---|---|---|
| 1K | 66.680 / 10.418 / 391.332 | 68.996 / 9.222 / 357.854 | 62.870 / 7.747 / 305.509 |
| 4K | 95.722 / 11.593 / 457.333 | 83.797 / 9.772 / 392.732 | 73.037 / 8.125 / 329.581 |
| 8K | 203.093 / 12.704 / 604.002 | 141.097 / 10.688 / 481.286 | 107.446 / 8.813 / 387.752 |
| 16K | 485.712 / 14.681 / 953.257 | 316.973 / 11.668 / 689.608 | 206.178 / 9.717 / 517.637 |

Each cell contains eight requests with five technical repetitions each.
`native_summary.csv` contains request-bootstrap intervals.
`native_paired.csv` contains paired TP1-to-TP2/4 reductions and intervals.
The study does not interpret HF-to-vLLM latency differences as an architecture speedup.

The native profile contains real NCCL AllReduce and custom `cross_device_reduce_1stage` kernels.
It also contains paged cache-write, fused normalization, GEMM, and FlashAttention kernels.
There is no sparse expert Dispatch/Combine collective.
`native_kernel_service.csv` reports rank-local service and kernel counts.
It does not report those sums as global critical-path percentages.

## Critical-path conclusion

Long prefill has substantial attention and MLP intervals in the exact reference.
Cached decode has a small measured attention-core interval in the light lane.
The eager cache adds context-dependent copy/append work.
Host launch gaps appear in the heavy trace, but its observer overhead invalidates clean-time attribution.

A complete clean exclusive breakdown is `PARTIAL`.
QKV, norm, gate, and launch activity remain available as diagnostic trace-local measurements.
Their clean exclusive percentages are `UNRESOLVED`.
The study does not manufacture a 100% clean-time table from inclusive event sums.

Figures and per-loop data are in `metrics/presentation_v1`.
The per-loop heatmap uses actual observer intervals, not independent statistical samples.
`figures/actual_trace_timeline.png` shows a measured single-GPU cached-forward timeline.
Its title retains the heavy observer warning.
There is no synchronized measured four-GPU pipeline timeline.

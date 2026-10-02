# Native MoE/EP component evidence

All numbers below are local CUDA spans. We select one cumulative-critical rank and keep dispatch/expert/combine on that same rank; we do not sum unrelated rank maxima or subtract cross-device timestamps. Prepare/dispatch can include peer producer-readiness waits and host layout work, not just network service. The active mixed-stage kernel contains resident decode rows as well as prefill rows; fused expert time cannot be allocated between them by a FLOPs ratio.

| Arm | Input | n | Dispatch total | Expert total | Max-rank expert total | Combine total | MoE total | Max prepare span | Overhead screen |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| FULL | VISUAL_PREFILL | 3 | 48.93 [48.41, 49.96] | 25.25 [25.21, 25.33] | 25.98 [25.87, 26.03] | 5.82 [5.62, 6.09] | 80.00 [79.35, 81.27] | 29.20 [29.07, 29.34] | USABLE_LOCAL_SPAN |
| FULL | TOKEN_MATCHED_TEXT | 3 | 21.49 [20.43, 23.35] | 24.75 [24.72, 24.82] | 25.00 [24.85, 25.26] | 6.42 [5.62, 7.99] | 52.66 [50.77, 56.06] | 1.32 [0.71, 2.36] | UNVALIDATED |
| CP512 | VISUAL_PREFILL | 3 | 104.35 [103.35, 105.53] | 83.18 [82.68, 83.55] | 86.01 [85.85, 86.28] | 17.40 [16.45, 18.65] | 204.93 [203.31, 207.73] | 28.95 [28.80, 29.04] | USABLE_LOCAL_SPAN |
| CP512 | TOKEN_MATCHED_TEXT | 3 | 74.73 [73.18, 76.35] | 79.25 [78.65, 79.63] | 81.48 [81.06, 81.81] | 21.54 [20.93, 21.98] | 175.51 [174.63, 176.69] | 0.93 [0.87, 1.00] | USABLE_LOCAL_SPAN |
| LP_AUTO | VISUAL_PREFILL | 3 | 48.70 [47.84, 49.91] | 25.54 [25.09, 26.38] | 26.33 [25.81, 27.23] | 6.07 [5.73, 6.62] | 80.31 [78.72, 82.91] | 29.47 [29.38, 29.61] | USABLE_LOCAL_SPAN |
| LP_AUTO | TOKEN_MATCHED_TEXT | 3 | 18.33 [17.75, 19.11] | 24.83 [24.69, 24.92] | 24.89 [24.69, 25.04] | 5.48 [5.34, 5.67] | 48.64 [47.86, 49.66] | 0.61 [0.55, 0.66] | USABLE_LOCAL_SPAN |
| LP_G8 | VISUAL_PREFILL | 3 | 48.86 [48.28, 49.56] | 24.95 [24.88, 25.00] | 25.67 [25.47, 25.77] | 5.90 [5.74, 6.13] | 79.71 [79.37, 80.39] | 30.03 [29.43, 31.13] | USABLE_LOCAL_SPAN |
| LP_G8 | TOKEN_MATCHED_TEXT | 3 | 19.54 [18.52, 21.48] | 25.16 [24.94, 25.55] | 25.21 [25.04, 25.55] | 5.49 [5.28, 5.72] | 50.19 [48.89, 52.76] | 0.87 [0.80, 0.92] | USABLE_LOCAL_SPAN |
| CP256 | VISUAL_PREFILL | 3 | 188.66 [159.53, 245.57] | 160.16 [157.21, 163.40] | 165.51 [159.55, 177.12] | 51.04 [39.03, 74.09] | 399.86 [355.77, 483.07] | 28.25 [28.24, 28.25] | USABLE_LOCAL_SPAN |
| CP256 | TOKEN_MATCHED_TEXT | 3 | 157.62 [139.15, 183.94] | 153.28 [152.67, 153.98] | 161.37 [154.80, 169.27] | 55.23 [47.09, 66.40] | 366.12 [339.41, 403.02] | 1.83 [1.30, 2.78] | USABLE_LOCAL_SPAN |
| CP1024 | VISUAL_PREFILL | 3 | 62.27 [61.56, 63.11] | 44.58 [44.24, 45.18] | 44.63 [44.29, 45.18] | 6.24 [6.11, 6.43] | 113.10 [112.07, 113.76] | 28.33 [28.25, 28.40] | USABLE_LOCAL_SPAN |
| CP1024 | TOKEN_MATCHED_TEXT | 3 | 33.59 [32.21, 34.45] | 41.89 [41.59, 42.46] | 43.07 [42.61, 43.58] | 9.58 [9.41, 9.75] | 85.06 [84.08, 85.62] | 0.68 [0.58, 0.79] | USABLE_LOCAL_SPAN |
| LP_G16 | VISUAL_PREFILL | 3 | 47.65 [46.77, 49.40] | 26.10 [25.88, 26.44] | 26.10 [25.88, 26.44] | 5.10 [4.87, 5.44] | 78.85 [77.52, 81.28] | 29.41 [29.33, 29.50] | USABLE_LOCAL_SPAN |
| LP_G16 | TOKEN_MATCHED_TEXT | 3 | 19.66 [18.63, 20.55] | 25.41 [24.36, 26.04] | 25.91 [25.61, 26.12] | 6.36 [5.53, 7.79] | 51.43 [50.21, 52.71] | 1.07 [0.79, 1.33] | USABLE_LOCAL_SPAN |
| LP_G24 | VISUAL_PREFILL | 3 | 73.55 [48.79, 86.66] | 25.52 [25.35, 25.67] | 29.19 [26.06, 30.85] | 10.22 [5.65, 12.76] | 109.29 [79.78, 124.98] | 34.28 [29.31, 37.07] | USABLE_LOCAL_SPAN |
| LP_G24 | TOKEN_MATCHED_TEXT | 3 | 41.73 [20.13, 52.70] | 25.36 [25.22, 25.51] | 28.37 [25.22, 29.98] | 10.30 [5.45, 12.77] | 77.39 [50.81, 90.90] | 2.41 [1.02, 3.22] | USABLE_LOCAL_SPAN |
| LP_G48 | VISUAL_PREFILL | 3 | 75.01 [53.14, 86.37] | 25.50 [25.42, 25.62] | 29.29 [26.04, 30.92] | 10.37 [5.33, 13.06] | 110.87 [83.92, 124.85] | 36.30 [35.07, 37.28] | USABLE_LOCAL_SPAN |
| LP_G48 | TOKEN_MATCHED_TEXT | 3 | 41.66 [19.91, 53.30] | 25.38 [25.29, 25.53] | 28.50 [25.29, 30.28] | 10.42 [5.22, 13.28] | 77.46 [50.42, 92.11] | 2.56 [0.93, 3.44] | USABLE_LOCAL_SPAN |

## Route/weight-working-set accounting (independent rich lane)

| Arm | Input | Reload amplification | Weight-access proxy (GB) | Logical remote dispatch (GB) |
| --- | --- | --- | --- | --- |
| FULL | VISUAL_PREFILL | 1.00 [1.00, 1.00] | 56.623 | 1.079 |
| FULL | TOKEN_MATCHED_TEXT | 1.00 [1.00, 1.00] | 44.185 | 1.092 |
| CP256 | VISUAL_PREFILL | 7.14 [7.14, 7.14] | 403.959 | 1.079 |
| CP256 | TOKEN_MATCHED_TEXT | 5.47 [5.47, 5.47] | 241.941 | 1.091 |
| CP512 | VISUAL_PREFILL | 3.80 [3.80, 3.80] | 214.866 | 1.079 |
| CP512 | TOKEN_MATCHED_TEXT | 3.06 [3.06, 3.06] | 135.301 | 1.092 |
| CP1024 | VISUAL_PREFILL | 1.97 [1.97, 1.97] | 111.670 | 1.079 |
| CP1024 | TOKEN_MATCHED_TEXT | 1.75 [1.75, 1.75] | 77.130 | 1.092 |
| LP_AUTO | VISUAL_PREFILL | 1.00 [1.00, 1.00] | 56.623 | 1.079 |
| LP_AUTO | TOKEN_MATCHED_TEXT | 1.00 [1.00, 1.00] | 44.185 | 1.092 |
| LP_G8 | VISUAL_PREFILL | 1.00 [1.00, 1.00] | 56.623 | 1.079 |
| LP_G8 | TOKEN_MATCHED_TEXT | 1.00 [1.00, 1.00] | 44.185 | 1.092 |
| LP_G16 | VISUAL_PREFILL | 1.00 [1.00, 1.00] | 56.623 | 1.079 |
| LP_G16 | TOKEN_MATCHED_TEXT | 1.00 [1.00, 1.00] | 44.185 | 1.092 |
| LP_G24 | VISUAL_PREFILL | 1.00 [1.00, 1.00] | 56.623 | 1.079 |
| LP_G24 | TOKEN_MATCHED_TEXT | 1.00 [1.00, 1.00] | 44.185 | 1.092 |
| LP_G48 | VISUAL_PREFILL | 1.00 [1.00, 1.00] | 56.623 | 1.079 |
| LP_G48 | TOKEN_MATCHED_TEXT | 1.00 [1.00, 1.00] | 44.185 | 1.092 |

Each expert is **9,437,184 bytes** from actual loaded BF16 gate/up+down tensors. ReloadAmp counts reactivations of prompt experts across chunks, not actual HBM reads. Payload is destination-deduplicated hidden-vector accounting; metadata/padding/wire traffic is not inferred. Per-rank send/recv matrices, per-layer times, active-expert counts and weight shapes remain in raw artifacts.

These counters and totals attach to **prefill-active mixed kernel invocations**, not an isolated prefill-only GEMM. LP also executes resident-only layers on each stage; their useful decode work is not charged to the prompt reload proxy. Low reload therefore does not imply identical whole-system HBM traffic or TTFT.

Logical top8 paths across all 48 layers: PASS. Complete four-rank **active injected-prefill** component frames: PASS. All captured frames complete: **False**; 288 partial frames (0 active-prefill) are retained in the trace-frame audit, never filled. Post-injection capture-boundary omissions are not inferred missing executions; native stage/token correctness is checked separately.

## Instrumentation overhead

| Metric | Trace minus hook-bypass (%) |
| --- | --- |
| lm_start_to_first_token_ms | 6.99 [5.48, 8.32] |
| mean_itl_ms | 0.75 [-0.09, 1.43] |
| peak_itl_ms | 2.90 [0.42, 5.09] |
| ttft_ms | 6.53 [5.25, 7.53] |

| Arm | Input | TTFT overhead (%) | Mean ITL overhead (%) | Stratum usable |
| --- | --- | --- | --- | --- |
| FULL | VISUAL_PREFILL | 3.46 [1.99, 5.56] | -0.20 [-0.44, -0.03] | True |
| FULL | TOKEN_MATCHED_TEXT | 10.30 [8.09, 14.23] | 0.46 [0.14, 0.66] | False |
| CP512 | VISUAL_PREFILL | 6.27 [4.12, 7.90] | 0.80 [0.67, 0.92] | True |
| CP512 | TOKEN_MATCHED_TEXT | 8.19 [7.99, 8.44] | 0.73 [0.41, 1.06] | True |
| LP_AUTO | VISUAL_PREFILL | 4.66 [4.40, 5.10] | 0.84 [0.18, 1.68] | True |
| LP_AUTO | TOKEN_MATCHED_TEXT | 7.88 [6.33, 10.07] | 0.55 [0.23, 0.92] | True |
| LP_G8 | VISUAL_PREFILL | 7.25 [5.92, 8.75] | 1.28 [0.44, 1.71] | True |
| LP_G8 | TOKEN_MATCHED_TEXT | 7.83 [4.72, 9.74] | 1.47 [1.30, 1.75] | True |
| CP256 | VISUAL_PREFILL | 8.78 [2.49, 16.04] | 1.25 [-0.48, 3.22] | True |
| CP256 | TOKEN_MATCHED_TEXT | 6.33 [5.19, 7.23] | -1.54 [-7.34, 1.77] | True |
| CP1024 | VISUAL_PREFILL | 4.49 [2.56, 6.32] | 0.83 [0.15, 1.48] | True |
| CP1024 | TOKEN_MATCHED_TEXT | 5.48 [3.64, 7.31] | 0.25 [0.20, 0.35] | True |
| LP_G16 | VISUAL_PREFILL | 8.65 [7.83, 9.19] | 1.81 [1.41, 2.11] | True |
| LP_G16 | TOKEN_MATCHED_TEXT | 8.02 [6.96, 10.11] | 1.59 [1.02, 2.26] | True |
| LP_G24 | VISUAL_PREFILL | -1.24 [-18.34, 7.72] | -6.13 [-14.78, 2.41] | True |
| LP_G24 | TOKEN_MATCHED_TEXT | 6.60 [3.83, 8.13] | 2.23 [1.84, 2.42] | True |
| LP_G48 | VISUAL_PREFILL | 7.37 [7.20, 7.60] | 3.76 [3.43, 4.13] | True |
| LP_G48 | TOKEN_MATCHED_TEXT | 7.31 [4.77, 9.41] | 3.50 [2.08, 4.68] | True |

Component breakdown usable under the <=10% screening rule: **False**. The individual failing strata are UNVALIDATED and cannot support an EP-common verdict; passing strata remain separately labeled. Unhooked clean serving results above remain separate.

## Per-layer cumulative critical-rank spans

| Arm | Input | Dispatch/layer ms | Expert/layer ms | Combine/layer ms | MoE/layer ms |
| --- | --- | --- | --- | --- | --- |
| FULL | VISUAL_PREFILL | 1.0194 | 0.5261 | 0.1212 | 1.6666 |
| FULL | TOKEN_MATCHED_TEXT | 0.4476 | 0.5157 | 0.1338 | 1.0972 |
| CP512 | VISUAL_PREFILL | 2.1740 | 1.7329 | 0.3624 | 4.2694 |
| CP512 | TOKEN_MATCHED_TEXT | 1.5568 | 1.6510 | 0.4487 | 3.6565 |
| LP_AUTO | VISUAL_PREFILL | 1.0146 | 0.5321 | 0.1265 | 1.6732 |
| LP_AUTO | TOKEN_MATCHED_TEXT | 0.3819 | 0.5173 | 0.1141 | 1.0133 |
| LP_G8 | VISUAL_PREFILL | 1.0180 | 0.5198 | 0.1229 | 1.6606 |
| LP_G8 | TOKEN_MATCHED_TEXT | 0.4072 | 0.5241 | 0.1144 | 1.0457 |
| CP256 | VISUAL_PREFILL | 3.9304 | 3.3367 | 1.0634 | 8.3305 |
| CP256 | TOKEN_MATCHED_TEXT | 3.2838 | 3.1933 | 1.1505 | 7.6276 |
| CP1024 | VISUAL_PREFILL | 1.2974 | 0.9288 | 0.1300 | 2.3562 |
| CP1024 | TOKEN_MATCHED_TEXT | 0.6998 | 0.8727 | 0.1996 | 1.7720 |
| LP_G16 | VISUAL_PREFILL | 0.9927 | 0.5437 | 0.1063 | 1.6427 |
| LP_G16 | TOKEN_MATCHED_TEXT | 0.4097 | 0.5293 | 0.1326 | 1.0715 |
| LP_G24 | VISUAL_PREFILL | 1.5323 | 0.5317 | 0.2129 | 2.2769 |
| LP_G24 | TOKEN_MATCHED_TEXT | 0.8693 | 0.5283 | 0.2147 | 1.6123 |
| LP_G48 | VISUAL_PREFILL | 1.5626 | 0.5312 | 0.2160 | 2.3098 |
| LP_G48 | TOKEN_MATCHED_TEXT | 0.8679 | 0.5288 | 0.2172 | 1.6138 |

Division is by **48 original decoder layers**, including all CP chunks for a
layer. It is not a per-chunk average or an attribution of fused mixed expert
time to individual modalities. Full/text's component lane exceeded the frozen
10% overhead screen (10.302% mean TTFT) and is diagnostic **UNVALIDATED**; it is
not used to prove an EP-common residual. CP/LP strata are screened separately.

## Accumulated victim dispatch excess (diagnostic)

Paired same forced decode position, sum of local max-duration differences across
the active layers. This is a duration proxy, not a synchronized global clock,
CWT, isolated wire time or a new primary serving metric.

| Arm | Input | Wave-peak accumulated excess ms [95% CI] |
| --- | --- | --- |
| FULL | VISUAL_PREFILL | 30.58 [29.76, 31.39] |
| FULL | TOKEN_MATCHED_TEXT | 2.91 [2.79, 3.09] |
| CP512 | VISUAL_PREFILL | 27.67 [26.23, 29.00] |
| CP512 | TOKEN_MATCHED_TEXT | 1.64 [1.35, 1.93] |
| LP_AUTO | VISUAL_PREFILL | 29.93 [29.88, 30.01] |
| LP_AUTO | TOKEN_MATCHED_TEXT | 0.63 [0.38, 0.82] |
| LP_G8 | VISUAL_PREFILL | 29.79 [29.01, 30.69] |
| LP_G8 | TOKEN_MATCHED_TEXT | 1.33 [0.39, 2.84] |

## EP-common conclusion

**NOT_CONFIRMED**, not a claim that EP is free. LP preserves Full-like active
prefill expert execution, whereas CP512 fragments it. The very long first
prepare interval can be peer-producer readiness, because DP padding exchange
occurs before the visual encoder. It is not measured intrinsic non-preemptible
A2A service/MNCD. No isolated network/readiness subtraction or architecture-
matched non-EP control is available here. The separate cached producer diagnostic
is described in modality_analysis.md. Large local communication span share alone
does not establish the specified common EP lower-bound gate.

Hardware DRAM/HBM reads, wire metadata/padding bytes, isolated CWT and Nsight
profiling are **NOT_RUN**, never filled with estimates. ExpertReloadAmp and the
9MiB/expert working-set sum are structural proxies, not measured HBM traffic.

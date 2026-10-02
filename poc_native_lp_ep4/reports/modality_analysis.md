# Matched-token modality comparison

Visual: 2029 total/1980 visual rows. Text-only: 2033 rows, same model/dtype/runtime. Positive delta means visual is slower. Wave pairs are matched by arm/repeat; 48 layers are not independent samples.

| Arm | Peak VSEL delta (ms) | TTFT delta (ms) | LM-entry wall delta (ms) | Peak ITL delta (ms) |
| --- | --- | --- | --- | --- |
| CP512 | 105.96 [100.20, 112.57] | 112.97 [101.27, 124.54] | -0.06 [-7.68, 8.45] | 106.07 [99.84, 113.06] |
| FULL | 104.89 [99.97, 110.60] | 113.41 [106.36, 120.23] | 5.19 [-0.32, 9.92] | 108.13 [102.61, 113.64] |
| LP_AUTO | 102.43 [95.88, 107.89] | 97.77 [71.58, 114.84] | -3.51 [-22.49, 9.94] | 100.08 [93.48, 105.69] |
| LP_G8 | 101.95 [95.98, 109.78] | 112.59 [107.59, 118.64] | 9.59 [4.92, 14.50] | 101.94 [95.70, 110.10] |

## Local component diagnostic

| Arm | Dispatch delta (ms) | Expert delta (ms) | Combine delta (ms) | Max dispatch span delta (ms) |
| --- | --- | --- | --- | --- |
| CP1024 | 28.69 [28.05, 29.35] | 2.69 [1.86, 3.56] | -3.34 [-3.46, -3.23] | 27.64 [27.59, 27.74] |
| CP256 | 31.04 [11.10, 61.63] | 6.89 [4.03, 10.73] | -4.18 [-12.17, 7.69] | 26.42 [25.47, 26.95] |
| CP512 | 29.63 [27.83, 30.88] | 3.93 [3.84, 4.03] | -4.14 [-5.24, -2.29] | 28.02 [27.88, 28.13] |
| FULL | 27.44 [26.61, 27.97] | 0.50 [0.48, 0.51] | -0.61 [-1.89, 0.12] | 27.87 [26.98, 28.36] |
| LP_AUTO | 30.37 [29.25, 31.78] | 0.71 [0.21, 1.46] | 0.59 [0.20, 1.28] | 28.86 [28.75, 28.98] |
| LP_G16 | 27.99 [26.97, 28.85] | 0.69 [-0.17, 2.08] | -1.26 [-2.36, -0.66] | 28.34 [28.00, 28.71] |
| LP_G24 | 31.82 [28.65, 34.32] | 0.16 [0.12, 0.21] | -0.08 [-0.44, 0.19] | 31.87 [28.29, 33.86] |
| LP_G48 | 33.35 [32.22, 34.60] | 0.11 [0.08, 0.17] | -0.06 [-0.57, 0.29] | 33.74 [33.25, 34.14] |
| LP_G8 | 29.32 [26.80, 30.93] | -0.21 [-0.59, 0.02] | 0.41 [0.27, 0.55] | 29.15 [28.51, 30.34] |

| Arm | Input | Mean destination fanout | Mean layer rank max/mean | Logical dispatch (GB, 48 layers) |
| --- | --- | --- | --- | --- |
| FULL | VISUAL_PREFILL | 3.623 | 1.236 | 1.445 |
| FULL | TOKEN_MATCHED_TEXT | 3.629 | 1.336 | 1.450 |
| CP256 | VISUAL_PREFILL | 3.624 | 1.236 | 1.446 |
| CP256 | TOKEN_MATCHED_TEXT | 3.626 | 1.338 | 1.449 |
| CP512 | VISUAL_PREFILL | 3.624 | 1.235 | 1.446 |
| CP512 | TOKEN_MATCHED_TEXT | 3.627 | 1.339 | 1.450 |
| CP1024 | VISUAL_PREFILL | 3.623 | 1.236 | 1.445 |
| CP1024 | TOKEN_MATCHED_TEXT | 3.629 | 1.336 | 1.450 |
| LP_AUTO | VISUAL_PREFILL | 3.623 | 1.236 | 1.445 |
| LP_AUTO | TOKEN_MATCHED_TEXT | 3.629 | 1.336 | 1.450 |
| LP_G8 | VISUAL_PREFILL | 3.623 | 1.236 | 1.445 |
| LP_G8 | TOKEN_MATCHED_TEXT | 3.629 | 1.336 | 1.450 |
| LP_G16 | VISUAL_PREFILL | 3.623 | 1.236 | 1.445 |
| LP_G16 | TOKEN_MATCHED_TEXT | 3.629 | 1.336 | 1.450 |
| LP_G24 | VISUAL_PREFILL | 3.623 | 1.236 | 1.445 |
| LP_G24 | TOKEN_MATCHED_TEXT | 3.629 | 1.336 | 1.450 |
| LP_G48 | VISUAL_PREFILL | 3.623 | 1.236 | 1.445 |
| LP_G48 | TOKEN_MATCHED_TEXT | 3.629 | 1.336 | 1.450 |

A large first-layer prepare span alone cannot identify an intrinsic visual A2A mechanism: the installed DP batch-size synchronization precedes vision encoding, so a peer can wait inside prepare for the visual producer. Lifecycle and optional lossless-encoder controls must be interpreted separately. Host lifecycle timestamps are not GPU collective wait time.

## Stable-denominator serving penalty

| Arm | Paired visual-minus-text peak ITL (%) [95% CI] | Pairs |
| --- | --- | --- |
| CP512 | 99.76 [94.15, 106.28] | 10 |
| FULL | 105.12 [97.51, 112.06] | 10 |
| LP_AUTO | 88.60 [79.08, 97.45] | 10 |
| LP_G8 | 88.09 [82.77, 93.92] | 10 |

Frozen G8 logical dispatch across all 48 layers is **1.445331 GB visual** versus **1.450332 GB text**. Destination fanout is 3.6231 versus 3.6285; mean layer assignment max/mean is 1.2364 versus 1.3365. Visual is not larger in these logical traffic/skew counters. These actual measurements argue against explaining the large victim penalty solely as greater visual A2A payload/fanout.

## Actual producer lifecycle (separate light trace)

| Arm | Input | Admission→first worker ms | Vision encoder CUDA ms | Initial input pipeline CUDA ms | Arrival→LM entry ms |
| --- | --- | --- | --- | --- | --- |
| FULL | VISUAL_PREFILL | 80.02 [77.68, 82.48] | 31.65 [29.31, 35.11] | 37.08 [34.74, 40.54] | 211.41 [205.18, 217.17] |
| FULL | TOKEN_MATCHED_TEXT | 0.46 [0.42, 0.50] | N/A | 2.28 [2.14, 2.55] | 106.80 [103.06, 112.64] |
| CP512 | VISUAL_PREFILL | 82.55 [81.39, 83.27] | 29.78 [29.28, 30.06] | 35.02 [34.53, 35.29] | 209.57 [208.40, 210.45] |
| CP512 | TOKEN_MATCHED_TEXT | 0.43 [0.42, 0.45] | N/A | 2.20 [2.06, 2.44] | 103.80 [103.25, 104.34] |
| LP_AUTO | VISUAL_PREFILL | 79.32 [75.62, 83.04] | 30.50 [27.61, 34.05] | 36.00 [33.14, 39.65] | 206.95 [201.25, 212.34] |
| LP_AUTO | TOKEN_MATCHED_TEXT | 0.44 [0.43, 0.47] | N/A | 2.05 [2.03, 2.06] | 101.14 [100.80, 101.77] |
| LP_G8 | VISUAL_PREFILL | 80.99 [79.00, 82.09] | 30.62 [28.80, 34.22] | 36.07 [34.24, 39.65] | 215.76 [206.60, 222.72] |
| LP_G8 | TOKEN_MATCHED_TEXT | 0.44 [0.42, 0.48] | N/A | 2.08 [2.07, 2.09] | 101.60 [101.29, 102.10] |
| CP256 | VISUAL_PREFILL | 55.24 [48.49, 68.24] | 36.50 [31.92, 45.54] | 42.18 [37.26, 50.87] | 191.56 [174.11, 214.80] |
| CP256 | TOKEN_MATCHED_TEXT | 0.41 [0.40, 0.41] | N/A | 2.87 [1.99, 3.85] | 101.49 [101.38, 101.58] |
| CP1024 | VISUAL_PREFILL | 58.96 [48.82, 71.26] | 41.58 [28.98, 50.31] | 46.98 [34.43, 55.77] | 193.04 [176.45, 201.93] |
| CP1024 | TOKEN_MATCHED_TEXT | 0.41 [0.39, 0.45] | N/A | 2.02 [1.98, 2.04] | 101.24 [100.69, 102.08] |
| LP_G16 | VISUAL_PREFILL | 73.68 [71.60, 76.25] | 29.56 [27.18, 32.58] | 35.35 [32.89, 38.67] | 202.76 [198.74, 209.35] |
| LP_G16 | TOKEN_MATCHED_TEXT | 0.40 [0.38, 0.42] | N/A | 2.05 [2.02, 2.10] | 102.85 [101.99, 104.29] |
| LP_G24 | VISUAL_PREFILL | 70.63 [66.88, 77.26] | 37.07 [30.61, 48.57] | 42.98 [36.30, 54.61] | 220.30 [195.62, 235.89] |
| LP_G24 | TOKEN_MATCHED_TEXT | 0.48 [0.43, 0.54] | N/A | 2.97 [2.76, 3.14] | 126.95 [111.96, 134.62] |
| LP_G48 | VISUAL_PREFILL | 71.02 [66.29, 75.60] | 32.78 [29.14, 39.38] | 39.25 [35.50, 45.53] | 229.46 [212.82, 241.07] |
| LP_G48 | TOKEN_MATCHED_TEXT | 0.45 [0.44, 0.45] | N/A | 2.78 [2.12, 3.12] | 127.97 [104.05, 145.18] |

The roughly 80ms host delivery/setup/executor interval and roughly 30ms vision
encoder duration occur once for the injected image, not once per LP stage.
These different clock intervals must not be added and equated to a GPU global
critical path. The host segment does not identify an exact IPC/serialization
routine; that attribution would require another profiler. The installed source
does show that the other DP can reach DeepEP prepare before visual input becomes
ready, making the first prepare span a peer-readiness diagnostic.

Visual and matched text have almost equal LM row counts. Actual source-SP,
destination-deduplicated logical traffic/fanout/skew is included above and as
per-rank matrices in final_metrics.json; it does not explain the observed first
producer floor merely by a 4-token difference. This is a **modality-common
input-producer/runtime residual in the tested harness**, not proof of an intrinsic
MLLM-specific wire-communication mechanism.

## Lossless vision-output control (secondary, three repeats)

This diagnostic reuses exact frozen CPU processor tensor/grid hashes and the
unchanged GPU vision+DeepStack output. It is not a production caching policy or
a clean hot-path optimization. Exact CPU input hashing and clone lookup remain
included in the measured request path. Cached and uncached Full have identical
prefill hidden/logits and unforced eight-token continuation in six comparisons;
cached Full/LP native correctness passes three repeats for both inputs.

| Arm | Input | Cached secondary TTFT ms | Peak VSEL ms | Peak ITL ms |
| --- | --- | --- | --- | --- |
| CP512 | VISUAL_PREFILL | 585.91 [574.68, 598.75] | 108.80 [105.28, 114.07] | 202.03 [198.35, 207.44] |
| CP512 | TOKEN_MATCHED_TEXT | 468.46 [465.80, 471.40] | 1.20 [-0.33, 2.83] | 94.69 [94.30, 95.37] |
| LP_G8 | VISUAL_PREFILL | 1057.32 [1044.64, 1066.31] | 122.74 [115.84, 133.96] | 216.00 [209.18, 227.52] |
| LP_G8 | TOKEN_MATCHED_TEXT | 930.99 [912.95, 943.57] | 20.96 [11.47, 32.38] | 114.65 [104.27, 126.48] |
| Arm | Input | Cached prepare span max ms | Cached expert total ms |
| --- | --- | --- | --- |
| CP512 | VISUAL_PREFILL | 4.15 [3.92, 4.32] | 83.31 [81.84, 85.15] |
| CP512 | TOKEN_MATCHED_TEXT | 1.01 [0.77, 1.38] | 79.38 [78.81, 79.72] |
| LP_G8 | VISUAL_PREFILL | 4.24 [4.16, 4.29] | 25.29 [25.13, 25.55] |
| LP_G8 | TOKEN_MATCHED_TEXT | 0.84 [0.68, 1.07] | 25.53 [25.12, 26.31] |
| Arm | Input | Cached admission→worker ms | Cached encoder-wrapper CUDA ms |
| --- | --- | --- | --- |
| CP512 | VISUAL_PREFILL | 88.39 [82.32, 92.62] | 0.12 [0.11, 0.12] |
| CP512 | TOKEN_MATCHED_TEXT | 0.42 [0.40, 0.46] | N/A |
| LP_G8 | VISUAL_PREFILL | 78.66 [63.35, 89.47] | 0.12 [0.11, 0.13] |
| LP_G8 | TOKEN_MATCHED_TEXT | 0.45 [0.42, 0.51] | N/A |

Use these actual diagnostic outcomes, including negative/no-improvement results,
not a hypothetical cache speedup. Cache-hit events are legacy wave/rank-only:
resident hits alone cannot prove the injection was cached. Exact numerical
identity and the injection-specific encoder-wrapper durations are independent
checks. Removing the vision encoder can shorten the local first-prepare span
while CPU hashing/setup still delays victim completion. These three-repeat
secondary runs are not pooled with or substituted for the ten-repeat primary.

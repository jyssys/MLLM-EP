# Measured native serving latency

Scope: CONTROLLED_NATIVE_LP_EP4_HARNESS. Values are mean [95% wave-bootstrap CI], ms. The token clock is engine completion, never external polling. CPU rendering occurs before model-ready arrival; GPU vision encoding is enabled. Model-ready TTFT and language-entry-to-first-token wall are separate. Resident decode is B8, original K8, fixed128 in the final lane.

## Frozen final (10 paired repeats)

| Arm | Input | n | TTFT | Mean ITL | P95 ITL | P99 ITL | Peak VSEL | Response peak ITL | LM-entry wall |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| DECODE_ONLY | DECODE_ONLY | 80 | N/A | 98.33 [96.33, 100.50] | 108.81 [106.53, 111.36] | 113.28 [110.70, 115.99] | 0.00 [0.00, 0.00] | 116.28 [113.57, 119.10] | N/A |
| FULL | VISUAL_PREFILL | 10 | 311.03 [300.76, 325.31] | 98.16 [95.08, 103.91] | 106.67 [103.51, 112.43] | 112.81 [106.73, 120.27] | 112.09 [109.06, 115.58] | 211.65 [205.03, 220.62] | 104.17 [99.79, 109.34] |
| FULL | TOKEN_MATCHED_TEXT | 10 | 197.62 [189.82, 209.34] | 97.23 [94.27, 102.83] | 106.41 [103.32, 112.13] | 114.74 [107.64, 122.69] | 7.19 [3.74, 11.51] | 103.53 [98.76, 109.60] | 98.98 [94.25, 105.29] |
| CP512 | VISUAL_PREFILL | 10 | 619.05 [584.06, 665.78] | 101.43 [95.69, 109.64] | 110.74 [104.70, 118.43] | 119.85 [112.40, 127.42] | 113.62 [109.42, 119.22] | 212.92 [202.33, 226.84] | 405.24 [381.67, 436.55] |
| CP512 | TOKEN_MATCHED_TEXT | 10 | 506.08 [476.12, 545.51] | 99.88 [94.21, 108.11] | 109.24 [103.53, 117.47] | 112.14 [105.49, 119.35] | 7.66 [4.83, 10.56] | 106.85 [100.45, 114.28] | 405.30 [381.06, 436.15] |
| LP_AUTO | VISUAL_PREFILL | 10 | 635.80 [615.63, 668.62] | 98.40 [95.28, 104.12] | 106.84 [103.63, 112.75] | 113.29 [108.93, 119.45] | 117.97 [113.96, 122.35] | 215.05 [209.78, 223.88] | 433.83 [418.24, 457.60] |
| LP_AUTO | TOKEN_MATCHED_TEXT | 10 | 538.03 [505.45, 582.49] | 100.04 [94.32, 108.30] | 109.40 [103.44, 117.99] | 113.20 [106.77, 120.55] | 15.54 [11.98, 19.59] | 114.96 [107.53, 123.96] | 437.34 [410.72, 472.95] |
| LP_G8 | VISUAL_PREFILL | 10 | 1069.27 [1036.02, 1128.59] | 99.03 [95.95, 104.77] | 108.99 [104.92, 115.14] | 118.24 [111.65, 125.75] | 121.33 [116.98, 127.51] | 217.77 [210.92, 229.23] | 863.87 [836.29, 912.40] |
| LP_G8 | TOKEN_MATCHED_TEXT | 10 | 956.68 [925.51, 1011.36] | 97.97 [94.76, 103.69] | 107.21 [104.07, 113.02] | 111.13 [107.00, 116.76] | 19.38 [16.55, 22.46] | 115.83 [111.90, 120.63] | 854.28 [827.04, 901.45] |

## Discovery (3 repeats/configuration, fixed96)

| Arm | Input | n | TTFT | Mean ITL | P95 ITL | P99 ITL | Peak VSEL | Response peak ITL | LM-entry wall |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| DECODE_ONLY | DECODE_ONLY | 12 | N/A | 94.44 [93.31, 95.95] | 102.06 [99.66, 104.73] | 109.93 [103.97, 117.79] | 0.00 [0.00, 0.00] | 110.05 [104.13, 117.85] | N/A |
| FULL | VISUAL_PREFILL | 3 | 299.99 [299.01, 300.90] | 94.20 [94.10, 94.40] | 102.58 [102.10, 103.11] | 207.79 [206.44, 208.68] | 113.80 [110.26, 116.25] | 207.95 [206.57, 208.85] | 96.37 [96.13, 96.57] |
| FULL | TOKEN_MATCHED_TEXT | 3 | 192.01 [186.07, 203.12] | 93.42 [93.20, 93.67] | 101.32 [98.70, 102.68] | 106.52 [103.66, 109.11] | 7.53 [1.23, 17.49] | 99.87 [94.36, 109.68] | 97.09 [91.61, 106.92] |
| CP256 | VISUAL_PREFILL | 3 | 966.50 [948.20, 991.33] | 94.58 [93.82, 95.83] | 102.31 [99.59, 104.70] | 212.54 [203.82, 229.11] | 118.55 [106.20, 137.22] | 212.71 [203.91, 229.41] | 759.25 [747.33, 771.23] |
| CP256 | TOKEN_MATCHED_TEXT | 3 | 843.07 [824.83, 857.12] | 93.16 [92.39, 94.17] | 102.27 [101.01, 103.46] | 108.58 [104.42, 113.31] | 11.18 [8.39, 12.67] | 102.99 [100.09, 104.96] | 749.13 [731.07, 763.22] |
| CP512 | VISUAL_PREFILL | 3 | 587.72 [579.21, 601.64] | 94.78 [94.23, 95.64] | 102.86 [102.45, 103.28] | 210.34 [208.19, 212.74] | 116.30 [111.61, 120.68] | 210.47 [208.31, 212.89] | 383.19 [375.13, 398.86] |
| CP512 | TOKEN_MATCHED_TEXT | 3 | 464.87 [460.21, 473.40] | 92.80 [92.11, 93.49] | 99.45 [93.35, 102.55] | 105.01 [104.71, 105.22] | 4.84 [1.87, 10.65] | 96.51 [93.35, 102.49] | 367.61 [366.52, 368.81] |
| CP1024 | VISUAL_PREFILL | 3 | 387.62 [385.08, 391.23] | 94.54 [93.91, 94.90] | 102.64 [102.32, 103.04] | 203.27 [200.99, 205.98] | 111.52 [110.39, 112.86] | 203.34 [201.10, 206.07] | 188.47 [187.99, 188.90] |
| CP1024 | TOKEN_MATCHED_TEXT | 3 | 278.61 [275.52, 280.94] | 93.08 [92.83, 93.22] | 102.05 [101.83, 102.40] | 107.31 [103.44, 114.44] | 2.51 [0.51, 5.67] | 94.45 [92.88, 96.76] | 184.86 [182.59, 187.50] |
| LP_AUTO | VISUAL_PREFILL | 3 | 621.17 [612.48, 632.63] | 95.24 [95.15, 95.41] | 106.13 [102.35, 111.05] | 219.03 [211.73, 227.84] | 127.36 [118.69, 137.21] | 219.20 [211.90, 227.92] | 412.36 [404.66, 419.61] |
| LP_AUTO | TOKEN_MATCHED_TEXT | 3 | 513.61 [512.12, 514.67] | 94.27 [93.75, 94.81] | 103.88 [102.17, 106.22] | 113.32 [113.08, 113.76] | 20.23 [18.28, 22.42] | 112.25 [111.35, 113.95] | 415.85 [413.14, 421.10] |
| LP_G4 | VISUAL_PREFILL | 3 | 610.61 [597.80, 629.54] | 94.66 [94.41, 95.04] | 102.24 [101.76, 102.56] | 211.78 [203.51, 220.02] | 120.10 [110.32, 128.70] | 211.95 [203.53, 220.23] | 413.45 [407.79, 423.65] |
| LP_G4 | TOKEN_MATCHED_TEXT | 3 | 521.83 [503.72, 557.81] | 93.87 [93.77, 93.96] | 102.39 [102.06, 102.65] | 112.56 [102.78, 124.15] | 20.18 [7.85, 33.49] | 112.58 [102.68, 124.29] | 421.02 [399.14, 453.69] |
| LP_G8 | VISUAL_PREFILL | 3 | 1094.47 [1044.53, 1160.49] | 102.90 [98.81, 105.21] | 126.79 [107.16, 140.56] | 217.45 [214.71, 221.03] | 119.95 [114.65, 129.64] | 217.65 [214.94, 221.20] | 894.74 [847.36, 958.26] |
| LP_G8 | TOKEN_MATCHED_TEXT | 3 | 963.66 [946.64, 981.67] | 101.28 [99.41, 102.34] | 111.81 [107.06, 117.18] | 144.77 [144.24, 145.74] | 15.81 [12.37, 17.66] | 115.63 [108.89, 119.08] | 864.24 [848.52, 881.80] |
| LP_G16 | VISUAL_PREFILL | 3 | 1936.71 [1842.59, 2005.03] | 101.91 [96.40, 106.47] | 117.90 [107.25, 126.91] | 237.02 [212.01, 254.74] | 139.41 [111.89, 163.23] | 237.09 [212.15, 254.77] | 1732.70 [1656.21, 1808.70] |
| LP_G16 | TOKEN_MATCHED_TEXT | 3 | 1808.12 [1734.86, 1905.99] | 100.52 [98.39, 103.79] | 112.70 [107.88, 119.60] | 145.77 [131.19, 156.03] | 30.59 [12.91, 59.20] | 126.10 [112.26, 150.75] | 1710.05 [1639.34, 1805.28] |
| LP_G24 | VISUAL_PREFILL | 3 | 2836.31 [2761.92, 2873.95] | 105.31 [102.62, 108.08] | 125.50 [119.26, 135.18] | 225.65 [217.91, 236.31] | 128.34 [116.91, 136.72] | 226.03 [218.17, 237.01] | 2625.34 [2556.82, 2660.81] |
| LP_G24 | TOKEN_MATCHED_TEXT | 3 | 2943.05 [2910.74, 2994.37] | 107.66 [105.68, 109.93] | 137.32 [129.80, 142.97] | 158.30 [157.18, 159.57] | 63.49 [58.42, 68.55] | 159.02 [158.05, 159.90] | 2830.79 [2782.73, 2889.09] |
| LP_G48 | VISUAL_PREFILL | 3 | 5196.80 [5162.24, 5220.61] | 101.29 [100.69, 101.65] | 105.96 [105.30, 106.47] | 214.63 [204.96, 228.16] | 120.71 [110.80, 134.23] | 214.78 [205.09, 228.28] | 5007.01 [4978.57, 5021.25] |
| LP_G48 | TOKEN_MATCHED_TEXT | 3 | 5067.68 [5053.57, 5079.50] | 99.87 [99.55, 100.10] | 105.06 [104.60, 105.61] | 110.64 [108.82, 113.57] | 15.40 [14.30, 16.24] | 109.36 [108.87, 109.64] | 4971.62 [4956.69, 4983.71] |

LP_AUTO follows the pinned upstream rule and is G4 for both inputs. G4 is not an independent algorithm. Best LP was frozen as **LP_G8** from discovery, before final repeats. The SLO proxy is peak ITL / paired decode-only median <=1.5, not a production SLO. Peak statistics include the whole injection response window: finer groups have a longer exposure window. These are repeated measurements of a fixed bounded pool, not claims across unseen workloads.

## First producer iteration versus later stages (secondary)

| Arm | Input | Phase | Wave n | Wave peak ITL | Paired wave peak VSEL |
| --- | --- | --- | --- | --- | --- |
| FULL | VISUAL_PREFILL | FIRST_PRODUCER_ITERATION | 10 | 211.65 [205.03, 220.62] | 112.09 [109.06, 115.58] |
| FULL | TOKEN_MATCHED_TEXT | FIRST_PRODUCER_ITERATION | 10 | 101.90 [97.09, 108.32] | 2.05 [-4.31, 8.36] |
| CP512 | VISUAL_PREFILL | FIRST_PRODUCER_ITERATION | 10 | 212.92 [202.33, 226.84] | 113.62 [109.42, 119.22] |
| CP512 | VISUAL_PREFILL | LATER_PREFILL_STAGE | 10 | 102.43 [96.26, 110.31] | 4.15 [2.93, 5.78] |
| CP512 | TOKEN_MATCHED_TEXT | FIRST_PRODUCER_ITERATION | 10 | 103.43 [97.15, 110.53] | 3.91 [-0.07, 8.05] |
| CP512 | TOKEN_MATCHED_TEXT | LATER_PREFILL_STAGE | 10 | 105.07 [98.59, 112.79] | 5.72 [3.01, 8.46] |
| LP_AUTO | VISUAL_PREFILL | FIRST_PRODUCER_ITERATION | 10 | 215.05 [209.78, 223.88] | 117.97 [113.96, 122.35] |
| LP_AUTO | VISUAL_PREFILL | LATER_PREFILL_STAGE | 10 | 111.50 [106.96, 117.43] | 13.86 [11.78, 16.40] |
| LP_AUTO | TOKEN_MATCHED_TEXT | FIRST_PRODUCER_ITERATION | 10 | 110.00 [103.43, 119.25] | 10.47 [8.26, 12.52] |
| LP_AUTO | TOKEN_MATCHED_TEXT | LATER_PREFILL_STAGE | 10 | 113.46 [105.57, 122.92] | 14.05 [10.44, 18.41] |
| LP_G8 | VISUAL_PREFILL | FIRST_PRODUCER_ITERATION | 10 | 217.77 [210.92, 229.23] | 121.33 [116.98, 127.51] |
| LP_G8 | VISUAL_PREFILL | LATER_PREFILL_STAGE | 10 | 113.37 [109.15, 118.84] | 17.76 [15.19, 19.94] |
| LP_G8 | TOKEN_MATCHED_TEXT | FIRST_PRODUCER_ITERATION | 10 | 107.04 [103.38, 113.29] | 8.50 [5.69, 11.13] |
| LP_G8 | TOKEN_MATCHED_TEXT | LATER_PREFILL_STAGE | 10 | 115.11 [111.08, 120.02] | 19.16 [16.13, 22.40] |

Actual DP0 scheduler/engine completion alignment residual is at most **0.194 ms**. Peer ITL uses the same forced token position. This separation neither removes the first burst from the primary metric nor changes the frozen winner. Later-stage peak exposure differs with group count; layers/stages/residents are not independent bootstrap samples. These host clocks are not GPU CWT.

## Interpretation and measurement boundary

Native LP is genuinely advancing hidden/residual state across serving iterations,
not replaying layers or summing post-hoc timings. The full resident decoder runs
48 layers and emits a real token at every stage. AUTO is **G4**, following the
actual pinned upstream <=2048-token rule; the earlier expected G8 was wrong.

All visual arms retain a first-iteration peak. Increasing to G16/G24/G48 does
not monotonically reduce this producer floor and increases TTFT/stage exposure.
The secondary later-stage analysis shows that CP512 and LP can already keep
ordinary later iterations reasonably close to decode-only. Do not delete the
first burst from the primary latency statistic to manufacture a passing SLO.

TTFT starts at model-ready injection (CPU image rendering happened while both
DPs were paused). GPU vision encoding remains enabled. The reported prefill wall
is **LM-entry to first completed token**, including interleaved resident decode,
final normalization/LM head/sampling/output. It is not isolated prefill GPU time.
The injected request emits one token in timing, so its measured completion E2E
equals this model-ready TTFT; resident fixed128 makespan is in raw worker output.

Resident B8 contains four frozen images duplicated as independent requests on
the two DPs: global eight requests, **four unique images**, not eight independent
input samples. The fixed native continuation controls token/position identity;
it is not a natural task-quality evaluation. One frozen injected visual and one
matched text prompt bound the generalization domain. CIs describe measurement
repeat uncertainty, not unseen-image accuracy or all serving workloads.

Ten paired repeats span five independent engine restarts. Restart-cluster CIs
are also published in final_metrics.json; five clusters remain a small sample.
P99 over a long fixed128 run can dilute a single input burst, hence response peak
ITL and paired VSEL remain primary. No fastest-run filtering is used.

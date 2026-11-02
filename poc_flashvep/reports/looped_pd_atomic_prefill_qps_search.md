# Atomic Prefill PD: Final QPS Search

Compiled at 2026-11-02T01:28:43.086700+00:00. Elapsed time: 21.426 hours.

Branch: `flashvep/looped-pd-atomic-prefill-qps-crossover`.

Compilation source commit: `147c4dc48215dd82cc85f77b044ff27325ea84fc`.

Each measured run records its own source archive, source commit, and configuration hash.
The final publication commit can differ from the measurement commit.
Use `git log -1 --format=%H` on this branch to obtain the publication SHA.

Actual analysis-tree SHA256: `a392ed79e603c437d00c3575d39166f104f681e2144fe757f6ef0524e7bc5c18`.

The compilation HEAD can precede uncommitted analysis changes.
See `state/analysis_source_manifest.json` for the exact file hashes.

Evidence root: `looped_pd_atomic_prefill_qps_crossover/`.

## Completion ledger


| Configuration status | Count |
| --- | --- |
| COMPLETE | 766 |
| FAILED | 15 |
| NOT_RUN | 1 |
| NOT_RUN_BLOCKED_CORRECTNESS | 55 |
| NOT_RUN_DUPLICATE_DIAGNOSTIC | 1 |
| NOT_RUN_EXTENSION_PRIORITY_TIME_BUDGET | 53 |
| NOT_RUN_FAILED_STRONG_P_SERVICE_PREREQUISITE | 6 |
| NOT_RUN_REPEATED_PROFILE_STALL_TIME_BUDGET | 4 |
| NOT_RUN_SUPERSEDED_AR_EXECUTION_FAMILY | 5 |
| NOT_RUN_SUPERSEDED_METADATA | 2 |
| NOT_RUN_UNSUPPORTED_OR_IGNORED_HUGINN_ASYNC | 11 |
| NOT_RUN_USER_11H_DEADLINE | 31 |

`configuration_matrix.csv` includes every actual configuration.
`requested_search_grid.csv` retains every requested input/output/chunk cell.
A configuration file alone is not an executed result.

## Frozen SLOs


| Cell | TTFT limit ms | Request-average TPOT limit ms | Secondary token ITL limit ms | Calibration requests |
| --- | --- | --- | --- | --- |
| qwen8_L3900 | 359.1716 | 20.2506 | 23.9847 | 16 |
| qwen8_L8000 | 679.7450 | 20.6578 | 24.4493 | 16 |
| ouro_L4096 | 1112.9926 | 53.7070 | 65.0829 | 16 |
| ouro_L8192 | 2665.6048 | 65.0846 | 83.9214 | 16 |
| huginn_L3500 | 3199.7863 | 192.8418 | 218.5536 | 16 |
| huginn_L3900 | 4060.8008 | 194.3331 | 238.9032 | 16 |
| ouro_L16384 | 4350.3904 | 90.6516 | 127.9981 | 8 |

Thresholds were frozen before the corresponding held-out search.
Optional 16K adds a new calibrated cell without changing existing thresholds.
The primary SLO uses request TTFT and request-average TPOT.

Token ITL tails remain separate.

## Independent final runs


| Run | Input/output | Arm/C | Qualified clean | Target QPS | Offered/completed | TTFT p95 ms | TPOT p95 ms | ITL p99 ms | E2E p95 ms | SLO attainment | SLO goodput req/s |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ext_final_huginn_l3500_c2048_r1_s0_v1 | 3500/32 | PD_G1/2048 | True | 0.9689 | 128/128 | 99444.7078 | 109.4074 | 143.8288 | 102761.4235 | 0.0234 | 0.0130 |
| ext_final_huginn_l3500_c2048_r1_s1_v1 | 3500/32 | PD_G1/2048 | True | 0.9689 | 128/128 | 40474.6839 | 121.6762 | 149.6356 | 44123.3424 | 0.0703 | 0.0498 |
| ext_final_huginn_l3500_c2048_r1_s2_v1 | 3500/32 | PD_G1/2048 | True | 0.9689 | 128/128 | 54188.6570 | 119.7430 | 139.4343 | 57801.3047 | 0.0234 | 0.0159 |
| ext_final_huginn_l3500_cfull_r1_s0_v1 | 3500/32 | PD_G1/3500 | True | 0.9689 | 128/128 | 55906.2436 | 135.6001 | 316.1604 | 59865.2229 | 0.0234 | 0.0162 |
| ext_final_huginn_l3500_cfull_r1_s1_v1 | 3500/32 | PD_G1/3500 | True | 0.9689 | 128/128 | 42604.9095 | 119.4575 | 152.6890 | 46172.4831 | 0.0938 | 0.0655 |
| ext_final_huginn_l3500_cfull_r1_s2_v1 | 3500/32 | PD_G1/3500 | True | 0.9689 | 128/128 | 49421.7841 | 122.1581 | 154.1491 | 52994.0339 | 0.0391 | 0.0273 |
| ext_final_ouro_l16384_c4096_r0_s0_v1 | 16384/32 | PD_G1/4096 | True | 0.9927 | 128/128 | 59670.4341 | 49.7927 | 71.6746 | 61030.9731 | 0.0234 | 0.0164 |
| ext_final_ouro_l16384_c4096_r1_s0_v1 | 16384/32 | PD_G1/4096 | True | 1.2408 | 128/128 | 83535.8524 | 48.3344 | 64.4375 | 84879.7901 | 0.0234 | 0.0164 |
| ext_final_ouro_l16384_c4096_r1_s1_v1 | 16384/32 | PD_G1/4096 | True | 1.2408 | 128/128 | 76969.3397 | 49.1188 | 72.3365 | 78322.3856 | 0.0156 | 0.0107 |
| ext_final_ouro_l16384_c4096_r1_s2_v1 | 16384/32 | PD_G1/4096 | True | 1.2408 | 128/128 | 86586.1725 | 46.0121 | 69.4294 | 87926.6049 | 0.0156 | 0.0105 |
| ext_final_ouro_l16384_c4096_r2_s0_v1 | 16384/32 | PD_G1/4096 | True | 1.4890 | 128/128 | 98672.7075 | 52.3937 | 81.8545 | 100031.2830 | 0.0234 | 0.0163 |
| ext_final_ouro_l16384_cfull_r0_s0_v1 | 16384/32 | PD_G1/16384 | True | 0.9927 | 128/128 | 54328.0723 | 53.6031 | 72.0045 | 55763.9636 | 0.0391 | 0.0283 |
| ext_final_ouro_l16384_cfull_r1_s0_v1 | 16384/32 | PD_G1/16384 | True | 1.2408 | 128/128 | 140990.3289 | 48.1039 | 73.0355 | 142425.9317 | 0.0078 | 0.0041 |
| ext_final_ouro_l16384_cfull_r1_s1_v1 | 16384/32 | PD_G1/16384 | True | 1.2408 | 128/128 | 128841.5018 | 50.9762 | 98.4808 | 130269.5778 | 0.0000 | 0.0000 |
| ext_final_ouro_l16384_cfull_r1_s2_v1 | 16384/32 | PD_G1/16384 | True | 1.2408 | 128/128 | 73137.5772 | 53.7100 | 85.4062 | 74574.6232 | 0.0156 | 0.0113 |
| ext_final_ouro_l16384_cfull_r2_s0_v1 | 16384/32 | PD_G1/16384 | True | 1.4890 | 128/128 | 128505.8049 | 50.0908 | 77.8771 | 129954.3760 | 0.0078 | 0.0047 |
| ext_final_qwen8_l8000_c4096_r1_s0_v1 | 8000/200 | PD_G1/4096 | True | 3.6136 | 160/160 | 6000.8841 | 56.8058 | 196.0436 | 15196.2027 | 0.0437 | 0.1290 |
| ext_final_qwen8_l8000_c4096_r1_s1_v1 | 8000/200 | PD_G1/4096 | True | 3.6136 | 160/160 | 3862.4882 | 42.7925 | 162.6406 | 12283.8988 | 0.1812 | 0.5257 |
| ext_final_qwen8_l8000_c4096_r1_s2_v1 | 8000/200 | PD_G1/4096 | True | 3.6136 | 160/160 | 4966.8746 | 42.2199 | 165.5929 | 13213.4871 | 0.1625 | 0.4871 |
| ext_final_qwen8_l8000_cfull_r1_s0_v1 | 8000/200 | PD_G1/8000 | True | 3.6136 | 160/160 | 3708.4276 | 43.0368 | 241.4913 | 12215.7859 | 0.0813 | 0.2553 |
| ext_final_qwen8_l8000_cfull_r1_s1_v1 | 8000/200 | PD_G1/8000 | True | 3.6136 | 160/160 | 3780.5255 | 43.1932 | 212.6340 | 12005.0289 | 0.2188 | 0.6357 |
| final_ouro_l8192_o32_dp_cfull_r0_s0_v1 | 8192/32 | DP/8192 | True | 2.1905 | 128/128 | 9502.7726 | 323.1252 | 1607.8641 | 15151.3773 | 0.1094 | 0.2370 |
| final_ouro_l8192_o32_dp_cfull_r0_s1_v1 | 8192/32 | DP/8192 | True | 2.1905 | 128/128 | 2167.4408 | 185.8675 | 1298.1438 | 7000.7086 | 0.1484 | 0.3466 |
| final_ouro_l8192_o32_dp_cfull_r0_s2_v1 | 8192/32 | DP/8192 | True | 2.1905 | 128/128 | 2568.0239 | 168.1151 | 1064.8496 | 6260.4687 | 0.2891 | 0.6180 |
| final_ouro_l8192_o32_dp_cfull_r1_s0_v1 | 8192/32 | DP/8192 | True | 2.7381 | 128/128 | 10426.7903 | 237.5160 | 1611.4367 | 15876.4536 | 0.0156 | 0.0356 |
| final_ouro_l8192_o32_dp_cfull_r1_s1_v1 | 8192/32 | DP/8192 | True | 2.7381 | 130/130 | 5952.6001 | 191.6278 | 1995.4408 | 11184.5707 | 0.0000 | 0.0000 |
| final_ouro_l8192_o32_dp_cfull_r1_s2_v1 | 8192/32 | DP/8192 | True | 2.7381 | 128/128 | 4001.0649 | 212.5625 | 1481.8543 | 8198.0841 | 0.0312 | 0.0786 |
| final_ouro_l8192_o32_dp_cfull_r2_s0_v1 | 8192/32 | DP/8192 | True | 3.2858 | 145/145 | 13866.1278 | 243.3191 | 2370.9921 | 18863.8520 | 0.0000 | 0.0000 |
| final_ouro_l8192_o32_dp_cfull_r2_s1_v1 | 8192/32 | DP/8192 | True | 3.2858 | 159/159 | 16200.6687 | 238.4944 | 2987.1611 | 20278.2463 | 0.0000 | 0.0000 |
| final_ouro_l8192_o32_dp_cfull_r2_s2_v1 | 8192/32 | DP/8192 | True | 3.2858 | 144/144 | 12763.9636 | 290.3920 | 3283.8273 | 16394.5322 | 0.0000 | 0.0000 |
| final_ouro_l8192_o32_pd_g1_c4096_r0_s0_v1 | 8192/32 | PD_G1/4096 | True | 2.1905 | 128/128 | 31784.5980 | 47.6638 | 72.4814 | 33093.1046 | 0.0156 | 0.0221 |
| final_ouro_l8192_o32_pd_g1_c4096_r0_s1_v1 | 8192/32 | PD_G1/4096 | True | 2.1905 | 128/128 | 35877.9715 | 46.7688 | 79.5023 | 36982.3069 | 0.0000 | 0.0000 |
| final_ouro_l8192_o32_pd_g1_c4096_r0_s2_v1 | 8192/32 | PD_G1/4096 | True | 2.1905 | 128/128 | 64404.6060 | 43.3275 | 105.0509 | 65490.2052 | 0.0000 | 0.0000 |
| final_ouro_l8192_o32_pd_g1_c4096_r1_s0_v1 | 8192/32 | PD_G1/4096 | True | 2.7381 | 128/128 | 76066.6881 | 44.5866 | 88.8134 | 77149.8651 | 0.0000 | 0.0000 |
| final_ouro_l8192_o32_pd_g1_c4096_r1_s1_v1 | 8192/32 | PD_G1/4096 | True | 2.7381 | 130/130 | 43254.3105 | 54.9854 | 175.8221 | 44420.2610 | 0.0000 | 0.0000 |
| final_ouro_l8192_o32_pd_g1_c4096_r1_s2_v1 | 8192/32 | PD_G1/4096 | True | 2.7381 | 128/128 | 35672.4435 | 100.0922 | 126.5224 | 36742.5931 | 0.0000 | 0.0000 |
| final_ouro_l8192_o32_pd_g1_c4096_r2_s0_v1 | 8192/32 | PD_G1/4096 | True | 3.2858 | 145/145 | 52136.7683 | 57.2442 | 113.8883 | 53267.8691 | 0.0000 | 0.0000 |
| final_ouro_l8192_o32_pd_g1_c4096_r2_s1_v1 | 8192/32 | PD_G1/4096 | True | 3.2858 | 159/159 | 62235.9813 | 46.0880 | 151.9438 | 63303.4807 | 0.0000 | 0.0000 |
| final_ouro_l8192_o32_pd_g1_c4096_r2_s2_v1 | 8192/32 | PD_G1/4096 | True | 3.2858 | 144/144 | 52942.2207 | 58.7175 | 128.3872 | 54574.5039 | 0.0000 | 0.0000 |
| final_ouro_l8192_o32_pd_g1_cfull_r0_s0_v1 | 8192/32 | PD_G1/8192 | True | 2.1905 | 128/128 | 31913.7389 | 59.4610 | 150.8148 | 33122.5354 | 0.0000 | 0.0000 |
| final_ouro_l8192_o32_pd_g1_cfull_r0_s1_v1 | 8192/32 | PD_G1/8192 | True | 2.1905 | 128/128 | 39115.4031 | 117.5744 | 304.0156 | 40390.2155 | 0.0000 | 0.0000 |
| final_ouro_l8192_o32_pd_g1_cfull_r0_s2_v1 | 8192/32 | PD_G1/8192 | True | 2.1905 | 128/128 | 30537.3013 | 61.8628 | 288.9874 | 32114.0418 | 0.0000 | 0.0000 |
| final_ouro_l8192_o32_pd_g1_cfull_r1_s0_v1 | 8192/32 | PD_G1/8192 | True | 2.7381 | 128/128 | 38192.4257 | 52.0791 | 138.8182 | 39294.0670 | 0.0156 | 0.0234 |
| final_ouro_l8192_o32_pd_g1_cfull_r1_s1_v1 | 8192/32 | PD_G1/8192 | True | 2.7381 | 130/130 | 45999.9517 | 84.4718 | 484.2095 | 47314.0310 | 0.0000 | 0.0000 |
| final_ouro_l8192_o32_pd_g1_cfull_r1_s2_v1 | 8192/32 | PD_G1/8192 | True | 2.7381 | 128/128 | 40645.7757 | 66.4093 | 143.1896 | 41719.3037 | 0.0000 | 0.0000 |
| final_ouro_l8192_o32_pd_g1_cfull_r2_s0_v1 | 8192/32 | PD_G1/8192 | True | 3.2858 | 145/145 | 63667.8465 | 55.5127 | 124.0562 | 64926.0748 | 0.0000 | 0.0000 |
| final_ouro_l8192_o32_pd_g1_cfull_r2_s1_v1 | 8192/32 | PD_G1/8192 | True | 3.2858 | 159/159 | 73753.2033 | 118.8146 | 385.4624 | 75378.3453 | 0.0000 | 0.0000 |
| final_ouro_l8192_o32_pd_g1_cfull_r2_s2_v1 | 8192/32 | PD_G1/8192 | True | 3.2858 | 144/144 | 58437.4055 | 55.6060 | 188.4060 | 59925.9736 | 0.0000 | 0.0000 |

All mean/p50/p95/p99 values remain in `serving_runs.csv`.
That file separates target, realized, and completed rates.
It includes all offered requests in the SLO denominator.

Timeouts and missing requests cannot increase goodput.
An unfinished run can retain completed-subset latency diagnostics.
Without a complete measured window, its throughput and goodput remain NOT_MEASURED.

## P/D evidence


| Run | Class | P rate | P finite reference | P kernel busy | D finite reserve | Binding D KV refusal fraction | Raw D KV refusal fraction | D credit clear | D TPOT margin | Staggered D pass |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ext_final_huginn_l3500_c2048_r1_s0_v1 | MIXED_OR_INCONCLUSIVE | 0.5585 | 0.8425 | NOT_MEASURED | True | 0.0000 | 0.0119 | True | True | True |
| ext_final_huginn_l3500_c2048_r1_s1_v1 | MIXED_OR_INCONCLUSIVE | 0.7251 | 0.8425 | NOT_MEASURED | True | 0.0000 | 0.6419 | True | True | True |
| ext_final_huginn_l3500_c2048_r1_s2_v1 | MIXED_OR_INCONCLUSIVE | 0.6898 | 0.8425 | NOT_MEASURED | True | 0.0000 | 0.4986 | True | True | True |
| ext_final_huginn_l3500_cfull_r1_s0_v1 | MIXED_OR_INCONCLUSIVE | 0.7130 | 0.8425 | NOT_MEASURED | True | 0.0030 | 0.9061 | True | True | False |
| ext_final_huginn_l3500_cfull_r1_s1_v1 | MIXED_OR_INCONCLUSIVE | 0.7047 | 0.8425 | NOT_MEASURED | True | 0.0000 | 0.5443 | True | True | False |
| ext_final_huginn_l3500_cfull_r1_s2_v1 | MIXED_OR_INCONCLUSIVE | 0.7232 | 0.8425 | NOT_MEASURED | True | 0.0000 | 0.5978 | True | True | False |
| ext_final_ouro_l16384_c4096_r0_s0_v1 | MIXED_OR_INCONCLUSIVE | 0.7062 | 0.9927 | NOT_MEASURED | True | 0.0000 | 0.0000 | True | True | False |
| ext_final_ouro_l16384_c4096_r1_s0_v1 | MIXED_OR_INCONCLUSIVE | 0.7001 | 0.9927 | NOT_MEASURED | True | 0.0000 | 0.0000 | True | True | False |
| ext_final_ouro_l16384_c4096_r1_s1_v1 | MIXED_OR_INCONCLUSIVE | 0.6670 | 0.9927 | NOT_MEASURED | True | 0.0000 | 0.0000 | True | True | False |
| ext_final_ouro_l16384_c4096_r1_s2_v1 | MIXED_OR_INCONCLUSIVE | 0.6697 | 0.9927 | NOT_MEASURED | True | 0.0000 | 0.0000 | True | True | False |
| ext_final_ouro_l16384_c4096_r2_s0_v1 | MIXED_OR_INCONCLUSIVE | 0.6940 | 0.9927 | NOT_MEASURED | True | 0.0000 | 0.0000 | True | True | False |
| ext_final_ouro_l16384_cfull_r0_s0_v1 | MIXED_OR_INCONCLUSIVE | 0.7305 | 0.9927 | NOT_MEASURED | True | 0.0000 | 0.3609 | True | True | False |
| ext_final_ouro_l16384_cfull_r1_s0_v1 | MIXED_OR_INCONCLUSIVE | 0.5175 | 0.9927 | NOT_MEASURED | True | 0.0000 | 0.2366 | True | True | True |
| ext_final_ouro_l16384_cfull_r1_s1_v1 | MIXED_OR_INCONCLUSIVE | 0.5493 | 0.9927 | NOT_MEASURED | True | 0.0000 | 0.2691 | True | True | True |
| ext_final_ouro_l16384_cfull_r1_s2_v1 | MIXED_OR_INCONCLUSIVE | 0.7267 | 0.9927 | NOT_MEASURED | True | 0.0000 | 0.3540 | True | True | True |
| ext_final_ouro_l16384_cfull_r2_s0_v1 | MIXED_OR_INCONCLUSIVE | 0.5479 | 0.9927 | NOT_MEASURED | True | 0.0000 | 0.2569 | True | True | True |
| ext_final_qwen8_l8000_c4096_r1_s0_v1 | DECODE_BOUND | 3.5384 | 3.6136 | 0.9364 | True | 0.0000 | NOT_MEASURED | True | False | False |
| ext_final_qwen8_l8000_c4096_r1_s1_v1 | KV_ADMISSION_BOUND | 3.5628 | 3.6136 | NOT_MEASURED | True | 0.0000 | NOT_MEASURED | False | True | False |
| ext_final_qwen8_l8000_c4096_r1_s2_v1 | MIXED_OR_INCONCLUSIVE | 3.3054 | 3.6136 | NOT_MEASURED | True | 0.0000 | NOT_MEASURED | True | True | True |
| ext_final_qwen8_l8000_cfull_r1_s0_v1 | KV_ADMISSION_BOUND | 3.6349 | 3.6136 | NOT_MEASURED | True | 0.0000 | NOT_MEASURED | False | True | False |
| ext_final_qwen8_l8000_cfull_r1_s1_v1 | KV_ADMISSION_BOUND | 3.5628 | 3.6136 | NOT_MEASURED | True | 0.0000 | NOT_MEASURED | False | True | False |
| final_ouro_l8192_o32_pd_g1_c4096_r0_s0_v1 | KV_ADMISSION_BOUND | 1.4940 | 2.1905 | NOT_MEASURED | True | 0.4277 | 0.4843 | False | True | False |
| final_ouro_l8192_o32_pd_g1_c4096_r0_s1_v1 | KV_ADMISSION_BOUND | 1.5003 | 2.1905 | NOT_MEASURED | True | 0.3750 | 0.4236 | False | True | False |
| final_ouro_l8192_o32_pd_g1_c4096_r0_s2_v1 | KV_ADMISSION_BOUND | 1.0606 | 2.1905 | NOT_MEASURED | True | 0.3121 | 0.3439 | False | True | False |
| final_ouro_l8192_o32_pd_g1_c4096_r1_s0_v1 | KV_ADMISSION_BOUND | 1.1205 | 2.1905 | NOT_MEASURED | True | 0.2756 | 0.3228 | False | True | False |
| final_ouro_l8192_o32_pd_g1_c4096_r1_s1_v1 | KV_ADMISSION_BOUND | 1.4545 | 2.1905 | NOT_MEASURED | True | 0.4706 | 0.4958 | False | True | False |
| final_ouro_l8192_o32_pd_g1_c4096_r1_s2_v1 | KV_ADMISSION_BOUND | 1.5151 | 2.1905 | NOT_MEASURED | True | 0.4720 | 0.5120 | False | False | False |
| final_ouro_l8192_o32_pd_g1_c4096_r2_s0_v1 | KV_ADMISSION_BOUND | 1.4955 | 2.1905 | NOT_MEASURED | True | 0.4474 | 0.5175 | False | True | False |
| final_ouro_l8192_o32_pd_g1_c4096_r2_s1_v1 | KV_ADMISSION_BOUND | 1.4386 | 2.1905 | NOT_MEASURED | True | 0.4615 | 0.5043 | False | True | False |
| final_ouro_l8192_o32_pd_g1_c4096_r2_s2_v1 | KV_ADMISSION_BOUND | 1.4535 | 2.1905 | NOT_MEASURED | True | 0.4370 | 0.4622 | False | False | False |
| final_ouro_l8192_o32_pd_g1_cfull_r0_s0_v1 | KV_ADMISSION_BOUND | 1.4691 | 2.1905 | NOT_MEASURED | True | 0.3291 | 0.5063 | False | False | False |
| final_ouro_l8192_o32_pd_g1_cfull_r0_s1_v1 | KV_ADMISSION_BOUND | 1.3395 | 2.1905 | NOT_MEASURED | True | 0.3259 | 0.4963 | False | False | False |
| final_ouro_l8192_o32_pd_g1_cfull_r0_s2_v1 | KV_ADMISSION_BOUND | 1.4646 | 2.1905 | NOT_MEASURED | True | 0.3269 | 0.5064 | False | False | False |
| final_ouro_l8192_o32_pd_g1_cfull_r1_s0_v1 | KV_ADMISSION_BOUND | 1.4940 | 2.1905 | NOT_MEASURED | True | 0.3333 | 0.5476 | False | True | False |
| final_ouro_l8192_o32_pd_g1_cfull_r1_s1_v1 | KV_ADMISSION_BOUND | 1.4214 | 2.1905 | 0.4901 | True | 0.3362 | 0.5000 | False | False | False |
| final_ouro_l8192_o32_pd_g1_cfull_r1_s2_v1 | KV_ADMISSION_BOUND | 1.3889 | 2.1905 | NOT_MEASURED | True | 0.3810 | 0.5476 | False | False | False |
| final_ouro_l8192_o32_pd_g1_cfull_r2_s0_v1 | KV_ADMISSION_BOUND | 1.2961 | 2.1905 | NOT_MEASURED | True | 0.3167 | 0.5417 | False | True | False |
| final_ouro_l8192_o32_pd_g1_cfull_r2_s1_v1 | KV_ADMISSION_BOUND | 1.4386 | 2.1905 | NOT_MEASURED | True | 0.3525 | 0.5164 | False | False | False |
| final_ouro_l8192_o32_pd_g1_cfull_r2_s2_v1 | KV_ADMISSION_BOUND | 1.3874 | 2.1905 | NOT_MEASURED | True | 0.3220 | 0.5085 | False | True | False |

The complete per-run gate table is `bottleneck_classification.csv`.
P saturation requires actual kernel activity, service proximity, and growing ready backlog.
D reserve requires compute, KV credit, readiness, and frozen SLO margin.

NVML activity alone proves neither saturation nor spare compute.

## Observed host submission and scheduling


| Run | P ModelRunner submissions | P submission host wall s | P process CPU s during submission spans | D schedule calls | D scheduler host wall s |
| --- | --- | --- | --- | --- | --- |
| ext_final_huginn_l3500_c2048_r1_s0_v1 | 260 | 173.5841 | 347.7210 | 1024873 | 6.4183 |
| ext_final_huginn_l3500_cfull_r1_s0_v1 | 130 | 159.6544 | 319.2351 | 518425 | 6.1495 |
| ext_final_ouro_l16384_c4096_r1_s0_v1 | 478 | 149.2973 | 316.6101 | 2352033 | 8.9501 |
| ext_final_ouro_l16384_cfull_r1_s0_v1 | 87 | 173.5671 | 347.5377 | 17457188 | 56.8538 |
| final_ouro_l8192_o32_pd_g1_c4096_r1_s0_v1 | 71 | 92.6510 | 185.4397 | 6219493 | 22.3119 |
| final_ouro_l8192_o32_pd_g1_cfull_r1_s0_v1 | 62 | 62.8742 | 125.7644 | 3435867 | 12.4831 |

These counters include warm-up. They are not measured-request-only totals.
A ModelRunner submission can contain many GPU kernels.
Process CPU spans include all process threads, including background transport work.

CPU seconds can exceed wall seconds. Host submission spans can overlap GPU computation.
These counters do not identify an exclusive CPU fraction or prove a specific CPU critical path.
The raw `host_execution_*.json` files preserve every source counter.

Explicit transport observers remain diagnostics, not clean Serving timings.

## Profiling and transport


| Profile | Matched clean run | Wall overhead fraction | Eligible transfer to clean | P recorded kernel fraction | D graph-active fraction |
| --- | --- | --- | --- | --- | --- |
| ext_candidate_profile_ouro_l16384_v1 | ext_ouro_l16384_p2_drain1_credit32g_c16384_f1p25_pilot24_v1 | 0.0118 | True | 0.5648 | 0.3175 |
| ext_candidate_profile_ouro_l8192_v1 | ext_ouro_p4_drain2_credit8g_c4096_f1p25_pilot24_v1 | 0.1958 | False | 0.4628 | 0.4159 |
| profile_ouro_final_center_s1_v1 | final_ouro_l8192_o32_pd_g1_cfull_r1_s1_v1 | 0.0497 | True | 0.4901 | 0.4772 |
| ext_profile_tool_qwen_bi_eager_P_only_v1 | ext_profile_tool_qwen_bi_eager_P_clean_v1 | 0.1029 | False | 0.4278 | NOT_MEASURED |
| profile_ouro_final_center_v1 | final_ouro_l8192_o32_pd_g1_cfull_r1_s0_v1 | 0.6958 | False | 0.3313 | 0.4879 |
| ext_profile_qwen8_l8000_c4096_r1_s0_v1 | ext_final_qwen8_l8000_c4096_r1_s0_v1 | -0.0468 | True | 0.9364 | NOT_MEASURED |
| profile_ouro_native_transport_credit1g_pilot_v1 | pilot_ouro_l8192_o32_cfull_credit1g_f0p5_v1 | 0.0247 | True | 0.3565 | 0.3820 |
| profile_graph_traced_ouro_l512_v2 | profile_graph_clean_ouro_l512_v2 | -0.3487 | False | 0.0846 | 0.2689 |
| ext_profile_ouro_l16384_c4096_r1_s1_v1 | ext_final_ouro_l16384_c4096_r1_s1_v1 | 0.1419 | False | 0.5140 | 0.7340 |
| profile_tool_smoke_ouro_l512_v1 | NOT_MEASURED | NOT_MEASURED | False | 0.0742 | NOT_MEASURED |
| ext_profile_tool_qwen_bi_compiled_none_P_only_v1 | ext_profile_tool_qwen_bi_compiled_none_P_clean_v1 | 0.0158 | True | 0.1077 | NOT_MEASURED |
| ext_profile_ouro_l16384_c4096_r1_s0_v1 | ext_final_ouro_l16384_c4096_r1_s0_v1 | 0.1667 | False | 0.5172 | 0.7505 |
| ext_profile_ouro_l16384_cfull_r1_s0_v1 | ext_final_ouro_l16384_cfull_r1_s0_v1 | -0.2655 | False | 0.5852 | 0.3295 |

Graph-level tracing omits D graph-internal kernels.
D graph-active time is not exclusive kernel time or SM occupancy.
The parser transfers a profile only to its matched clean run within the preregistered 5% wall-overhead limit.

The model-free two-process NIXL probe measures 6-GiB payload completion at about 79.993 ms with 1-GiB wire credit.
Its effective rate is about 80.55 GB/s with exact payload checks.
This contiguous-buffer operator measurement is not native paged-KV E2E latency or pure DMA time.

The rejected single-process payload probe remains preserved.
It did not reproduce native process/progress geometry and cannot estimate native Serving bandwidth.

## Uncertainty, figures, and host shares

`paired_start_comparisons.csv` reports paired 95% bootstrap intervals across independent starts.
Three starts remain a small inferential sample.
Queued requests are not independent server starts.

`slo_sensitivity.csv` contains secondary threshold sensitivity only.
Those scores do not change the frozen primary decision.
`figures/figure_manifest.json` maps each plot to actual runs.

The timeline separates measured host envelopes from recorded device intervals.
Blank unrecorded intervals do not become measured GPU idle.
`e2e_component_shares.csv` and `e2e_component_shares_summary.csv` report exclusive host interval percentages.

Their components sum to 100% per request or run.
They are not dispatch/GEMM/DMA critical-path percentages.
Overlapping kernels and communication cannot be summed into exclusive E2E shares.

The exclusive host decomposition clamps asynchronous event boundaries to the preceding boundary.
Raw coordinator timestamps remain unchanged in each run.

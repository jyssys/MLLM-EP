# Strong C3 Continuous Serving

상태: FINAL.
생성 시각: 2026-11-23T02:03:52.335382+00:00.
경과 연구 시간: 18.951시간.
기준 Commit: `5ebed239073b4fa3709361038cbec026d1c32b78`.
연구 Branch: `flashvep/longbench400-v2-output-aware-2d-20261122`.

## 실행 계약

증거 수준은 `MEASURED_ONLINE`이다. Main은 Natural EOS와 Output Cap2048을 사용한다.
C3는 실제 Continuous Batching을 사용한다. 실제 Batch는 각 실행의 CSV에 기록한다.
B8192/S1024는 Scheduler 상한이다. 이 값은 실제 Batch가 아니다.
Stock Token은 원래 Compiled Full-model 경로를 사용한다. C0 Mode0 Token으로 대체하지 않았다.
Hybrid는 두 축을 모두 분할한다. Layer-only 또는 Token-only를 Hybrid로 이름만 변경하지 않았다.
고정 후보는 Discovery에서 선택했다. Held-out 성능을 본 뒤 후보를 변경하지 않는다.
Main400은 Discovery와 Held-out 요청을 모두 포함한다. Held-out192 결과는 별도 CI로 표시한다.
Natural EOS의 Output Token 총수가 5%를 초과해 다르면 Work-unmatched로 표시한다.
Fixed-length Replay는 별도 Diagnostic이다. 이를 Natural Serving 이득으로 사용하지 않는다.
Finite-cohort Throughput은 Sustainable Capacity가 아니다.

## 동결 후보

| Family | 설정 | Layer Cut 또는 Token Work Cut | Hybrid Suffix 비율 | D Cap | Source Credit | P Partial Compile | D Partial Compile | Short Fused | N1 Decode Point Screen |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Layer | Layer_Cap2_Credit16 | 2 | 0 | 2 | 16 | True | False | False | True |
| StockToken | StockToken_Cap0_Credit0 | 8 | 0 | 0 | 0 | False | False | False | True |
| Hybrid | Hybrid_FusedShort_DCompiledInterval_Cap2_Credit8_ContextFixV15 | 2 | 0.125 | 2 | 8 | True | True | True | True |
| Deflection | Deflection_Cap1_Credit8 | 0 | 0 | 1 | 8 | False | False | False | True |

Stock Token의 Partial Compile 표시는 적용 대상이 아니다. Stock Token은 원래 Full-model Compiled 경로를 사용한다.
## C3 Iso-work Bridge의 제한

Bridge는 Input 3K–6K와 20K–36K의 원문을 각각 16개 사용한다.
Layer Last8, Stock Token Work Cut8 및 Hybrid Last4/Suffix≈1/8은 정확한 Aggregate P/D Rows를 맞춘다.
Output16/192와 QPS4를 사용한다. 각 조건은 독립 Start3회다.
Axis의 D Cap은 2다. Layer/Hybrid Source Credit은 8이다. Stock Token의 Source Credit은 0이다.
Credit0은 무제한 Coordinator Window다. Worker KV 및 Scheduler 제한이 없다는 뜻은 아니다.
C0와 QPS, 실제 Batch 및 Compiler 경로가 다르다.
따라서 C0/C3 차이를 Batch 하나의 인과 효과로 표현하지 않는다.

## c3_main_serving_v2

| QPS | Method | Starts | TTFT p95 ms | Mean TBT ms | ITL p99 ms | TPOT p95 ms/token | Max ITL Start-mean ms | Raw Tokens/s | SLO Tokens/s | 모든 ITL 통과율 | Joint 통과율 | 품질 PASS | Decode CI+Absolute PASS |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0.8 | Agg | 1 | 1270.442 | 9.177 | 34.901 | 30.068 | 411.040 | 44.174 | 32.676 | 0.8475 | 0.8475 | True | False |
| 0.8 | NativeStrong | 1 | 1727.856 | 7.429 | 10.305 | 8.348 | 61.646 | 44.653 | 41.068 | 0.9925 | 0.9925 | True | False |
| 1.4 | Agg | 1 | 1435.589 | 12.066 | 212.673 | 51.315 | 471.470 | 81.567 | 38.757 | 0.6675 | 0.6675 | True | False |
| 1.4 | Deflection | 1 | 4021.845 | 7.195 | 8.495 | 7.945 | 156.129 | 75.472 | 71.788 | 0.9900 | 0.9525 | True | False |
| 1.4 | Hybrid | 1 | 4261.524 | 7.260 | 8.555 | 8.047 | 59.177 | 76.601 | 75.500 | 0.9950 | 0.9900 | True | False |
| 1.4 | Layer | 1 | 4797.525 | 7.217 | 8.531 | 7.945 | 114.784 | 77.402 | 72.885 | 0.9925 | 0.9475 | True | False |
| 1.4 | NativeCommon0 | 1 | 3217.418 | 7.451 | 12.914 | 8.173 | 68.413 | 74.862 | 70.125 | 0.9825 | 0.9825 | True | False |
| 1.4 | NativeCommon1 | 1 | 4007.970 | 7.192 | 8.483 | 7.945 | 167.661 | 75.553 | 71.902 | 0.9950 | 0.9575 | True | False |
| 1.4 | NativeCommon2 | 1 | 3604.900 | 7.278 | 8.862 | 8.056 | 381.739 | 74.273 | 70.910 | 0.9925 | 0.9700 | True | False |
| 1.4 | NativeStrong | 1 | 2779.103 | 7.666 | 16.072 | 8.946 | 52.976 | 79.518 | 67.635 | 0.9700 | 0.9700 | True | False |
| 1.4 | StockToken | 1 | 4510.098 | 7.284 | 8.914 | 8.135 | 66.491 | 77.405 | 72.890 | 0.9925 | 0.9450 | True | False |
| 2.0 | Agg | 3 | 1477.006 | 16.387 | 277.839 | 141.579 | 426.820 | 118.693 | 37.482 | 0.5208 | 0.5200 | True | False |
| 2.0 | Deflection | 3 | 10422.885 | 7.174 | 8.416 | 7.916 | 308.544 | 109.228 | 66.568 | 0.9958 | 0.5983 | True | True |
| 2.0 | Hybrid | 3 | 11767.609 | 7.272 | 8.572 | 8.050 | 236.634 | 105.978 | 49.802 | 0.9858 | 0.4650 | True | True |
| 2.0 | Layer | 3 | 11679.085 | 7.219 | 8.538 | 7.955 | 96.674 | 112.677 | 49.029 | 0.9925 | 0.4642 | True | True |
| 2.0 | NativeCommon0 | 3 | 8760.690 | 7.718 | 14.928 | 10.940 | 1199.024 | 109.153 | 73.836 | 0.9933 | 0.6758 | True | False |
| 2.0 | NativeCommon1 | 3 | 13826.798 | 7.182 | 8.468 | 7.924 | 103.974 | 110.220 | 51.983 | 0.9950 | 0.4450 | True | True |
| 2.0 | NativeCommon2 | 3 | 8533.732 | 7.366 | 8.968 | 8.195 | 99.045 | 107.297 | 75.502 | 0.9925 | 0.7108 | True | True |
| 2.0 | NativeStrong | 3 | 8161.605 | 7.872 | 18.723 | 8.968 | 178.294 | 109.402 | 70.139 | 0.9567 | 0.7350 | True | True |
| 2.0 | StockToken | 3 | 11483.980 | 7.525 | 9.478 | 8.438 | 290.827 | 114.035 | 69.358 | 0.9967 | 0.6075 | True | True |
| 2.6 | Agg | 1 | 1781.656 | 25.593 | 333.445 | 239.756 | 499.724 | 151.723 | 28.323 | 0.3625 | 0.3625 | True | False |
| 2.6 | NativeCommon0 | 1 | 32618.641 | 7.882 | 18.645 | 13.358 | 69.525 | 123.855 | 13.694 | 0.9975 | 0.1150 | True | False |
| 2.6 | NativeCommon2 | 1 | 26401.024 | 7.422 | 9.136 | 8.200 | 86.996 | 121.355 | 13.297 | 0.9925 | 0.1150 | True | False |
| 2.6 | NativeStrong | 1 | 25992.035 | 7.933 | 18.236 | 9.087 | 94.034 | 129.661 | 14.988 | 0.9400 | 0.1250 | True | False |
| 2.6 | StockToken | 1 | 21187.251 | 7.753 | 9.981 | 8.743 | 63.245 | 128.257 | 16.927 | 0.9975 | 0.1200 | True | False |
| 3.2 | Agg | 1 | 2669.509 | 49.609 | 368.818 | 265.246 | 513.725 | 177.778 | 14.935 | 0.2000 | 0.2000 | True | False |
| 3.2 | Deflection | 1 | 54313.869 | 7.175 | 8.482 | 7.918 | 68.001 | 117.955 | 8.002 | 0.9975 | 0.0775 | True | False |
| 3.2 | Layer | 1 | 61239.445 | 7.217 | 8.514 | 7.940 | 65.595 | 120.224 | 6.036 | 0.9950 | 0.0625 | True | False |
| 3.2 | NativeCommon0 | 1 | 59396.962 | 7.876 | 18.684 | 13.212 | 157.302 | 123.958 | 5.902 | 0.9825 | 0.0575 | True | False |
| 3.2 | NativeCommon1 | 1 | 58712.564 | 7.172 | 8.499 | 7.918 | 142.791 | 117.265 | 5.500 | 0.9950 | 0.0575 | True | False |
| 3.2 | NativeCommon2 | 1 | 53433.426 | 7.374 | 9.036 | 8.161 | 53.552 | 119.856 | 5.810 | 0.9975 | 0.0625 | True | False |
| 3.2 | NativeStrong | 1 | 53322.080 | 8.065 | 22.373 | 9.178 | 355.395 | 126.512 | 6.018 | 0.8850 | 0.0625 | True | False |
| 3.2 | StockToken | 1 | 50061.985 | 7.687 | 9.638 | 8.621 | 64.201 | 129.359 | 7.905 | 0.9975 | 0.0725 | True | False |
| 4.0 | Agg | 1 | 5932.787 | 120.304 | 404.527 | 278.590 | 537.758 | 201.069 | 2.810 | 0.0550 | 0.0550 | True | False |
| 4.0 | NativeStrong | 1 | 73210.861 | 7.994 | 20.599 | 9.049 | 70.604 | 120.636 | 3.709 | 0.8750 | 0.0400 | True | False |
| 4.8 | Agg | 1 | 17299.230 | 151.100 | 397.943 | 272.394 | 572.794 | 214.221 | 0.143 | 0.0425 | 0.0125 | True | False |
| 4.8 | NativeStrong | 1 | 90124.348 | 7.880 | 18.833 | 8.880 | 65.883 | 118.685 | 3.363 | 0.9275 | 0.0350 | True | False |
| 6.0 | Agg | 1 | 32851.978 | 159.382 | 403.997 | 272.158 | 539.054 | 212.997 | 0.038 | 0.0475 | 0.0025 | True | False |
| 6.0 | NativeStrong | 1 | 103389.876 | 8.021 | 22.335 | 9.092 | 116.275 | 123.254 | 2.741 | 0.9025 | 0.0300 | True | False |

Max ITL 열은 Start별 최댓값의 평균이다. 개별 Start 최댓값은 Per-start CSV에 있다.

| QPS | 비교 | Paired Starts | TTFT p95 비율 [95% CI] | SLO-goodput 비율 [95% CI] | Output Work 불일치 |
| --- | --- | --- | --- | --- | --- |
| 0.8 | Agg/NativeStrong | 1 | 0.7353 [CI 없음] | 0.7956 [CI 없음] | False |
| 1.4 | Agg/NativeStrong | 1 | 0.5166 [CI 없음] | 0.5730 [CI 없음] | False |
| 1.4 | Layer/StockToken | 1 | 1.0637 [CI 없음] | 0.9999 [CI 없음] | False |
| 1.4 | Hybrid/StockToken | 1 | 0.9449 [CI 없음] | 1.0358 [CI 없음] | False |
| 1.4 | Hybrid/Layer | 1 | 0.8883 [CI 없음] | 1.0359 [CI 없음] | False |
| 1.4 | Layer/NativeStrong | 1 | 1.7263 [CI 없음] | 1.0776 [CI 없음] | False |
| 1.4 | StockToken/NativeStrong | 1 | 1.6229 [CI 없음] | 1.0777 [CI 없음] | False |
| 1.4 | Hybrid/NativeStrong | 1 | 1.5334 [CI 없음] | 1.1163 [CI 없음] | False |
| 1.4 | Deflection/NativeStrong | 1 | 1.4472 [CI 없음] | 1.0614 [CI 없음] | True |
| 1.4 | Layer/NativeCommon2 | 1 | 1.3308 [CI 없음] | 1.0279 [CI 없음] | False |
| 1.4 | StockToken/NativeCommon0 | 1 | 1.4018 [CI 없음] | 1.0394 [CI 없음] | False |
| 1.4 | Hybrid/NativeCommon2 | 1 | 1.1821 [CI 없음] | 1.0647 [CI 없음] | False |
| 1.4 | Deflection/NativeCommon1 | 1 | 1.0035 [CI 없음] | 0.9984 [CI 없음] | False |
| 2.0 | Agg/NativeStrong | 3 | 0.1810 [0.1250, 0.3192] | 0.5344 [0.4629, 0.6578] | True |
| 2.0 | Layer/StockToken | 3 | 1.0170 [0.7862, 1.2199] | 0.7069 [0.5710, 0.8283] | False |
| 2.0 | Hybrid/StockToken | 3 | 1.0247 [0.7388, 1.3917] | 0.7180 [0.3894, 0.9141] | True |
| 2.0 | Hybrid/Layer | 3 | 1.0076 [0.9042, 1.1519] | 1.0158 [0.6730, 1.1725] | True |
| 2.0 | Layer/NativeStrong | 3 | 1.4310 [1.2567, 1.8980] | 0.6990 [0.5460, 0.7982] | True |
| 2.0 | StockToken/NativeStrong | 3 | 1.4071 [1.0306, 2.4521] | 0.9889 [0.9071, 1.0788] | True |
| 2.0 | Hybrid/NativeStrong | 3 | 1.4418 [1.2872, 1.8055] | 0.7101 [0.3770, 0.8871] | True |
| 2.0 | Deflection/NativeStrong | 3 | 1.2771 [1.1271, 1.7226] | 0.9491 [0.8899, 0.9849] | False |
| 2.0 | Layer/NativeCommon2 | 3 | 1.3686 [1.2247, 1.6797] | 0.6494 [0.5468, 0.7095] | True |
| 2.0 | StockToken/NativeCommon0 | 3 | 1.3109 [0.9409, 2.1611] | 0.9394 [0.8294, 1.0965] | True |
| 2.0 | Hybrid/NativeCommon2 | 3 | 1.3790 [1.2646, 1.6007] | 0.6596 [0.3750, 0.7886] | False |
| 2.0 | Deflection/NativeCommon1 | 3 | 0.7538 [0.5899, 0.9716] | 1.2806 [1.0996, 1.7190] | True |
| 2.6 | Agg/NativeStrong | 1 | 0.0685 [CI 없음] | 1.8897 [CI 없음] | False |
| 2.6 | StockToken/NativeStrong | 1 | 0.8151 [CI 없음] | 1.1293 [CI 없음] | True |
| 2.6 | StockToken/NativeCommon0 | 1 | 0.6495 [CI 없음] | 1.2361 [CI 없음] | True |
| 3.2 | Agg/NativeStrong | 1 | 0.0501 [CI 없음] | 2.4815 [CI 없음] | False |
| 3.2 | Layer/StockToken | 1 | 1.2233 [CI 없음] | 0.7636 [CI 없음] | False |
| 3.2 | Layer/NativeStrong | 1 | 1.1485 [CI 없음] | 1.0029 [CI 없음] | False |
| 3.2 | StockToken/NativeStrong | 1 | 0.9389 [CI 없음] | 1.3134 [CI 없음] | False |
| 3.2 | Deflection/NativeStrong | 1 | 1.0186 [CI 없음] | 1.3295 [CI 없음] | True |
| 3.2 | Layer/NativeCommon2 | 1 | 1.1461 [CI 없음] | 1.0388 [CI 없음] | False |
| 3.2 | StockToken/NativeCommon0 | 1 | 0.8428 [CI 없음] | 1.3393 [CI 없음] | False |
| 3.2 | Deflection/NativeCommon1 | 1 | 0.9251 [CI 없음] | 1.4549 [CI 없음] | False |
| 4.0 | Agg/NativeStrong | 1 | 0.0810 [CI 없음] | 0.7575 [CI 없음] | True |
| 4.8 | Agg/NativeStrong | 1 | 0.1919 [CI 없음] | 0.0426 [CI 없음] | True |
| 6.0 | Agg/NativeStrong | 1 | 0.3177 [CI 없음] | 0.0139 [CI 없음] | True |

[c3_main_serving_v2_per_start.csv](dense_ar_longbench400_v2_20261122/figures/c3_main_serving_v2_per_start.csv)

[c3_main_serving_v2_summary.csv](dense_ar_longbench400_v2_20261122/figures/c3_main_serving_v2_summary.csv)

[c3_main_serving_v2_paired_comparisons.csv](dense_ar_longbench400_v2_20261122/figures/c3_main_serving_v2_paired_comparisons.csv)

동결 계획의 실행 상태는 다음과 같다. 미완료 조건의 값을 보간하지 않는다.

| QPS | Method | 실행 상태 | 계획 Start 수 |
| --- | --- | --- | --- |
| 0.8 | Agg | COMPLETE | 1 |
| 0.8 | Deflection | NOT_RUN | 1 |
| 0.8 | Hybrid | NOT_RUN | 1 |
| 0.8 | Layer | NOT_RUN | 1 |
| 0.8 | NativeCommon0 | NOT_RUN | 1 |
| 0.8 | NativeCommon1 | NOT_RUN | 1 |
| 0.8 | NativeCommon2 | NOT_RUN | 1 |
| 0.8 | NativeStrong | COMPLETE | 1 |
| 0.8 | StockToken | NOT_RUN | 1 |
| 1.4 | Agg | COMPLETE | 1 |
| 1.4 | Deflection | COMPLETE | 1 |
| 1.4 | Hybrid | COMPLETE | 1 |
| 1.4 | Layer | COMPLETE | 1 |
| 1.4 | NativeCommon0 | COMPLETE | 1 |
| 1.4 | NativeCommon1 | COMPLETE | 1 |
| 1.4 | NativeCommon2 | COMPLETE | 1 |
| 1.4 | NativeStrong | COMPLETE | 1 |
| 1.4 | StockToken | COMPLETE | 1 |
| 2.0 | Agg | COMPLETE | 3 |
| 2.0 | Deflection | COMPLETE | 3 |
| 2.0 | Hybrid | COMPLETE | 3 |
| 2.0 | Layer | COMPLETE | 3 |
| 2.0 | NativeCommon0 | COMPLETE | 3 |
| 2.0 | NativeCommon1 | COMPLETE | 3 |
| 2.0 | NativeCommon2 | COMPLETE | 3 |
| 2.0 | NativeStrong | COMPLETE | 3 |
| 2.0 | StockToken | COMPLETE | 3 |
| 2.6 | Agg | COMPLETE | 1 |
| 2.6 | Deflection | NOT_RUN | 1 |
| 2.6 | Hybrid | NOT_RUN | 1 |
| 2.6 | Layer | NOT_RUN | 1 |
| 2.6 | NativeCommon0 | COMPLETE | 1 |
| 2.6 | NativeCommon1 | NOT_RUN | 1 |
| 2.6 | NativeCommon2 | COMPLETE | 1 |
| 2.6 | NativeStrong | COMPLETE | 1 |
| 2.6 | StockToken | COMPLETE | 1 |
| 3.2 | Agg | COMPLETE | 1 |
| 3.2 | Deflection | COMPLETE | 1 |
| 3.2 | Hybrid | NOT_RUN | 1 |
| 3.2 | Layer | COMPLETE | 1 |
| 3.2 | NativeCommon0 | COMPLETE | 1 |
| 3.2 | NativeCommon1 | COMPLETE | 1 |
| 3.2 | NativeCommon2 | COMPLETE | 1 |
| 3.2 | NativeStrong | COMPLETE | 1 |
| 3.2 | StockToken | COMPLETE | 1 |
| 4.0 | Agg | COMPLETE | 1 |
| 4.0 | Agg | NOT_RUN | 2 |
| 4.0 | Deflection | NOT_RUN | 3 |
| 4.0 | Hybrid | NOT_RUN | 3 |
| 4.0 | Layer | NOT_RUN | 3 |
| 4.0 | NativeCommon0 | NOT_RUN | 3 |
| 4.0 | NativeCommon1 | NOT_RUN | 3 |
| 4.0 | NativeCommon2 | NOT_RUN | 3 |
| 4.0 | NativeStrong | COMPLETE | 1 |
| 4.0 | NativeStrong | NOT_RUN | 2 |
| 4.0 | StockToken | NOT_RUN | 3 |
| 4.8 | Agg | COMPLETE | 1 |
| 4.8 | Agg | NOT_RUN | 2 |
| 4.8 | Deflection | NOT_RUN | 3 |
| 4.8 | Hybrid | NOT_RUN | 3 |
| 4.8 | Layer | NOT_RUN | 3 |
| 4.8 | NativeCommon0 | NOT_RUN | 3 |
| 4.8 | NativeCommon1 | NOT_RUN | 3 |
| 4.8 | NativeCommon2 | NOT_RUN | 3 |
| 4.8 | NativeStrong | COMPLETE | 1 |
| 4.8 | NativeStrong | NOT_RUN | 2 |
| 4.8 | StockToken | NOT_RUN | 3 |
| 6.0 | Agg | COMPLETE | 1 |
| 6.0 | Deflection | NOT_RUN | 1 |
| 6.0 | Hybrid | NOT_RUN | 1 |
| 6.0 | Layer | NOT_RUN | 1 |
| 6.0 | NativeCommon0 | NOT_RUN | 1 |
| 6.0 | NativeCommon1 | NOT_RUN | 1 |
| 6.0 | NativeCommon2 | NOT_RUN | 1 |
| 6.0 | NativeStrong | COMPLETE | 1 |
| 6.0 | StockToken | NOT_RUN | 1 |


## c3_heldout_serving_v2

| QPS | Method | Starts | TTFT p95 ms | Mean TBT ms | ITL p99 ms | TPOT p95 ms/token | Max ITL Start-mean ms | Raw Tokens/s | SLO Tokens/s | 모든 ITL 통과율 | Joint 통과율 | 품질 PASS | Decode CI+Absolute PASS |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2.0 | Agg | 3 | 1523.243 | 16.760 | 280.247 | 148.461 | 616.269 | 117.467 | 40.023 | 0.5312 | 0.5312 | True | False |
| 2.0 | Deflection | 3 | 7775.740 | 7.200 | 8.438 | 7.906 | 156.801 | 105.418 | 79.394 | 0.9896 | 0.7500 | True | True |
| 2.0 | Hybrid | 3 | 10891.217 | 7.295 | 8.536 | 7.965 | 192.685 | 103.427 | 63.416 | 0.9826 | 0.6250 | True | True |
| 2.0 | Layer | 3 | 10118.770 | 7.235 | 8.503 | 7.948 | 39.234 | 105.793 | 62.345 | 0.9965 | 0.5833 | True | True |
| 2.0 | NativeCommon0 | 3 | 9050.683 | 7.737 | 13.951 | 11.636 | 102.184 | 111.326 | 85.096 | 0.9549 | 0.7587 | True | False |
| 2.0 | NativeCommon1 | 3 | 9251.179 | 7.179 | 8.447 | 7.921 | 109.647 | 103.927 | 58.279 | 0.9878 | 0.5764 | True | True |
| 2.0 | NativeCommon2 | 3 | 9481.805 | 7.398 | 8.865 | 8.111 | 98.270 | 112.035 | 84.373 | 0.9896 | 0.7396 | True | True |
| 2.0 | NativeStrong | 3 | 7007.557 | 7.793 | 18.395 | 8.594 | 64.963 | 112.244 | 73.508 | 0.9427 | 0.8125 | True | False |
| 2.0 | StockToken | 3 | 9663.517 | 7.489 | 9.254 | 8.296 | 39.323 | 108.278 | 82.721 | 0.9948 | 0.7604 | True | True |

Max ITL 열은 Start별 최댓값의 평균이다. 개별 Start 최댓값은 Per-start CSV에 있다.

| QPS | 비교 | Paired Starts | TTFT p95 비율 [95% CI] | SLO-goodput 비율 [95% CI] | Output Work 불일치 |
| --- | --- | --- | --- | --- | --- |
| 2.0 | Agg/NativeStrong | 3 | 0.2174 [0.1883, 0.2598] | 0.5445 [0.5208, 0.5674] | True |
| 2.0 | Layer/StockToken | 3 | 1.0471 [0.7719, 1.5230] | 0.7537 [0.6358, 0.8741] | True |
| 2.0 | Hybrid/StockToken | 3 | 1.1270 [0.6096, 1.9694] | 0.7666 [0.6051, 1.0342] | True |
| 2.0 | Hybrid/Layer | 3 | 1.0763 [0.7821, 1.2922] | 1.0172 [0.7936, 1.2524] | True |
| 2.0 | Layer/NativeStrong | 3 | 1.4440 [1.2620, 1.7733] | 0.8481 [0.7555, 1.0054] | True |
| 2.0 | StockToken/NativeStrong | 3 | 1.3790 [0.9085, 2.3230] | 1.1253 [0.9327, 1.2846] | True |
| 2.0 | Hybrid/NativeStrong | 3 | 1.5542 [1.3309, 1.7863] | 0.8627 [0.7774, 0.9597] | True |
| 2.0 | Deflection/NativeStrong | 3 | 1.1096 [0.9842, 1.3077] | 1.0801 [0.9291, 1.2218] | False |
| 2.0 | Layer/NativeCommon2 | 3 | 1.0672 [0.9268, 1.2278] | 0.7389 [0.6495, 0.8353] | True |
| 2.0 | StockToken/NativeCommon0 | 3 | 1.0677 [0.5579, 1.9595] | 0.9721 [0.8146, 1.1326] | True |
| 2.0 | Hybrid/NativeCommon2 | 3 | 1.1486 [0.7285, 1.5869] | 0.7516 [0.6321, 0.9238] | True |
| 2.0 | Deflection/NativeCommon1 | 3 | 0.8405 [0.6606, 0.9739] | 1.3623 [1.0722, 1.9035] | False |

[c3_heldout_serving_v2_per_start.csv](dense_ar_longbench400_v2_20261122/figures/c3_heldout_serving_v2_per_start.csv)

[c3_heldout_serving_v2_summary.csv](dense_ar_longbench400_v2_20261122/figures/c3_heldout_serving_v2_summary.csv)

[c3_heldout_serving_v2_paired_comparisons.csv](dense_ar_longbench400_v2_20261122/figures/c3_heldout_serving_v2_paired_comparisons.csv)

동결 계획의 실행 상태는 다음과 같다. 미완료 조건의 값을 보간하지 않는다.

| QPS | Method | 실행 상태 | 계획 Start 수 |
| --- | --- | --- | --- |
| 2.0 | Agg | COMPLETE | 3 |
| 2.0 | Deflection | COMPLETE | 3 |
| 2.0 | Hybrid | COMPLETE | 3 |
| 2.0 | Layer | COMPLETE | 3 |
| 2.0 | NativeCommon0 | COMPLETE | 3 |
| 2.0 | NativeCommon1 | COMPLETE | 3 |
| 2.0 | NativeCommon2 | COMPLETE | 3 |
| 2.0 | NativeStrong | COMPLETE | 3 |
| 2.0 | StockToken | COMPLETE | 3 |
| 4.0 | Agg | NOT_RUN | 3 |
| 4.0 | Deflection | NOT_RUN | 3 |
| 4.0 | Hybrid | NOT_RUN | 3 |
| 4.0 | Layer | NOT_RUN | 3 |
| 4.0 | NativeCommon0 | NOT_RUN | 3 |
| 4.0 | NativeCommon1 | NOT_RUN | 3 |
| 4.0 | NativeCommon2 | NOT_RUN | 3 |
| 4.0 | NativeStrong | NOT_RUN | 3 |
| 4.0 | StockToken | NOT_RUN | 3 |
| 4.8 | Agg | NOT_RUN | 3 |
| 4.8 | Deflection | NOT_RUN | 3 |
| 4.8 | Hybrid | NOT_RUN | 3 |
| 4.8 | Layer | NOT_RUN | 3 |
| 4.8 | NativeCommon0 | NOT_RUN | 3 |
| 4.8 | NativeCommon1 | NOT_RUN | 3 |
| 4.8 | NativeCommon2 | NOT_RUN | 3 |
| 4.8 | NativeStrong | NOT_RUN | 3 |
| 4.8 | StockToken | NOT_RUN | 3 |


## c3_heldout_replay_v2

| QPS | Method | Starts | TTFT p95 ms | Mean TBT ms | ITL p99 ms | TPOT p95 ms/token | Max ITL Start-mean ms | Raw Tokens/s | SLO Tokens/s | 모든 ITL 통과율 | Joint 통과율 | 품질 PASS | Decode CI+Absolute PASS |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2.0 | Agg | 3 | 1472.130 | 16.558 | 279.021 | 149.522 | 481.128 | 113.211 | 40.872 | 0.5399 | 0.5399 | True | False |
| 2.0 | Deflection | 3 | 11355.039 | 7.178 | 8.361 | 7.885 | 174.927 | 107.076 | 72.418 | 0.9896 | 0.6615 | True | False |
| 2.0 | Hybrid | 3 | 12278.553 | 7.255 | 8.537 | 7.958 | 64.952 | 104.654 | 62.403 | 0.9896 | 0.5677 | True | True |
| 2.0 | Layer | 3 | 10909.657 | 7.209 | 8.485 | 7.956 | 32.783 | 107.165 | 67.903 | 0.9965 | 0.6163 | True | True |
| 2.0 | NativeCommon0 | 3 | 7674.363 | 7.842 | 14.785 | 12.339 | 40.380 | 112.538 | 91.222 | 0.9688 | 0.8021 | True | False |
| 2.0 | NativeCommon1 | 2 | 11060.007 | 7.174 | 8.442 | 7.916 | 46.776 | 107.567 | 64.115 | 0.9948 | 0.5573 | True | False |
| 2.0 | NativeCommon2 | 3 | 8316.890 | 7.474 | 9.076 | 8.069 | 894.561 | 112.533 | 81.110 | 0.9948 | 0.7066 | True | True |
| 2.0 | NativeStrong | 3 | 6969.188 | 7.809 | 18.063 | 8.605 | 56.950 | 112.787 | 73.124 | 0.9358 | 0.8108 | True | False |
| 2.0 | StockToken | 3 | 9414.313 | 7.466 | 9.194 | 8.296 | 31.893 | 109.389 | 79.192 | 0.9948 | 0.7188 | True | True |

Max ITL 열은 Start별 최댓값의 평균이다. 개별 Start 최댓값은 Per-start CSV에 있다.

| QPS | 비교 | Paired Starts | TTFT p95 비율 [95% CI] | SLO-goodput 비율 [95% CI] | Output Work 불일치 |
| --- | --- | --- | --- | --- | --- |
| 2.0 | Agg/NativeStrong | 3 | 0.2112 [0.1875, 0.2451] | 0.5589 [0.5388, 0.5789] | False |
| 2.0 | Layer/StockToken | 3 | 1.1588 [1.1379, 1.1892] | 0.8574 [0.8004, 0.9224] | False |
| 2.0 | Hybrid/StockToken | 3 | 1.3042 [1.2040, 1.4201] | 0.7880 [0.7381, 0.8332] | False |
| 2.0 | Hybrid/Layer | 3 | 1.1255 [1.0581, 1.1941] | 0.9190 [0.8082, 1.0164] | False |
| 2.0 | Layer/NativeStrong | 3 | 1.5654 [1.1436, 2.1667] | 0.9286 [0.7812, 1.1602] | False |
| 2.0 | StockToken/NativeStrong | 3 | 1.3508 [0.9624, 1.9042] | 1.0830 [0.9760, 1.2566] | False |
| 2.0 | Hybrid/NativeStrong | 3 | 1.7618 [1.3645, 2.3125] | 0.8534 [0.7940, 0.9349] | False |
| 2.0 | Deflection/NativeStrong | 3 | 1.6293 [1.2051, 2.2249] | 0.9903 [0.8718, 1.1620] | False |
| 2.0 | Layer/NativeCommon2 | 3 | 1.3117 [1.0613, 1.6436] | 0.8372 [0.7698, 0.9216] | False |
| 2.0 | StockToken/NativeCommon0 | 3 | 1.2267 [0.9131, 1.6334] | 0.8681 [0.8178, 0.9389] | False |
| 2.0 | Hybrid/NativeCommon2 | 3 | 1.4763 [1.2396, 1.8331] | 0.7694 [0.7364, 0.8023] | False |
| 2.0 | Deflection/NativeCommon1 | 2 | 1.0192 [CI 없음] | 1.1260 [CI 없음] | False |

[c3_heldout_replay_v2_per_start.csv](dense_ar_longbench400_v2_20261122/figures/c3_heldout_replay_v2_per_start.csv)

[c3_heldout_replay_v2_summary.csv](dense_ar_longbench400_v2_20261122/figures/c3_heldout_replay_v2_summary.csv)

[c3_heldout_replay_v2_paired_comparisons.csv](dense_ar_longbench400_v2_20261122/figures/c3_heldout_replay_v2_paired_comparisons.csv)

동결 계획의 실행 상태는 다음과 같다. 미완료 조건의 값을 보간하지 않는다.

| QPS | Method | 실행 상태 | 계획 Start 수 |
| --- | --- | --- | --- |
| 2.0 | Agg | COMPLETE | 3 |
| 2.0 | Deflection | COMPLETE | 3 |
| 2.0 | Hybrid | COMPLETE | 3 |
| 2.0 | Layer | COMPLETE | 3 |
| 2.0 | NativeCommon0 | COMPLETE | 3 |
| 2.0 | NativeCommon1 | COMPLETE | 2 |
| 2.0 | NativeCommon1 | NOT_RUN | 1 |
| 2.0 | NativeCommon2 | COMPLETE | 3 |
| 2.0 | NativeStrong | COMPLETE | 3 |
| 2.0 | StockToken | COMPLETE | 3 |
| 4.0 | Agg | NOT_RUN | 3 |
| 4.0 | Deflection | NOT_RUN | 3 |
| 4.0 | Hybrid | NOT_RUN | 3 |
| 4.0 | Layer | NOT_RUN | 3 |
| 4.0 | NativeCommon0 | NOT_RUN | 3 |
| 4.0 | NativeCommon1 | NOT_RUN | 3 |
| 4.0 | NativeCommon2 | NOT_RUN | 3 |
| 4.0 | NativeStrong | NOT_RUN | 3 |
| 4.0 | StockToken | NOT_RUN | 3 |
| 4.8 | Agg | NOT_RUN | 3 |
| 4.8 | Deflection | NOT_RUN | 3 |
| 4.8 | Hybrid | NOT_RUN | 3 |
| 4.8 | Layer | NOT_RUN | 3 |
| 4.8 | NativeCommon0 | NOT_RUN | 3 |
| 4.8 | NativeCommon1 | NOT_RUN | 3 |
| 4.8 | NativeCommon2 | NOT_RUN | 3 |
| 4.8 | NativeStrong | NOT_RUN | 3 |
| 4.8 | StockToken | NOT_RUN | 3 |


## c3_continuous_iso8_output_bridge_v2

| QPS | Method | Starts | TTFT p95 ms | Mean TBT ms | ITL p99 ms | TPOT p95 ms/token | Max ITL Start-mean ms | Raw Tokens/s | SLO Tokens/s | 모든 ITL 통과율 | Joint 통과율 | 품질 PASS | Decode CI+Absolute PASS |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 4.0 | Hybrid | 3 | 381.058 | 6.738 | 7.316 | 6.830 | 7.642 | 53.093 | 53.093 | 1.0000 | 1.0000 | True | True |
| 4.0 | Layer | 3 | 367.757 | 6.768 | 7.336 | 6.818 | 7.446 | 53.300 | 53.300 | 1.0000 | 1.0000 | True | True |
| 4.0 | NativeCommon2 | 3 | 331.479 | 6.716 | 7.302 | 6.841 | 8.456 | 53.977 | 53.977 | 1.0000 | 1.0000 | True | True |
| 4.0 | NativeStrong | 3 | 285.222 | 6.884 | 15.222 | 7.511 | 22.249 | 54.165 | 54.165 | 1.0000 | 1.0000 | True | False |
| 4.0 | StockToken | 3 | 374.337 | 6.776 | 7.338 | 6.855 | 7.697 | 53.489 | 53.489 | 1.0000 | 1.0000 | True | True |
| 4.0 | Hybrid | 3 | 11615.257 | 6.964 | 7.413 | 7.125 | 22.011 | 167.669 | 83.828 | 1.0000 | 0.5000 | True | True |
| 4.0 | Layer | 3 | 11152.773 | 6.987 | 7.413 | 7.182 | 22.718 | 178.514 | 74.295 | 0.9792 | 0.4167 | True | False |
| 4.0 | NativeCommon2 | 3 | 6015.440 | 7.081 | 7.504 | 7.249 | 18.178 | 243.917 | 182.911 | 0.9792 | 0.7500 | True | False |
| 4.0 | NativeStrong | 3 | 320.814 | 8.272 | 17.609 | 8.752 | 34.965 | 498.588 | 387.400 | 0.7917 | 0.7917 | True | False |
| 4.0 | StockToken | 3 | 7559.291 | 7.078 | 7.531 | 7.237 | 34.179 | 222.503 | 129.644 | 0.9167 | 0.5833 | True | False |
| 4.0 | Hybrid | 3 | 7005.609 | 7.846 | 9.179 | 8.298 | 25.218 | 21.518 | 13.445 | 0.9792 | 0.6250 | True | False |
| 4.0 | Layer | 3 | 7409.358 | 7.581 | 8.728 | 7.991 | 8.971 | 20.737 | 13.399 | 1.0000 | 0.6458 | True | True |
| 4.0 | NativeCommon2 | 3 | 10152.086 | 7.711 | 8.884 | 7.946 | 9.069 | 16.871 | 7.732 | 1.0000 | 0.4583 | True | True |
| 4.0 | NativeStrong | 3 | 10152.218 | 7.718 | 8.892 | 7.946 | 9.560 | 16.855 | 7.724 | 1.0000 | 0.4583 | True | True |
| 4.0 | StockToken | 3 | 6380.495 | 7.781 | 8.951 | 8.045 | 9.598 | 22.775 | 16.612 | 1.0000 | 0.7292 | True | True |
| 4.0 | Hybrid | 3 | 22765.584 | 8.025 | 8.561 | 8.273 | 10.611 | 102.758 | 21.404 | 1.0000 | 0.2083 | True | True |
| 4.0 | Layer | 3 | 22167.610 | 8.038 | 8.572 | 8.288 | 33.057 | 104.851 | 21.834 | 0.9792 | 0.2083 | True | False |
| 4.0 | NativeCommon2 | 3 | 11387.273 | 9.166 | 10.171 | 9.766 | 32.974 | 175.376 | 62.104 | 0.9583 | 0.3542 | True | False |
| 4.0 | NativeStrong | 3 | 10206.396 | 9.793 | 27.610 | 10.558 | 85.681 | 182.422 | 34.234 | 0.4167 | 0.1875 | True | False |
| 4.0 | StockToken | 3 | 20428.051 | 8.324 | 9.537 | 9.354 | 29.523 | 111.325 | 23.203 | 0.9792 | 0.2083 | True | False |

Max ITL 열은 Start별 최댓값의 평균이다. 개별 Start 최댓값은 Per-start CSV에 있다.

| QPS | 비교 | Paired Starts | TTFT p95 비율 [95% CI] | SLO-goodput 비율 [95% CI] | Output Work 불일치 |
| --- | --- | --- | --- | --- | --- |
| 4.0 | Layer/StockToken | 3 | 0.9824 [0.9316, 1.0363] | 0.9965 [0.9927, 1.0011] | False |
| 4.0 | Hybrid/StockToken | 3 | 1.0180 [0.9023, 1.1242] | 0.9926 [0.9883, 0.9982] | False |
| 4.0 | Hybrid/Layer | 3 | 1.0362 [0.9662, 1.0848] | 0.9961 [0.9932, 0.9988] | False |
| 4.0 | Layer/NativeStrong | 3 | 1.2894 [1.1920, 1.4168] | 0.9840 [0.9754, 0.9927] | False |
| 4.0 | StockToken/NativeStrong | 3 | 1.3124 [1.1964, 1.5041] | 0.9875 [0.9786, 0.9986] | False |
| 4.0 | Hybrid/NativeStrong | 3 | 1.3360 [1.2594, 1.4439] | 0.9802 [0.9688, 0.9915] | False |
| 4.0 | Layer/StockToken | 3 | 1.4754 [1.4280, 1.5227] | 0.5731 [0.5243, 0.6339] | False |
| 4.0 | Hybrid/StockToken | 3 | 1.5366 [1.5074, 1.5627] | 0.6466 [0.5239, 0.7578] | False |
| 4.0 | Hybrid/Layer | 3 | 1.0415 [1.0263, 1.0556] | 1.1283 [0.9807, 1.2483] | False |
| 4.0 | Layer/NativeStrong | 3 | 34.7639 [28.8597, 44.4232] | 0.1918 [0.1633, 0.2527] | False |
| 4.0 | StockToken/NativeStrong | 3 | 23.5628 [20.2056, 29.1642] | 0.3347 [0.2681, 0.4759] | False |
| 4.0 | Hybrid/NativeStrong | 3 | 36.2055 [30.4615, 45.5787] | 0.2164 [0.2031, 0.2470] | False |
| 4.0 | Layer/StockToken | 3 | 1.1613 [1.1485, 1.1774] | 0.8066 [0.7001, 0.9008] | False |
| 4.0 | Hybrid/StockToken | 3 | 1.0980 [1.0830, 1.1202] | 0.8094 [0.7116, 0.9144] | False |
| 4.0 | Hybrid/Layer | 3 | 0.9455 [0.9383, 0.9531] | 1.0034 [0.9633, 1.0418] | False |
| 4.0 | Layer/NativeStrong | 3 | 0.7298 [0.7141, 0.7444] | 1.7347 [1.6483, 1.8184] | False |
| 4.0 | StockToken/NativeStrong | 3 | 0.6285 [0.6077, 0.6450] | 2.1507 [1.9187, 2.4163] | False |
| 4.0 | Hybrid/NativeStrong | 3 | 0.6901 [0.6806, 0.6985] | 1.7407 [1.7172, 1.7553] | False |
| 4.0 | Layer/StockToken | 3 | 1.0852 [1.0745, 1.0957] | 0.9410 [0.7441, 1.1899] | False |
| 4.0 | Hybrid/StockToken | 3 | 1.1144 [1.1091, 1.1208] | 0.9225 [0.9188, 0.9262] | False |
| 4.0 | Hybrid/Layer | 3 | 1.0270 [1.0229, 1.0322] | 0.9803 [0.7797, 1.2372] | False |
| 4.0 | Layer/NativeStrong | 3 | 2.1719 [2.1125, 2.2305] | 0.6378 [0.3397, 1.8289] | False |
| 4.0 | StockToken/NativeStrong | 3 | 2.0015 [1.9614, 2.0483] | 0.6778 [0.4551, 1.5282] | False |
| 4.0 | Hybrid/NativeStrong | 3 | 2.2305 [2.1800, 2.2827] | 0.6252 [0.4181, 1.4155] | False |

[c3_continuous_iso8_output_bridge_v2_per_start.csv](dense_ar_longbench400_v2_20261122/figures/c3_continuous_iso8_output_bridge_v2_per_start.csv)

[c3_continuous_iso8_output_bridge_v2_summary.csv](dense_ar_longbench400_v2_20261122/figures/c3_continuous_iso8_output_bridge_v2_summary.csv)

[c3_continuous_iso8_output_bridge_v2_paired_comparisons.csv](dense_ar_longbench400_v2_20261122/figures/c3_continuous_iso8_output_bridge_v2_paired_comparisons.csv)

동결 계획의 실행 상태는 다음과 같다. 미완료 조건의 값을 보간하지 않는다.

| QPS | Method | 실행 상태 | 계획 Start 수 |
| --- | --- | --- | --- |
| 4.0 | Hybrid | COMPLETE | 12 |
| 4.0 | Layer | COMPLETE | 12 |
| 4.0 | NativeCommon2 | COMPLETE | 12 |
| 4.0 | NativeStrong | COMPLETE | 12 |
| 4.0 | StockToken | COMPLETE | 12 |

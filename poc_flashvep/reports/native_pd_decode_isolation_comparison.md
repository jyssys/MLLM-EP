# Native PD Decode Isolation 및 긴 구간 비교

## 동일 QPS의 Isolation 비율

Native는 P0/D1이다.
기존 NIXL, Source-token Reuse와 Cohort-isolated D Admission을 유지했다.
이 경로는 논문 Dynamo의 완전한 재현이 아니다.
Agg-DP2의 같은 QPS와 Prompt를 비교했다.
아래 비율은 High/Low QPS 비율이 아니다.

| Method | QPS | Metric | Aggregated/Native [95% CI] |
| --- | --- | --- | --- |
| Agg-DP2 B8192/S1024 | 2 | mean_TBT_ms | 1.270 [1.260, 1.278] |
| Agg-DP2 B8192/S1024 | 2 | ITL_p99_ms | 9.012 [8.982, 9.043] |
| Agg-DP2 B8192/S1024 | 2 | TPOT_p95_ms | 1.464 [1.401, 1.548] |
| Agg-DP2 B8192/S1024 | 6 | mean_TBT_ms | 2.504 [2.500, 2.508] |
| Agg-DP2 B8192/S1024 | 6 | ITL_p99_ms | 10.960 [10.936, 10.984] |
| Agg-DP2 B8192/S1024 | 6 | TPOT_p95_ms | 3.086 [3.084, 3.089] |
| Agg-DP2 B8192/S1024 | 10 | mean_TBT_ms | 2.545 [2.539, 2.549] |
| Agg-DP2 B8192/S1024 | 10 | ITL_p99_ms | 11.022 [11.002, 11.046] |
| Agg-DP2 B8192/S1024 | 10 | TPOT_p95_ms | 3.113 [3.108, 3.117] |

Native의 D 상한은16이다.
Agg-DP2 B8192의 상한은1024다.
따라서 Isolation 비율에는 Batch 구성과 Admission의 차이도 포함된다.
이 전체 비율을 순수한 Prefill Kernel 간섭 비용으로 표현하지 않는다.
동일 S16의 큰 Budget 실행군도 별도로 보존했다.

## Native의 Cohort Ingress와 TTFT

현재 Native는 기존 Decode Cohort가 끝날 때까지 다음 Import Group을 기다린다.
일반적인 Continuous D Ingress의 최적 기준선과 다르다.
아래 분해는 같은 요청의 Host Timestamp를 사용한다.
세 구간의 합은 요청별 TTFT와 일치한다.

| QPS | Mean TTFT (s) | Arrival→P-ready (s) | P-ready→D Socket (s) | D Socket→First Output (s) | 중간 대기의 TTFT 비율 (%) |
| --- | --- | --- | --- | --- | --- |
| 2 | 3.463 | 0.531 | 2.699 | 0.232 | 77.989 |
| 3 | 5.047 | 1.137 | 3.629 | 0.282 | 72.068 |
| 4 | 8.923 | 3.354 | 5.305 | 0.263 | 59.480 |
| 5 | 11.493 | 6.065 | 5.141 | 0.287 | 44.757 |
| 6 | 14.089 | 8.041 | 5.746 | 0.301 | 40.793 |
| 7 | 15.237 | 9.567 | 5.370 | 0.300 | 35.247 |
| 8 | 16.591 | 10.656 | 5.631 | 0.304 | 33.936 |
| 9 | 17.647 | 11.747 | 5.621 | 0.280 | 31.852 |
| 10 | 17.949 | 12.297 | 5.380 | 0.272 | 29.975 |

Arrival→P-ready에는 P Queue, 계산과 알림이 포함된다.
P-ready→D Socket에는 Cohort Guard와 Host Ingress 대기가 포함된다.
D Socket→First Output에는 Import, Barrier와 출력 전달이 포함된다.
이를 GPU 계산이나 Wire Transfer의 배타적 비용으로 표현하지 않는다.
Native의 높은 TTFT 전체를 Prefill Compute 병목으로 설명하지 않는다.
그 차이를 Dynamic Layer Scheduling의 잠재적 이득으로 그대로 계산하지 않는다.
원시 Start별 분해와 CI는 `analysis/native_cohort_ingress_TTFT_attribution.json`에 있다.

## 600초 관측과 Drain

| Run | 측정 초 | 측정 도착/완료 | Arrival-cohort Mean TBT (ms) | Active-window Mean TBT (ms) | 완료 Output Token/s | 실제 방출 Token/s | Outstanding 증가 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| long_DP2_B8192_S1024_O200_q2.0_s2026111631_W600_v1 | 600.0 | 1200/1197 | 32.548 | 32.538 | 399.000 | 400.020 | 3 |
| long_DP2_B8192_S1024_O200_q2.0_s2026111632_W600_v1 | 600.0 | 1309/1314 | 33.457 | 33.457 | 438.000 | 437.402 | -5 |
| long_DP2_B8192_S1024_O200_q2.0_s2026111633_W600_v1 | 600.0 | 1248/1243 | 32.808 | 32.891 | 414.333 | 415.230 | 5 |
| long_DP2_B8192_S1024_O200_q6.0_s2026111631_W600_v1 | 600.0 | 3624/2618 | 82.784 | 82.463 | 872.667 | 879.408 | 1006 |
| long_DP2_B8192_S1024_O200_q6.0_s2026111632_W600_v1 | 600.0 | 3708/2613 | 82.589 | 82.031 | 871.000 | 882.708 | 1095 |
| long_DP2_B8192_S1024_O200_q6.0_s2026111633_W600_v1 | 600.0 | 3654/2674 | 81.328 | 81.335 | 891.333 | 892.360 | 980 |
| long_Native_O200_q2.0_s2026111631_W600_v1 | 600.0 | 1200/1196 | 25.028 | 25.026 | 398.667 | 399.647 | 4 |
| long_Native_O200_q6.0_s2026111631_W600_v1 | 600.0 | 3624/1632 | 26.256 | 26.272 | 544.000 | 544.000 | 1992 |

| 600초 Metric | QPS6 / QPS2 [95% CI] |
| --- | --- |
| mean_TBT_ms | 2.497 [2.471, 2.529] |
| active_window_observed_mean_TBT_ms | 2.486 [2.456, 2.521] |
| ITL_p99_ms | 1.256 [1.250, 1.262] |
| TPOT_p95_ms | 2.073 [2.031, 2.121] |

Arrival-cohort Latency는 측정 구간의 모든 도착 요청을 완료까지 추적한다.
Active-window Latency는 측정 구간 안의 Token Interval만 사용한다.
두 통계를 대체하지 않는다.
완료 Output Token/s와 실제 Token 방출률도 구분했다.
Warmup과 Drain을 지속 가능한 처리 용량에 포함해 과장하지 않는다.
Queue Growth, 완료/도착 비율과 KV Lifetime을 함께 보존했다.
증가하는 Queue는 정상상태가 아니라 Overload Diagnostic이다.

## P 병목의 한계

Host Waiting Queue가 증가했다고 모두 Compute-ready로 분류하지 않는다.
WAITING_REMOTE_KVS, KV Allocation과 Source Retention을 구분한다.
Host Source-ready Rate는 순수 GPU Prefill 처리율이 아니다.
이 캠페인의 Clean P Kernel Busy는 Profiler Gate를 통과한 경우에만 인정한다.
서로 다른 실행에서 얻은 P Busy와 D 여유를 결합하지 않았다.

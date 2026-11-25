# H200 Decode Batch1 통제

Actual Batch의 현재 증거 수준은 `DIAGNOSTIC`다. Native-B1과 Token-B1은 동일 D FIFO Admission으로 한 번에 한 D 요청만 허용한다. 전체 Client, E, P는 열린 루프와 연속 Scheduling을 유지한다. Native Unrestricted는 별도 실용 기준이다. NIXL과 엔진 KV42GiB는 동일하게 유지하며, 하나의 요청에 대한 Suffix Prefill은 여러 Query Rows를 사용할 수 있다.

| 통제 Arm | 측정 |
| --- | --- |
| native-strong-B1 | 완료된 온라인 Cohort만 있음 |
| native-common-B1 | Actual Batch Counter 완료 |
| static10-B1 | Actual Batch Counter 완료 |
| idle10-B1 | Actual Batch Counter 완료 |
| native-common-free | Actual Batch Counter 완료 |

| Method | Run/Cohort | Counter 자체 검증 | Scheduled 전체 최대 | Scheduler Gen 최대 | Prepared Decode 최대 | Query Rows 최대 | Padded Rows 최대 | Decode Requests×Calls | D HTTP 동시 최대 | P-ready D Pending 최대 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| native-common-free | `b_common_free/native_common_free_2_observed` | 통과 | 4 | 4 | 4 | 4 | 4 | 1×3,747, 2×650, 3×38, 4×2 | 4 | 0 |
| idle10-B1 | `b_idle10/idle10_b1_2_observed` | 통과 | 1 | 1 | 1 | 2895 | 2895 | 1×5,169 | 1 | 5 |
| native-common-B1 | `b_native_common_v2/native_common_b1_2_observed` | 통과 | 1 | 1 | 1 | 1 | 1 | 1×5,169 | 1 | 5 |
| static10-B1 | `b_static10/static10_b1_2_observed` | 통과 | 1 | 1 | 1 | 3054 | 3054 | 1×5,169 | 1 | 7 |

Scheduled와 Prepared는 다른 계층의 관측이다. Padding은 실제 CUDA Graph 입력의 행 수이며 Active Decode Request Count로 세지 않는다. D HTTP 동시 최대는 완결된 HTTP 구간 교집합으로, Pending 최대는 P-ready Snapshot으로 계산했다. Pending Snapshot은 관측 가능한 하한이다. Counter 구간의 Geometry와 Client 지표는 원시 기록이 존재하는 Cohort에 한정한다. 같은 프로세스의 Clean/Counter/Clean은 독립 N3에 해당하지 않는다.

## 정합성 회귀 및 계측 검증

| Arm | Run | 검사 | 상태 Gate | Strict Greedy |
| --- | --- | --- | --- | --- |
| NativeCommon-B1 | b_native_common | N1 | 1/1 | 1/1 |
| NativeCommon-B1 | b_native_common | N8 | 8/8 | 6/8 |
| Static10-B1 | b_static10 | N1 | 1건 통과 | 1/1 |
| Static10-B1 | b_static10 | N8 | 8건 통과 | 8/8 |

| Run | 5% CPC Gate | 최대 Clean Spread | 최대 Counter Deviation | 실패 Metric |
| --- | --- | --- | --- | --- |
| b_common_free | 실패 | 3.63% | 7.12% | ITL_p99_ms |
| b_idle10 | 실패 | 4.47% | 8.38% | TTFT_p95_ms |
| b_native_common_v2 | 실패 | 6.18% | 9.96% | TTFT_p50_ms, ITL_p99_ms |
| b_static10 | 실패 | 11.58% | 6.57% | TTFT_p50_ms, TTFT_p95_ms |
| b_strong_b1 | 실패 | 6.34% | 7.73% | TTFT_p50_ms, TTFT_p95_ms, ITL_p99_ms |

### 요청 지연과 작업량

| Method | QPS | 출력 계약 | Run/Cohort | TTFT mean ms | TTFT p50 ms | TTFT p95 ms | Mean TBT ms | ITL p99 ms | TPOT p95 ms | 출력 토큰 | Evidence |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| native-common-free | 2.0 | Per-request equal-output replay | `b_common_free/native_common_free_2_batch_clean_after` | 2,280.33 | 2,227.89 | 4,390.59 | 4.79 | 12.37 | 5.13 | 5269 | MEASURED_ONLINE |
| native-common-free | 2.0 | Per-request equal-output replay | `b_common_free/native_common_free_2_batch_clean_before` | 2,266.99 | 2,171.21 | 4,552.83 | 4.80 | 12.00 | 5.28 | 5269 | MEASURED_ONLINE |
| native-common-free | 2.0 | Per-request equal-output replay | `b_common_free/native_common_free_2_observed` | 2,371.15 | 2,118.23 | 4,683.82 | 4.81 | 11.32 | 5.12 | 5269 | DIAGNOSTIC |
| idle10-B1 | 2.0 | Per-request equal-output replay | `b_idle10/idle10_b1_2_batch_clean_after` | 2,475.71 | 2,084.52 | 5,052.37 | 4.66 | 6.33 | 4.90 | 5269 | MEASURED_ONLINE |
| idle10-B1 | 2.0 | Per-request equal-output replay | `b_idle10/idle10_b1_2_batch_clean_before` | 2,513.48 | 2,179.80 | 4,871.38 | 4.64 | 6.09 | 4.81 | 5269 | MEASURED_ONLINE |
| idle10-B1 | 2.0 | Per-request equal-output replay | `b_idle10/idle10_b1_2_observed` | 2,658.85 | 2,144.68 | 5,377.93 | 4.64 | 6.18 | 4.79 | 5269 | DIAGNOSTIC |
| native-common-B1 | 2.0 | Per-request equal-output replay | `b_native_common/native_common_b1_2_batch_clean_before` | 2,479.37 | 2,105.74 | 5,116.49 | 4.63 | 6.07 | 4.78 | 5269 | MEASURED_ONLINE |
| native-common-B1 | 2.0 | Per-request equal-output replay | `b_native_common_v2/native_common_b1_2_batch_clean_after` | 2,715.92 | 2,226.30 | 5,350.26 | 4.65 | 5.78 | 4.81 | 5269 | MEASURED_ONLINE |
| native-common-B1 | 2.0 | Per-request equal-output replay | `b_native_common_v2/native_common_b1_2_batch_clean_before` | 2,563.81 | 2,111.62 | 5,147.30 | 4.68 | 6.15 | 4.83 | 5269 | MEASURED_ONLINE |
| native-common-B1 | 2.0 | Per-request equal-output replay | `b_native_common_v2/native_common_b1_2_observed` | 2,548.57 | 2,147.75 | 5,163.10 | 4.66 | 6.56 | 4.83 | 5269 | DIAGNOSTIC |
| static10-B1 | 2.0 | Per-request equal-output replay | `b_static10/static10_b1_2_batch_clean_after` | 3,022.30 | 2,236.54 | 6,174.54 | 4.63 | 6.01 | 4.80 | 5269 | MEASURED_ONLINE |
| static10-B1 | 2.0 | Per-request equal-output replay | `b_static10/static10_b1_2_batch_clean_before` | 2,695.20 | 2,158.22 | 5,498.68 | 4.67 | 5.95 | 4.81 | 5269 | MEASURED_ONLINE |
| static10-B1 | 2.0 | Per-request equal-output replay | `b_static10/static10_b1_2_observed` | 2,739.88 | 2,053.03 | 6,125.62 | 4.64 | 5.83 | 4.82 | 5269 | DIAGNOSTIC |
| native-strong-B1 | 2.0 | Per-request equal-output replay | `b_strong_b1/native_strong_b1_2_batch_clean_after` | 2,449.55 | 2,016.10 | 4,878.60 | 4.60 | 6.26 | 4.77 | 5269 | MEASURED_ONLINE |
| native-strong-B1 | 2.0 | Per-request equal-output replay | `b_strong_b1/native_strong_b1_2_batch_clean_before` | 2,594.95 | 2,148.00 | 5,178.64 | 4.62 | 6.30 | 4.83 | 5269 | MEASURED_ONLINE |
| native-strong-B1 | 2.0 | Per-request equal-output replay | `b_strong_b1/native_strong_b1_2_observed` | 2,507.90 | 2,121.68 | 5,114.21 | 4.60 | 5.80 | 4.74 | 5269 | DIAGNOSTIC |

### 처리량과 SLO

| Method | QPS | 출력 계약 | 입력+출력 tok/s | 출력 tok/s | 완료 req/s | SLO-goodput tok/s | TTFT Pass | Decode Pass | Joint SLO |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| native-common-free | 2.0 | Per-request equal-output replay | 21,170.22 | 100.65 | 1.91 | 99.32 | 99.00% | 99.00% | 98.00% |
| native-common-free | 2.0 | Per-request equal-output replay | 21,158.74 | 100.60 | 1.91 | 93.27 | 98.00% | 98.00% | 96.00% |
| native-common-free | 2.0 | Per-request equal-output replay | 21,054.24 | 100.10 | 1.90 | 92.16 | 98.00% | 98.00% | 96.00% |
| idle10-B1 | 2.0 | Per-request equal-output replay | 20,837.47 | 99.07 | 1.88 | 92.58 | 94.00% | 98.00% | 92.00% |
| idle10-B1 | 2.0 | Per-request equal-output replay | 20,921.51 | 99.47 | 1.89 | 96.09 | 96.00% | 99.00% | 95.00% |
| idle10-B1 | 2.0 | Per-request equal-output replay | 20,678.48 | 98.31 | 1.87 | 87.40 | 92.00% | 100.00% | 92.00% |
| native-common-B1 | 2.0 | Per-request equal-output replay | 20,899.94 | 99.37 | 1.89 | 96.33 | 94.00% | 99.00% | 93.00% |
| native-common-B1 | 2.0 | Per-request equal-output replay | 20,688.76 | 98.36 | 1.87 | 80.57 | 87.00% | 97.00% | 85.00% |
| native-common-B1 | 2.0 | Per-request equal-output replay | 20,811.56 | 98.95 | 1.88 | 89.69 | 92.00% | 99.00% | 91.00% |
| native-common-B1 | 2.0 | Per-request equal-output replay | 20,757.05 | 98.69 | 1.87 | 88.63 | 93.00% | 100.00% | 93.00% |
| static10-B1 | 2.0 | Per-request equal-output replay | 20,326.69 | 96.64 | 1.83 | 70.05 | 72.00% | 98.00% | 70.00% |
| static10-B1 | 2.0 | Per-request equal-output replay | 20,589.75 | 97.89 | 1.86 | 85.48 | 87.00% | 99.00% | 86.00% |
| static10-B1 | 2.0 | Per-request equal-output replay | 20,351.52 | 96.76 | 1.84 | 80.21 | 86.00% | 100.00% | 86.00% |
| native-strong-B1 | 2.0 | Per-request equal-output replay | 20,874.24 | 99.25 | 1.88 | 93.97 | 96.00% | 99.00% | 95.00% |
| native-strong-B1 | 2.0 | Per-request equal-output replay | 20,799.24 | 98.89 | 1.88 | 87.67 | 90.00% | 100.00% | 90.00% |
| native-strong-B1 | 2.0 | Per-request equal-output replay | 20,912.17 | 99.43 | 1.89 | 97.39 | 94.00% | 100.00% | 94.00% |

NativeCommon Unrestricted에서 NativeCommon-B1으로 바꾸자 ITL p99가 11.321→6.562ms였다. Static10-B1의 5.827ms와 비교한 절대 격차는 5.494→0.735ms였다. 부호 있는 Native−Static 차이는 +5.494→+0.735ms였다. 이는 Counter를 켠 독립 N1의 방향성 근거다. 인과 기여율이나 적격 C3 우위를 확정하지 않는다.


| Run | Cohort | Evidence | 원인 |
| --- | --- | --- | --- |
| b_native_common | 초기화 | FAILED_EXECUTION | HTTP Error 500: Internal Server Error |

원시 근거는 각 Run의 `analysis/*_batch.json`, 그 Receipt가 가리키는 `histograms.json`, `raw/proxy_epd.jsonl`에 있다. `actual_d_batch.csv`와 `cohorts.csv`는 재집계 자료다.

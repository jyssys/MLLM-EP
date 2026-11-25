# H200 Prefill-heavy 탐색과 계측

현재 Motivation 판정은 `P_HEAVY_D_IDLE_NOT_MEASURED`다. Capacity Knee 탐색은 실제 E/P/D Online Main100에서 수행하며, 적격 P Kernel Busy·비교 가능한 독립 P Capacity·실제 P Backlog·D Quiet 구간·Decode SLO를 함께 확인한다. E가 바쁜 구간은 포함한다.

## A. QPS2 Baseline 복구

### 요청 지연과 작업량

| Method | QPS | 출력 계약 | Run/Cohort | TTFT mean ms | TTFT p50 ms | TTFT p95 ms | Mean TBT ms | ITL p99 ms | TPOT p95 ms | 출력 토큰 | Evidence |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| native-common-free | 2.0 | Natural EOS | `a_common/native_common_free_2` | 2,592.29 | 2,175.35 | 5,230.94 | 4.80 | 11.95 | 5.26 | 5468 | MEASURED_ONLINE |
| native-strong-free | 2.0 | Natural EOS | `a_strong_v2/native_strong_free_2` | 2,268.76 | 2,174.25 | 4,587.53 | 4.79 | 11.79 | 5.11 | 5523 | MEASURED_ONLINE |

### 처리량과 SLO

| Method | QPS | 출력 계약 | 입력+출력 tok/s | 출력 tok/s | 완료 req/s | SLO-goodput tok/s | TTFT Pass | Decode Pass | Joint SLO |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| native-common-free | 2.0 | Natural EOS | 20,584.58 | 101.55 | 1.86 | 75.71 | 93.00% | 95.00% | 88.00% |
| native-strong-free | 2.0 | Natural EOS | 21,161.27 | 105.44 | 1.91 | 96.25 | 99.00% | 96.00% | 95.00% |

Setup Correctness와 이번 실제 open-loop Online 구간은 별개다. H100에서 재현한 수치와 직접 알고리즘 Speedup을 계산하지 않는다. `MEASURED_ONLINE`은 개별 100요청 Cohort를 뜻한다. H200 NativeStrong/Main100은 별도 Workload로 유지한다.


| Run | Cohort | Evidence | 원인 |
| --- | --- | --- | --- |
| a_strong | 초기화 | FAILED_EXECUTION | ../mllm_epd_runtime_env/bin/python: undefined symbol: pidfd_open |
| a_strong_retry | 초기화 | FAILED_EXECUTION | [Errno 2] No such file or directory: '/home/esjung/MLLM-EP-github/poc_flashvep/reports/mllm_epd_prefill_heavy_token_split_20261124/state/model_download_complete.json' |

## C–E. QPS 탐색 및 계측

| Method | QPS | Run/Cohort | E Busy | P Busy | D Busy | P req/s | P tok/s | P HTTP 존재 | D HTTP/대기 없음 | TTFT p95 ms | ITL p99 ms | TPOT p95 ms | Evidence |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| native-common-B1 | 0.5 | `c_native_b1_screen/native_common_b1_0p5` | — | — | — | 0.54 | 5,681.27 | 23.48% | 86.35% | 2,305.48 | 5.57 | 4.78 | MEASURED_ONLINE |
| native-common-B1 | 1.0 | `c_native_b1_screen/native_common_b1_1` | — | — | — | 1.00 | 10,612.28 | 44.13% | 73.54% | 2,765.55 | 5.67 | 4.78 | MEASURED_ONLINE |
| native-common-B1 | 2.0 | `c_native_b1_screen/native_common_b1_2` | — | — | — | 1.71 | 17,993.31 | 68.27% | 55.39% | 5,314.72 | 6.17 | 4.79 | MEASURED_ONLINE |
| native-common-B1 | 3.0 | `c_native_b1_screen/native_common_b1_3` | — | — | — | 2.18 | 24,449.31 | 86.49% | 30.16% | 10,768.26 | 6.00 | 4.81 | MEASURED_ONLINE |
| native-common-B1 | 3.0 | `d_profile_b1_q3/native_common_b1_3_profile` | 58.91% | 62.22% | 60.35% | 2.08 | 23,353.63 | 83.89% | 29.06% | 10,160.71 | 6.57 | 4.98 | DIAGNOSTIC |
| native-common-B1 | 4.0 | `c_native_b1_screen/native_common_b1_4` | — | — | — | 2.91 | 29,526.38 | 94.76% | 0.00% | 12,022.97 | 6.35 | 4.79 | MEASURED_ONLINE |
| native-common-B1 | 4.0 | `d_profile_b1_q4_sampled/native_common_b1_4_profile` | 73.33% | 79.71% | 86.28% | 3.06 | 32,126.07 | 94.71% | 0.46% | 11,359.43 | 7.28 | 5.16 | DIAGNOSTIC |
| native-common-B1 | 5.0 | `c_native_b1_screen/native_common_b1_5` | — | — | — | 3.55 | 36,542.29 | 100.00% | 9.05% | 13,095.61 | 6.14 | 4.77 | MEASURED_ONLINE |
| native-common-B1 | 5.0 | `d_profile_b1_q5_deferred/native_common_b1_5_profile` | 88.02% | 96.40% | 81.03% | 3.55 | 36,542.29 | 100.00% | 7.80% | 13,105.07 | 6.68 | 5.18 | DIAGNOSTIC |
| native-common-B1 | 5.0 | `d_profile_b1_q5_sampled/native_common_b1_5_profile` | 86.08% | 96.83% | 84.02% | 3.64 | 36,839.15 | 100.00% | 4.69% | 13,543.83 | 6.82 | 5.31 | DIAGNOSTIC |
| native-common-B1 | 6.0 | `c_native_b1_screen/native_common_b1_6` | — | — | — | 3.39 | 33,072.78 | 100.00% | 14.86% | 16,259.79 | 6.25 | 4.81 | MEASURED_ONLINE |
| native-common-B1 | 6.0 | `d_profile_b1_q6_sampled/native_common_b1_6_profile` | 97.59% | 93.81% | 79.44% | 3.50 | 36,235.72 | 100.00% | 9.30% | 16,178.84 | 6.98 | 5.17 | DIAGNOSTIC |
| native-common-B1 | 8.0 | `c_native_b1_screen/native_common_b1_8` | — | — | — | 3.79 | 37,471.70 | 99.15% | 2.44% | 19,850.20 | 6.36 | 4.81 | MEASURED_ONLINE |
| native-common-free | 1.0 | `c_native_free_screen/native_common_free_1` | — | — | — | 1.00 | 10,612.28 | 43.69% | 76.31% | 2,693.32 | 10.10 | 5.21 | MEASURED_ONLINE |
| native-common-free | 3.0 | `c_native_free_screen/native_common_free_3` | — | — | — | 2.13 | 23,618.60 | 82.65% | 48.76% | 9,520.12 | 12.62 | 5.11 | MEASURED_ONLINE |
| native-common-free | 5.0 | `c_native_free_screen/native_common_free_5` | — | — | — | 3.37 | 34,548.84 | 100.00% | 29.80% | 12,550.78 | 14.66 | 5.26 | MEASURED_ONLINE |
| native-common-free | 5.0 | `d_profile_free_q5_deferred/native_common_free_5_profile` | 88.16% | 93.86% | 60.25% | 3.64 | 36,697.50 | 100.00% | 28.76% | 12,321.46 | 13.68 | 5.65 | DIAGNOSTIC |
| native-common-free | 5.0 | `d_profile_free_q5_sampled/native_common_free_5_profile` | 87.05% | 92.95% | 61.16% | 3.55 | 36,484.30 | 100.00% | 28.25% | 11,931.25 | 14.74 | 5.73 | DIAGNOSTIC |
| native-common-free | 8.0 | `c_native_free_screen/native_common_free_8` | — | — | — | 3.50 | 32,045.67 | 100.00% | 30.67% | 18,883.88 | 13.10 | 5.16 | MEASURED_ONLINE |

### 처리량과 SLO

| Method | QPS | 출력 계약 | 입력+출력 tok/s | 출력 tok/s | 완료 req/s | SLO-goodput tok/s | TTFT Pass | Decode Pass | Joint SLO |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| native-common-B1 | 0.5 | Natural EOS | 5,765.54 | 29.06 | 0.52 | 25.66 | 100.00% | 99.00% | 99.00% |
| native-common-B1 | 1.0 | Natural EOS | 11,329.35 | 57.10 | 1.02 | 57.10 | 100.00% | 100.00% | 100.00% |
| native-common-B1 | 2.0 | Natural EOS | 20,632.30 | 103.65 | 1.86 | 87.29 | 88.00% | 97.00% | 86.00% |
| native-common-B1 | 3.0 | Natural EOS | 25,401.97 | 127.61 | 2.29 | 83.23 | 63.00% | 99.00% | 63.00% |
| native-common-B1 | 4.0 | Natural EOS | 29,717.39 | 149.29 | 2.68 | 90.96 | 63.00% | 99.00% | 62.00% |
| native-common-B1 | 5.0 | Natural EOS | 32,368.34 | 163.77 | 2.92 | 42.13 | 17.00% | 96.00% | 17.00% |
| native-common-B1 | 6.0 | Natural EOS | 32,750.08 | 163.32 | 2.95 | 45.17 | 18.00% | 97.00% | 18.00% |
| native-common-B1 | 8.0 | Natural EOS | 32,427.45 | 164.42 | 2.93 | 38.76 | 15.00% | 94.00% | 14.00% |
| native-common-free | 1.0 | Natural EOS | 11,313.85 | 57.03 | 1.02 | 57.03 | 100.00% | 100.00% | 100.00% |
| native-common-free | 3.0 | Natural EOS | 25,756.93 | 130.97 | 2.32 | 86.15 | 63.00% | 98.00% | 62.00% |
| native-common-free | 5.0 | Natural EOS | 33,166.79 | 156.77 | 2.99 | 88.20 | 56.00% | 98.00% | 55.00% |
| native-common-free | 8.0 | Natural EOS | 32,540.25 | 161.84 | 2.94 | 45.00 | 34.00% | 95.00% | 33.00% |

P HTTP는 Host의 진행 중 요청 구간이며 P Backlog나 Kernel Busy 범위를 과대 대표할 수 있다. D HTTP와 P-ready 대기가 없는 구간도 Suffix 안전성의 보수적 Proxy다. Kernel·Scheduler 관측이 적격인 Profile만 P-heavy 결론을 지지한다. Capture는 p20–p80 예정 Arrival과 실제 공통 ROI의 교집합이다.

| Run | Gate | Collector | 실패한 Metric과 간섭 |
| --- | --- | --- | --- |
| d_profile_b1_q3 | DIAGNOSTIC | 통과 | ITL_p99_ms (spread 10.03%, deviation 4.75%) |
| d_profile_b1_q4_sampled | DIAGNOSTIC | 통과 | TTFT_p50_ms (spread 0.10%, deviation 14.48%); ITL_p99_ms (spread 0.58%, deviation 13.31%); TPOT_p95_ms (spread 1.57%, deviation 7.40%) |
| d_profile_b1_q5_deferred | DIAGNOSTIC | 통과 | TPOT_p95_ms (spread 1.03%, deviation 7.12%) |
| d_profile_b1_q5_sampled | DIAGNOSTIC | 통과 | TTFT_p50_ms (spread 14.28%, deviation 10.01%); TTFT_p95_ms (spread 10.20%, deviation 0.38%); ITL_p99_ms (spread 7.33%, deviation 2.11%); TPOT_p95_ms (spread 0.65%, deviation 10.23%) |
| d_profile_b1_q5_sustained_r3 | DIAGNOSTIC | 실패 | — |
| d_profile_b1_q5_sustained_r3_buffer2 | DIAGNOSTIC | 통과 | ITL_p99_ms (spread 5.52%, deviation 4.10%) |
| d_profile_b1_q6_sampled | DIAGNOSTIC | 통과 | TTFT_p50_ms (spread 10.24%, deviation 2.47%); TPOT_p95_ms (spread 0.38%, deviation 7.43%) |
| d_profile_free_q5_deferred | DIAGNOSTIC | 통과 | TTFT_p50_ms (spread 7.97%, deviation 8.98%); TPOT_p95_ms (spread 2.60%, deviation 8.05%) |
| d_profile_free_q5_sampled | DIAGNOSTIC | 통과 | TTFT_p50_ms (spread 5.33%, deviation 14.17%); ITL_p99_ms (spread 4.95%, deviation 5.09%); TPOT_p95_ms (spread 1.04%, deviation 8.09%) |

| Receipt | D 정책 | Main100 반복 | 판정 | P Busy | P req/s Proximity | P tok/s Proximity | P 최근 Waiting | D Decode SLO | D Quiet ≥160ms의 P Busy Coverage | D Quiet ≥225ms의 P Busy Coverage |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| discovery_gate_free_q5.json | Unrestricted | 1.00 | P_HEAVY_D_IDLE_NOT_MEASURED | 92.95% | 111.76% | 104.13% | 86.04% | 95.00% | 27.22% | 21.63% |
| discovery_gate_free_q5_deferred.json | Unrestricted | 1.00 | P_HEAVY_D_IDLE_NOT_MEASURED | 93.86% | 114.63% | 104.74% | 87.06% | 95.00% | 26.22% | 24.22% |
| discovery_gate_q3.json | Batch1 | 1.00 | P_HEAVY_D_IDLE_NOT_MEASURED | 62.22% | 65.34% | 66.66% | 13.11% | 99.00% | 6.99% | 6.82% |
| discovery_gate_q4.json | Batch1 | 1.00 | P_HEAVY_D_IDLE_NOT_MEASURED | 79.71% | 96.29% | 91.69% | 60.75% | 95.00% | 0.00% | 0.00% |
| discovery_gate_q5.json | Batch1 | 1.00 | P_HEAVY_D_IDLE_NOT_MEASURED | 96.83% | 114.63% | 105.14% | 92.47% | 95.00% | 4.72% | 4.72% |
| discovery_gate_q5_deferred.json | Batch1 | 1.00 | P_HEAVY_D_IDLE_NOT_MEASURED | 96.40% | 111.76% | 104.30% | 100.00% | 97.00% | 7.32% | 7.32% |
| discovery_gate_q5_sustained_r3.json | Batch1 | 3.00 | P_HEAVY_D_IDLE_NOT_MEASURED | 95.41% | 104.12% | 103.82% | 87.08% | 95.67% | 0.88% | 0.88% |
| discovery_gate_q5_sustained_r3_buffer2.json | Batch1 | 3.00 | P_HEAVY_D_IDLE_NOT_MEASURED | 95.08% | 105.10% | 103.05% | 94.34% | 95.00% | 1.64% | 1.17% |
| discovery_gate_q6.json | Batch1 | 1.00 | P_HEAVY_D_IDLE_NOT_MEASURED | 93.81% | 110.05% | 103.42% | 94.12% | 96.00% | 7.99% | 7.99% |

- `discovery_gate_free_q5.json`은 `P_HEAVY_D_IDLE_NOT_MEASURED`다. 미통과 적용 항목은 trace_qualified이다.
- `discovery_gate_free_q5_deferred.json`은 `P_HEAVY_D_IDLE_NOT_MEASURED`다. 미통과 적용 항목은 trace_qualified이다.
- `discovery_gate_q3.json`은 `P_HEAVY_D_IDLE_NOT_MEASURED`다. 미통과 적용 항목은 trace_qualified, p_kernel_busy, p_service_proximity, p_not_starved, d_quiet_window_coverage이다.
- `discovery_gate_q4.json`은 `P_HEAVY_D_IDLE_NOT_MEASURED`다. 미통과 적용 항목은 trace_qualified, p_kernel_busy, d_quiet_window_coverage이다.
- `discovery_gate_q5.json`은 `P_HEAVY_D_IDLE_NOT_MEASURED`다. 미통과 적용 항목은 trace_qualified, d_quiet_window_coverage이다.
- `discovery_gate_q5_deferred.json`은 `P_HEAVY_D_IDLE_NOT_MEASURED`다. 미통과 적용 항목은 trace_qualified, d_quiet_window_coverage이다.
- `discovery_gate_q5_sustained_r3.json`은 `P_HEAVY_D_IDLE_NOT_MEASURED`다. 미통과 적용 항목은 trace_qualified, d_quiet_window_coverage이다.
- `discovery_gate_q5_sustained_r3_buffer2.json`은 `P_HEAVY_D_IDLE_NOT_MEASURED`다. 미통과 적용 항목은 trace_qualified, d_quiet_window_coverage이다.
- `discovery_gate_q6.json`은 `P_HEAVY_D_IDLE_NOT_MEASURED`다. 미통과 적용 항목은 trace_qualified, d_quiet_window_coverage이다.


| Run | Cohort | Evidence | 원인 |
| --- | --- | --- | --- |
| c_capacity | 초기화 | FAILED_EXECUTION | D-only replay failed: RuntimeError: {"error":{"message":"Handoff belongs to another input contract","type":"InternalServerError","param":null,"code":500}} |

`cohorts.csv`, `kernel_busy.csv`, `kernel_overlap.csv`, 실제 Trace와 `analysis/*_profile_gate.json`이 재현 자료다.

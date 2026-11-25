# H200 Idle-aware Token Split Online Serving

Phase G는 `NOT_RUN`이다. 기존 Idle10 정책에서는 계획 시 P HTTP가 진행 중이고 D HTTP가 비었을 때만 Split한다. 조건이 충족되지 않으면 Native 경로를 사용하며 Prefix 완료 후 보호 조건을 다시 확인한다.

| Run/Cohort | Step | Split 실행 건 | 원래 Native 실행 건 | 이전 Tokens | 전체 Offload | D Wait p95 ms | D at P-ready no pending |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `b_idle10/idle10_b1_2_batch_clean_after` | Phase B | 36 | 64 | 40151 | 3.64% | 1,361.36 | 39.00% |
| `b_idle10/idle10_b1_2_batch_clean_before` | Phase B | 35 | 65 | 37018 | 3.36% | 1,533.57 | 41.00% |
| `b_idle10/idle10_b1_2_observed` | Phase B | 38 | 62 | 38635 | 3.50% | 1,712.04 | 43.00% |

### 요청 지연과 작업량

| Method | QPS | 출력 계약 | Run/Cohort | TTFT mean ms | TTFT p50 ms | TTFT p95 ms | Mean TBT ms | ITL p99 ms | TPOT p95 ms | 출력 토큰 | Evidence |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| idle10-B1 | 2.0 | Per-request equal-output replay | `b_idle10/idle10_b1_2_batch_clean_after` | 2,475.71 | 2,084.52 | 5,052.37 | 4.66 | 6.33 | 4.90 | 5269 | MEASURED_ONLINE |
| idle10-B1 | 2.0 | Per-request equal-output replay | `b_idle10/idle10_b1_2_batch_clean_before` | 2,513.48 | 2,179.80 | 4,871.38 | 4.64 | 6.09 | 4.81 | 5269 | MEASURED_ONLINE |
| idle10-B1 | 2.0 | Per-request equal-output replay | `b_idle10/idle10_b1_2_observed` | 2,658.85 | 2,144.68 | 5,377.93 | 4.64 | 6.18 | 4.79 | 5269 | DIAGNOSTIC |

### 처리량과 SLO

| Method | QPS | 출력 계약 | 입력+출력 tok/s | 출력 tok/s | 완료 req/s | SLO-goodput tok/s | TTFT Pass | Decode Pass | Joint SLO |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| idle10-B1 | 2.0 | Per-request equal-output replay | 20,837.47 | 99.07 | 1.88 | 92.58 | 94.00% | 98.00% | 92.00% |
| idle10-B1 | 2.0 | Per-request equal-output replay | 20,921.51 | 99.47 | 1.89 | 96.09 | 96.00% | 99.00% | 95.00% |
| idle10-B1 | 2.0 | Per-request equal-output replay | 20,678.48 | 98.31 | 1.87 | 87.40 | 92.00% | 100.00% | 92.00% |

낮은 Split 실행률은 전체 효과 판단에 포함한다. Phase B의 Counter QPS2 실험만으로 사전 확인된 P-heavy에서 Static10 대비 추가 개선을 결론 내리지 않는다. 개선 정책과 미래 Idle 예측은 확인된 Prefill-heavy의 Static 결과 이후에만 평가한다.

# Heterogeneous/Bursty Workload의 Discovery 결과

## 범위

아래 결과는 각 방법의 단일 Finite-cohort Discovery다.
독립 확인이나 Sustainable Capacity가 아니다.
모든 방법은 GPU0/1의 같은 Qwen3-8B Revision을 사용했다.
Revision은 `b968826d9c46dd6066d109eabc6255188de91218`이다.
원본 BF16, Input8000, Full36 Decode와 42 GiB KV/GPU를 유지했다.

PD 방법은 Generic Async Decode를 사용했다.
Dynamic은 Direct Deadline Predicate, 기존 550 ms Margin과 Queue Recovery를 사용했다.
이 표는 Canonical KV/v5 모델의 최종 성능이 아니다.
Colocated는 정확성을 통과한 표준 Continuous Full Graph B512/S16이다.
동일 Trace 안에서는 모든 방법에 같은 Prompt, Arrival과 Output Cap을 적용했다.
결과를 좋게 보이게 하려고 Colocated의 Mixed Batching을 해제하지 않았다.

## Bursty Arrival — MEASURED

N256, Output16, 명목 평균 QPS 2.0이다.
40초 주기에서 8초는 3배 Rate를 사용한다.
나머지 32초는 0.5배 Rate를 사용한다.
실제 Generator Seed는 2026111461이다.
이는 Controlled Piecewise Poisson이다.
실제 Production Trace로 표현하지 않는다.

| 방법 | TTFT p95 (ms) | ITL p99 (ms) | TPOT p95 (ms) | Raw Output tokens/s | Decode 유효 tokens/s | Joint Pass |
|---|---:|---:|---:|---:|---:|---:|
| Native | 7006.447 | 24.669 | 22.984 | 32.576917 | 32.576917 | 256/256 |
| Static Last8 | 5072.292 | 25.361 | 24.118 | 32.682705 | 32.682705 | 256/256 |
| Static Last12 | 4023.320 | 26.461 | 24.186 | 32.716848 | 32.716848 | 256/256 |
| Dynamic | 4346.065 | 25.420 | 24.171 | 32.698366 | 32.698366 | 256/256 |
| Strong Colocated | 1884.683 | 26.731 | 24.150 | 32.974367 | 32.845561 | 255/256 |

Dynamic은 Native보다 낮은 TTFT p95를 기록했다.
그러나 Static Last12보다 TTFT p95가 높았다.
Colocated의 TTFT p95는 이 다섯 방법 중 가장 낮았다.
Dynamic의 독립적인 Bursty 이득은 이 실행에서 확인되지 않았다.
한 번의 순위 차이를 통계적 우위로 표현하지 않는다.

## Mixed Output — MEASURED

N128, 명목 QPS 1.0의 Poisson Arrival을 사용했다.
Arrival Seed는 2026111421이다.
Output Cap Generator Seed는 2026111461이다.
사전 확률은 Output16/64/200에 각각 0.6/0.3/0.1이다.
실제 표본은 81/36/11개 요청이다.
전체 출력 Work는 5800 Tokens다.
Natural EOS 분포가 아니라 고정 Cap의 통제된 혼합이다.

| 방법 | TTFT p95 (ms) | ITL p99 (ms) | TPOT p95 (ms) | Raw Output tokens/s | Decode 유효 tokens/s | Joint Pass |
|---|---:|---:|---:|---:|---:|---:|
| Native | 4480.575 | 24.073 | 23.666 | 48.059302 | 48.059302 | 128/128 |
| Static Last8 | 4984.265 | 24.339 | 23.969 | 47.851133 | 47.851133 | 128/128 |
| Static Last12 | 5048.114 | 24.994 | 24.643 | 47.729809 | 47.466472 | 126/128 |
| Dynamic | 4609.898 | 24.487 | 23.652 | 48.112750 | 46.453689 | 127/128 |
| Strong Colocated | 446.111 | 25.734 | 24.186 | 48.180222 | 48.047311 | 127/128 |

Dynamic과 Colocated의 Request Joint Pass는 같다.
그러나 Qualified Output tokens/s는 같지 않다.
긴 Output 요청의 실패는 더 많은 Token을 유효 출력에서 제외한다.
따라서 Request Attainment만으로 Primary Metric을 대체하지 않는다.
Raw Throughput의 작은 차이도 독립 확인 없이 정책 우위로 해석하지 않는다.

모든 숫자의 근거는 Evidence의 `analysis/async_updated_*_metrics.json`이다.
동일 이름의 Config와 Raw Request Record를 함께 보존한다.
[Controlled Figure](dense_ar_slo_throughput_dynamic_layer_pd_20261114/figures/final_7_bursty_and_mixed_output.svg)는 이 단일 Start들을 표시한다.

## 이 탐색으로 입증하지 못한 것

현재 표는 Dynamic의 독립적 우위를 보이지 않는다.
같은 Migration Work와 Group Geometry를 통제한 Heterogeneous 확인도 `NOT_RUN`이다.
따라서 전체 Workload 공간에서 Dynamic이 불필요하다고 결론 내리지 않는다.

2048/4096/8192/12288 길이의 State 및 단독 비용은 별도로 검토했다.
그러나 새 길이를 섞은 Online Serving의 Cost/Context/Group 계약은 충분히 검증하지 못했다.
그 결과는 `CORRECTNESS_ONLY` 또는 `STRUCTURAL_ACCOUNTING`과 구분한다.
이를 Heterogeneous-input Serving 성공으로 집계하지 않는다.

실제 Production Arrival Trace와 Natural EOS 확인은 `NOT_RUN`이다.
후속 개발보다 Strong Baseline과 독립 Capacity 확인을 우선했다.
Physical D Idle과 Spatial Compute Slack은 이 탐색에서도 `NOT_MEASURED`다.
Queue 또는 Budget의 변화만으로 GPU Idle을 추정하지 않는다.

최종 동결 방법의 확인 결과는 [최종 요약](dense_ar_slo_throughput_dynamic_layer_pd_final_summary.md)에 기록한다.

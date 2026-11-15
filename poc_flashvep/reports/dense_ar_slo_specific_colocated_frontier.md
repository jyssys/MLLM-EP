# SLO별 Strong Continuous Colocated Frontier

## 기준선 계약

GPU0/1 각각에 전체 Qwen3-8B Replica를 배치한다.
표준 vLLM Continuous Batching과 Async Scheduling을 사용한다.
Custom Cohort Scheduler로 Colocated를 대체하지 않는다.
원본 BF16, 정확한 Attention과 동일한 Output 기준을 유지한다.
Revision은 `b968826d9c46dd6066d109eabc6255188de91218`이다.
GPU당 KV Budget은 42 GiB다.
Prefix Caching은 비활성이다.

SLO Threshold는 이전 독립 Calibration의 값을 유지한다.
파일 SHA256은 `c8aa94213e4f4ce54821d01ba985a8e9e88a20566a3e8d1f3cc8060d1b3926d9`이다.
Input8000/Output16의 기준은 다음과 같다.

| Regime | TTFT | 개별 ITL | Request TPOT |
|---|---:|---:|---:|
| TTFT-priority | 2000 ms | 54 ms | 52 ms |
| Balanced | 5000 ms | 34 ms | 33 ms |
| Decode-priority | 10000 ms | 30 ms | 29 ms |

요청별 모든 조건을 충족해야 Joint Pass다.
첫 ITL을 포함한다.
ITL p99만 통과한 요청을 자동으로 Joint Pass로 처리하지 않는다.
95% Joint Attainment는 연구용 사전 기준이다.
Production Standard로 주장하지 않는다.

## Numerical Qualification

기존의 Batch-dependent RMSNorm 문제를 다시 검토했다.
GPU Phase Mask는 원래 Prefill/Decode의 Reduction 경로를 유지한다.
Prefill과 Decode가 섞인 Full CUDA Graph에서도 해당 구분을 유지한다.
작은 BF16 GPU 비교 396개는 Byte Equality를 통과했다.
N32 전체 Prompt KV, Final Hidden과 Greedy Output도 통과했다.
해당 시험에서 실제 Mixed Graph Replay는 472회였다.

기본 Compile3 경로는 32개 중 13개 Output이 달랐다.
그 결과는 적격 성능으로 사용하지 않는다.
Original Norm을 Opaque Operation으로 유지한 Compile3 경로는 별도 정합성을 통과했다.
그러나 Mode0보다 명확한 Serving 이득은 확인하지 못했다.
Compiler 내부의 정확한 수치 변화 원인은 `NOT_IDENTIFIED`다.
Opaque 경로의 통과가 Compiler 원인 전체를 분리했다는 뜻은 아니다.

Strong Colocated는 Mixed Full Graph를 지원한다.
Decode-only Graph나 Sequence Limit 1의 초기 결과를 최종 Strong 기준선으로 사용하지 않는다.
실행별 Config와 Qualification Gate는 Evidence에 보존한다.

## 목적별 설정과 600초 Discovery — MEASURED

Warm-up은 60초다.
측정은 600초다.
측정 이후에도 90초 동안 Arrival을 계속한다.
아래 수치는 단일 Discovery Start다.
독립 확인의 평균이나 정확한 전역 최대는 아니다.

| 목적과 설정 | Offered / Actual QPS | Raw Output tokens/s | 목적별 Qualified tokens/s | Joint Attainment | 안정성 및 SLO |
|---|---:|---:|---:|---:|---|
| Balanced, B512/S16, FULL, QPS5.0 | 5.0 / 4.946667 | 79.333333 | 79.066667 | 99.663073% | PASS |
| Decode, 같은 B512/S16 실행 | 5.0 / 4.946667 | 79.333333 | 79.040000 | 99.629380% | PASS |
| Decode, B512/S16, FULL, QPS5.2, RR | 5.2 / 5.366667 | 84.960000 | 83.973333 | 98.850932% | Queue 기준 FAIL |
| Raw, B8192/S32, Decode-only Graph | 6.4 / 6.590000 | 105.920000 | 0.826667의 Decode Goodput | 0.784016%의 Decode Joint | Raw Queue 기준 PASS, SLO FAIL |

B는 Token Budget이다.
S는 Replica별 Sequence Limit이다.
같은 2GPU와 원본 모델을 사용했다.
Raw 목적의 설정은 Decode-safe 설정이 아니다.
Raw 결과를 SLO-qualified Capacity로 표현하지 않는다.

QPS5.0 Full Graph 실행의 ITL p99는 26.925467 ms였다.
TPOT p95는 24.178399 ms였다.
Raw 목적 실행의 ITL p99는 299.659880 ms였다.
TPOT p95는 296.611123 ms였다.
큰 Token Budget의 Raw Throughput 이득과 Decode 간섭을 구분해야 한다.
두 설정은 Sequence Limit과 Graph Mode도 다르다.
따라서 299.659880 ms를 Token Budget 하나의 인과 효과로 분해하지 않는다.

QPS5.2 RR에서는 Outstanding Request가 6개에서 40개로 증가했다.
Queue Slope는 0.025679 requests/s였다.
Joint Attainment가 95%를 넘더라도 안정성 기준을 사후 완화하지 않는다.

근거는 다음 파일이다.

- [B512/S16 QPS5.0](dense_ar_slo_throughput_dynamic_layer_pd_20261114/analysis/rate_Colocated_B512_S16_ASYNC_FULL_O16_q5.0_s2026111454_W600.0_full_graph_midpoint_discovery_v1_sustainable_v2.json).
- [B512/S16 QPS5.2 RR](dense_ar_slo_throughput_dynamic_layer_pd_20261114/analysis/boundary_C_FULL_B512_S16_ROUND_ROBIN_O16_q5.2_s2026111459_W600_v1_sustainable_v2.json).
- [B8192/S32 Raw](dense_ar_slo_throughput_dynamic_layer_pd_20261114/analysis/objective_RAW_B8192_S32_O16_q6.4_s2026111456_W600_v1_sustainable_v2.json).

## 비교 해석

Balanced와 Decode-priority를 독립적으로 평가했다.
이번 Discovery에서는 같은 B512/S16 Full Graph 설정이 두 목적에서 강했다.
Raw 목적의 강한 설정은 달랐다.
한 설정을 모든 SLO의 최선이라고 가정하지 않는다.

이전의 Dynamic/Colocated Goodput 1.735×를 이번 결론으로 재사용하지 않는다.
이제는 Decode-priority에 맞춰 강화한 Colocated와 비교해야 한다.
최종 설정과 독립 확인 결과는 [최종 요약](dense_ar_slo_throughput_dynamic_layer_pd_final_summary.md)에 기록한다.
실패한 Tuning Run도 삭제하지 않는다.

전체 Config 탐색과 최종 선택은 서로 다른 증거다.
새 Generic KV Layout은 Colocated에서도 정합성과 A/B/A를 별도로 검사한다.
그 변경을 Dynamic만의 이득으로 계산하지 않는다.

## 최종 독립 확인

선택한 Colocated QPS5.0은 새 Prompt Pool의 독립 Start 3회 모두 Decode-priority 적격이었다.
Qualified Output 평균은 80.071111 tokens/s였다.
Start 단위 경험적 95% CI는 [78.997778, 80.898667]이다.
별도 1200초 Persistence도 80.12 Qualified tokens/s로 통과했다.

같은 설정의 QPS5.0은 Balanced에서 1/3만 적격이었다.
이를 Balanced 최대 Capacity라고 주장하지 않는다.
Dynamic의 확인된 낮은 QPS4.4는 평균 71.182222 Qualified tokens/s였다.
서로 다른 Arrival Seed를 따로 재표본화한 비율 CI도 Colocated 우위를 보였다.
전체 표와 한계는 [최종 요약](dense_ar_slo_throughput_dynamic_layer_pd_final_summary.md)에 있다.

# Dynamic과 Static의 동일 Migration Budget 비교

## 비교 범위

이 문서의 첫 비교는 Discovery N128 한 Start의 실측이다.
모델은 Qwen3-8B Revision `b968826d9c46dd6066d109eabc6255188de91218`이다.
Input8000/Output16, Offered QPS 4.8과 같은 Prompt/Arrival Trace를 사용했다.
GPU0/1, BF16, Full36 Decode와 42 GiB KV Budget을 유지했다.
이전 캠페인의 성능을 새 실측으로 재사용하지 않았다.

Generic Async Decode와 같은 Lookup 모델을 공통으로 적용했다.
Source Physical Prefill은 최대 4 Request다.
Source Macro와 Destination의 허용 상한은 각각 8 Request다.
Ready Work에 따라 실제 Group 크기는 달라졌다.
이 비교는 이후 Canonical Block Layout의 성능으로 표현하지 않는다.

## 실측 결과 — MEASURED

| 방법 | Migrated Requests / 128 | 실제 D Prefill Layers | TTFT p95 (ms) | ITL p99 (ms) | TPOT p95 (ms) | Finite-cohort Output tokens/s |
|---|---:|---:|---:|---:|---:|---:|
| Dynamic | 112 | 1120 | 4125.743 | 25.138 | 23.766 | 70.350 |
| Static, 같은 Request Fraction | 112 | 1344 | 3927.310 | 24.421 | 24.007 | 70.829 |
| Static, 같은 Layer Work | 94 | 1120 | 4390.902 | 24.938 | 24.174 | 69.287 |
| Static, 같은 Cut Histogram | 112 | 1120 | 4429.224 | 24.714 | 23.711 | 68.778 |
| Strong Static Last12, 전체 이전 | 128 | 1536 | 3425.048 | 24.977 | 24.321 | 72.862 |

Dynamic의 실제 분포는 Last0 16건, Last8 56건, Last12 56건이었다.
같은 Request Fraction 대조군은 Last12 112건과 Last0 16건을 실행했다.
같은 Layer Work 대조군은 Last12 93건, Last4 1건, Last0 34건을 실행했다.
같은 Histogram 대조군은 Dynamic과 각 Cut의 Request 수가 같다.
Static의 Cut 배정은 Discovery 결과로 만들었다.
Held-out 실행의 미래 Decision을 입력하지 않았다.

Budget과 실제 완료 Work는 [Matched Controls](dense_ar_slo_throughput_dynamic_layer_pd_20261114/analysis/async_D_discovery_matched_controls.json)에 있다.
각 Run의 전체 지표는 같은 Evidence Directory의 `analysis/*_metrics.json`에 있다.

## 해석

Dynamic은 같은 Layer Work와 같은 Histogram 대조군보다 낮은 TTFT p95를 기록했다.
그러나 같은 Request Fraction 대조군보다 TTFT p95가 높았다.
다른 강한 Static Last12도 별도로 비교한다.
하나의 유리한 Matched Control만 선택해서 Dynamic 우위를 주장하지 않는다.

같은 총 Layer 수는 같은 실행 Geometry를 의미하지 않는다.
Dynamic은 Last8/12를 각각 8 Request Macro로 7회씩 선택했다.
초기의 Native Macro는 1/2/3/4 Request로 구성됐다.
Static 대조군의 Macro 크기와 그 시간 순서는 달랐다.
Destination Cohort 크기도 달랐다.

이는 현재 비교의 Confound다.
동일 Layer Work 결과만으로 독립적인 State-aware 정책 이득을 입증하지 못했다.
Group Geometry를 고정한 독립 반복 우위도 아직 입증하지 못했다.
세 대조군의 Output Parity와 요청별 Work Accounting은 통과했다.
정합성 통과를 정책 우위로 해석하지 않는다.

근거는 [Geometry/Transport Audit v2](dense_ar_slo_throughput_dynamic_layer_pd_20261114/analysis/geometry_transport_audit_v2.json)다.
v1의 예상 KV Byte 공식 오류는 Erratum으로 보존했다.
v2는 Layer/Token당 4096 Bytes를 사용한다.
실제 Runtime, NIXL Payload와 원래 성능값은 이 분석 수정으로 바뀌지 않았다.

## 독립 확인과 최종 판단

최종 Static/Dynamic은 Discovery에서 선택한 뒤 동결한다.
확인에서는 독립 Start를 통계 단위로 사용한다.
한 Start의 Token 수나 Request 수를 독립 반복 수로 사용하지 않는다.
Capacity가 다른 두 방법의 Decode 지연을 같은 부하의 비열등성으로 표현하지 않는다.
최종 확인 결과는 [최종 요약](dense_ar_slo_throughput_dynamic_layer_pd_final_summary.md)에 기록한다.

이 문서의 N128 속도는 Sustainable Throughput이 아니다.
이 비교만으로 Dynamic의 독립적인 Systems Contribution은 확정되지 않는다.

## 최종 Runtime의 독립 확인

동결된 Static QPS4.4와 보충 Dynamic QPS4.4는 각각 독립 Start 3회를 통과했다.
Qualified 평균은 Static 70.417778, Dynamic 71.182222 tokens/s였다.
Ratio는 1.010856, 독립 Bootstrap 95% CI는 [0.988673, 1.035469]다.
TTFT p95 Ratio는 1.185589 [0.960853, 1.491237]다.
이 비교에서도 Dynamic의 유의미한 우위는 확인되지 않았다.

두 그룹의 Arrival Seed와 실제 이전량은 다르다.
이 결과를 Matched Work 또는 동일 Geometry 비교라고 표현하지 않는다.
새 Runtime의 엄격한 Matched-budget 독립 확인은 `NOT_RUN`이다.
현재 결론은 Dynamic의 독립적 정책 기여가 `NOT_ESTABLISHED`라는 것이다.

# Output64/200의 Migration0 원인

## 결론

기존 Migration0을 물리적인 D GPU 여유 부족으로 해석할 수 없다.
두 Admission Predicate가 모든 후보를 차단했다.
`Admission-wait Limit=0`만 완화해도 이전 후보는 생기지 않았다.
Known-ingress Predicate도 함께 제거해야 Offline 후보가 생겼다.
이 결과는 안전한 Online 실행 가능량이 아니다.

분석 기준은 Commit `92e5275f1175bc08680de77180b8f5c652f2d08f`다.
모델은 Qwen3-8B Revision `b968826d9c46dd6066d109eabc6255188de91218`다.
기존 Source와 Raw Evidence는 변경하지 않았다.

## Gate Attribution — TRACE_ANALYSIS

Output별 9개 Run, 1,152개 요청을 분석했다.
각 Run은 N128이다.
최종 Choice Reason은 다음과 같다.

| 최종 Reason | Output64 / 1,152 | Output200 / 1,152 |
|---|---:|---:|
| P Compute Backlog 없음 | 429 | 751 |
| D State가 오래됨 | 15 | 65 |
| 평가한 Cut이 모두 거절됨 | 708 | 336 |

후보별 Gate를 독립적으로 다시 계산했다.
Last2/4/8/12 각각에서 Known-ingress Window Predicate가 1,152/1,152개 요청을 거절했다.
Admission-wait Predicate는 각 후보에서 1,143/1,152개 요청을 거절했다.
이 두 수치는 겹친다.
두 수치를 더해서 거절 요청 수로 사용하면 안 된다.
최종 Choice의 조기 반환과 후보별 독립 Gate 계산도 서로 다른 집계다.

전체 후보별 교차 집계는 [Gate Attribution](dense_ar_slo_throughput_dynamic_layer_pd_20261114/analysis/long_output_gate_attribution.json)에 있다.
[Replay 코드](../dense_ar_slo_layer_pd/trace_audit.py)는 원래 Decision Log와 선택 결과를 대조한다.

## Counterfactual — OFFLINE_ORACLE

원래 Trace의 상태를 고정했다.
이전 후보를 추가해도 이후 Queue와 Decode 상태를 다시 계산하지 않았다.

| Predicate 변경 | Output64 후보 요청 | Output200 후보 요청 |
|---|---:|---:|
| Wait Limit만 50/250/1,000ms로 변경 | 0 | 0 |
| Wait와 Known-ingress Predicate를 모두 제거 | 608 / 1,152 | 268 / 1,152 |

608건과 268건은 모델상 선택 가능한 후보 수다.
실제 Migration 완료 수나 Decode-safe Capacity가 아니다.
이 수치로 SLO 보호나 Throughput 개선을 주장하지 않는다.
새 정책은 기존 Predicate를 단순히 삭제하는 대신 실제 영향을 받는 요청의 Deadline을 검사한다.
기존 Decode Cohort가 실제로 종료될 때까지 D Suffix 실행을 시작하지 않는다.

## Output200의 Source KV Retention — TRACE_ANALYSIS

QPS4.8의 독립 Seed 81/82/83을 분석했다.
요청당 8,000-token Prompt는 500개 KV Block을 사용한다.
19,113개 Free Block에서는 Prompt 38개를 보유할 수 있다.
이 값은 Structural Accounting이다.

| Seed | Source-ready → D Host Admission p95 | P KV Allocation Retry | 최소 P Free Block | 최소 D Free Block |
|---|---:|---:|---:|---:|
| 81 | 14,615.548ms | 24,349 | 113 | 10,905 |
| 82 | 14,341.490ms | 24,281 | 113 | 10,905 |
| 83 | 14,544.486ms | 16,012 | 113 | 10,905 |

Source는 ACK 전에 유효한 Prompt KV를 해제할 수 없다.
D Admission 대기가 길어지면 Source KV도 오래 남는다.
113개 Free Block으로는 다음 500-block Prompt를 할당할 수 없다.
따라서 이 Run의 Queue 증가를 순수한 P Compute 병목으로 분류하면 안 된다.
Retry는 반복 시도 횟수다.
서로 다른 거절 요청 수가 아니다.

위 지연은 Retention Lifetime의 하한이다.
실제 Wire Completion과 ACK까지의 시간은 포함하지 않는다.
Exclusive Transfer Time과 물리적인 GPU Busy/Idle은 `NOT_MEASURED`다.

## 198.889ms ITL Spike — TRACE_ANALYSIS

Seed83의 q107, Token Count17→18에서 다음 관계를 확인했다.

`198.888972ms = 28.077098ms + 170.811875ms`

왼쪽은 Coordinator ITL이다.
오른쪽 첫 값은 Worker Emission Interval이다.
마지막 값은 Receiver Delivery Lag의 증가량이다.
Identity Residual은 0ms다.
Worker Interval도 Host Scheduling을 포함한다.
이를 CUDA Kernel Time으로 해석하면 안 된다.
Delivery Lag가 증가한 정확한 Host 원인은 `NOT_IDENTIFIED`다.
GC 또는 GPU Resource Contention으로 단정하지 않는다.

전체 결과는 [Root Cause Evidence](dense_ar_slo_throughput_dynamic_layer_pd_20261114/analysis/long_output_zero_migration_root_cause.json)에 있다.
분석 코드는 [long_output_analysis.py](../dense_ar_slo_layer_pd/long_output_analysis.py)다.

## 한계

이 문서는 기존 Migration0의 원인을 설명한다.
새 정책의 Serving 성능과 장기 안정성은 별도 평가한다.
Host-state Opportunity를 물리적인 GPU Idle로 표현하지 않는다.
GPU Idle 부족이 구조적인 원인이라는 결론은 아직 성립하지 않는다.

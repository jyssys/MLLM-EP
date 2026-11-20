# BI0 Credit·KV Block·Batch Sensitivity

상태: `CLOSED`. One-factor Control과 다른 실행 Family를 구분한다.

## Source Credit Intervention

Credit은 P Dispatch에서 획득한다. 첫 D Client Token에서 반환한다.
Credit은 실제 GPU Batch도, KV Reservation도, Hidden Buffer Slot도 아니다.
Credit1도 이전 Decode 완료까지 기다리는 완전한 Serial Pipeline은 아니다.

Last8, Full200, QPS2.35, Arrival Seed2083을 사용했다.
Actual Batch는1/1/1이다. Credit1/4/8/16/32에서 Native/Layer/Token을 각각 실행했다.
Layer의 Hidden Pool은 모든 Credit에서32로 유지했다.
Pool32는 긴 Context State·Memory Preflight로 선정했다. 성능 결과로 고르지 않았다.
Primary h4 Confirmation과 이 h32 Family를 합치지 않는다.
각 Cell은 독립 Start1회다. 다음 TTFT 값에는 Start-level CI가 없다.

| Source Credit | Native TTFT p95 | Layer8 TTFT p95 | Token8 TTFT p95 |
|---|---:|---:|---:|
| 1 | 48.402s | 52.702s | 46.858s |
| 4 | 29.652s | 12.072s | 9.369s |
| 8 | 29.689s | 11.898s | 9.122s |
| 16 | 29.564s | 11.949s | 9.169s |
| 32 | 30.287s | 11.867s | 9.413s |

Credit1은 두 Axis의 TTFT를 크게 늘렸다.
Credit4를32로 늘려도 이 Seed의 Token 우위는 없어지지 않았다.
이 결과는 Credit4만이 Token 우위의 원인이라는 단순 설명과 맞지 않는다.
한 Seed의 결과만으로 모든 Backpressure 효과를 배제하지 않는다.
같은 Axis에서는 모든 Credit의200개 Greedy Output이 일치했다.
Layer는 각4804 Tokens, Token은 각4788 Tokens를 생성했다.
따라서 이 Credit Sweep 안의 Timing 차이는 Output Count 변화 때문이 아니다.
두 Axis 사이의 Output 차이는 BI0 Query Geometry 계약으로 별도 관리한다.

## KV Block

KV16 성능 전에 새 N8 Full State Gate와 Token Same-geometry Teacher를 실행했다.
이후 N32 Controlled Lifetime 검사를 실행했다.
통과한 설정만 KV16/128의 제한적인 Full200 비교에 사용한다.
Main 설정은 KV128로 유지한다. 유리한 결과에 따라 Main Block을 바꾸지 않는다.
NIXL Descriptor 수와 Host API Enclosure를 기록한다.
API Enclosure를 순수 DMA 시간으로 표현하지 않는다.

Full200/Last8/Seed2089의 One-factor 결과는 다음과 같다.
각 Cell은 독립 Start1회다. Source Credit4와 Hidden Slots4를 유지했다.

| KV Block | Layer TTFT p95 | Token TTFT p95 |
|---|---:|---:|
| 16 | 11.629s | 8.939s |
| 128 | 7.593s | 4.989s |

두 Block 모두 이 Seed에서는 Token이 빨랐다.
Block128은 두 Axis의 TTFT를 줄였다. 이를 모든 Workload의 전역 최적값으로 주장하지 않는다.
별도 Historical Input8000/Output16·KV16에서 관측한 첫 Layer 승리는 독립 Held-out3회에서 확정되지 않았다.

## Actual Batch

현재 Primary는 실제 Batch1/1/1이다.
이번 캠페인의 새 Controlled Decode Batch2/4 반복은 `NOT_RUN`이다.
Historical BI1 Batch2/4 결과는 이전 실측으로만 보존한다.
C3의 정상 Continuous Batch는 별도 Bridge다. 이를 Controlled Batch2/4로 부르지 않는다.

[Credit 전체 결과](dense_ar_bi0_axis_rootcause_20261120/analysis/credit_sweep_v1.json),
[Credit 대기 분해](dense_ar_bi0_axis_rootcause_20261120/analysis/source_credit_event_decomposition_v1.json)를 참조한다.

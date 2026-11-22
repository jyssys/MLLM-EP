# LongBench400 KV, Hidden 및 Iso-work Accounting

작성일: 2026-11-21 KST.
이 문서의 수식은 `STRUCTURAL_ACCOUNTING`이다.
GPU Kernel Time 또는 DMA Time의 실측값이 아니다.

## 보존하는 작업량

입력은 L Tokens다.
Upper Layer 수는 m이다.
Token Suffix 길이는 s다.
Prefix 길이는 p=L−s다.

`P Rows = p(36−m)`

`D Rows = pm + 36s = Lm + (36−m)s`

P와 D Rows의 합은 항상 36L이다.
Hybrid의 D Rows를 `Lm+36s`로 계산하면 ms를 중복 계산한다.
그 식을 사용하지 않는다.

P의 Causal Attention Pair 수는 `(36−m)p(p+1)/2`다.
D Pair 수는 전체 `36L(L+1)/2`에서 P Pair 수를 뺀 값이다.
Rows가 같아도 Attention의 P/D 배치는 다를 수 있다.

## 유효 KV와 H/R Payload

실제 Packed HND KV는 Token·Layer당 4,096 Bytes다.
유효 P KV Bytes는 `4,096 × P Rows`다.
Hidden과 Residual은 Prefix Token당 합계 16,384 Bytes다.
Token-only에는 이 H/R Payload가 없다.

| D 작업량 8L의 방법 | P 유효 Rows | P 유효 KV Bytes | 추가 H/R Bytes |
|---|---:|---:|---:|
| Layer8 | 28L | 114,688L | 16,384L |
| Token Suffix≈2L/9 | 28L | 114,688L | 0 |
| Hybrid4/Suffix≈L/8 | 28L | 114,688L | 14,336L |

위 표는 Alignment 이전의 정확한 산술식이다.
실제 Token Alignment 및 Block Padding은 별도로 기록한다.
Descriptor, Metadata와 Protocol Traffic도 이 표에 포함하지 않는다.
같은 Rows가 같은 Transfer Bytes라는 뜻은 아니다.
Payload Byte 비율을 Transfer Time 비율로 바꾸지 않는다.

## 현재 Source KV Allocator의 범위

현재 Source Worker는 36 Layers의 KV Cache를 예약한다.
실제 Worker Receipt에는 2,389 Blocks와 45,090,865,152 KV Bytes가 기록됐다.
Block128의 전체 Layer Page는 `128×36×4096=18,874,368 Bytes`다.
두 실측 숫자의 비율도 같은 Page 크기다.

Layer Source에서 계산하지 않는 Upper KV Plane도 Cache Group에 포함된다.
Prompt Page를 예약할 때 그 Plane의 물리적 공간도 함께 사용한다.
이를 유효한 계산 결과나 전송한 KV로 계산하지 않는다.

Alignment 이전의 Prompt-only Page 요구량은 다음과 같다.

| D 작업량 8L의 방법 | P의 Full36 Page 상당 Rows | P 유효 Rows | 미사용 Upper-plane 상당 Rows |
|---|---:|---:|---:|
| Layer8 | 36L | 28L | 8L |
| Token Suffix≈2L/9 | 28L | 28L | 0 |
| Hybrid4/Suffix≈L/8 | 31.5L | 28L | 3.5L |

실제 Page 수에는 `ceil(prefix/128)`이 적용된다.
Lookahead, 생성 Token 및 Lifetime 때문에 Peak Reservation은 더 커질 수 있다.
위 표를 Request별 Peak Memory 실측으로 표현하지 않는다.

모든 Worker의 물리 KV Budget은 계속 42GiB다.
유효 KV Rows가 같다고 실제로 수용하는 Request 수도 같지는 않다.
높은 Credit에서 Source KV 압박이 발생하는지 실제 Free Block Log로 검사한다.
기본 Credit4의 성능 차이를 이 산술만으로 KV 포화 때문이라고 확정하지 않는다.

## Credit와 Memory의 구분

Credit은 Coordinator의 Source Outstanding Window다.
Hidden Slot은 별도의 Buffer 자원이다.
KV Page도 별도의 자원이다.
Actual GPU Batch는 다시 별도의 실행 결과다.

Credit1/4/8/16의 추가 검사는 같은 Pool16 안에서 비교한다.
Native와 Token은 사용하지 않는 H/R Pool을 실제로 할당하지 않는다.
이 차이를 동일한 Hidden Memory라고 숨기지 않는다.
Pool32의 실제 Memory Admission 실패는 유효 성능점이 아니다.

현재 구현은 Partial-layer 전용 Source KV Allocator를 구현하지 않았다.
따라서 이를 Layer 축의 이론적 최적 Runtime이라고 주장하지 않는다.
그 Allocator를 바꾸면 NIXL Layout, Native Fallback 및 Lifetime을 다시 검증해야 한다.

## 근거

- Code: `geometry.py`, `adapter.py`, `hidden_transport.py`.
- 실제 Layout: Raw의 `KV_cache_layout_{prefill,decode}.json`.
- 실제 Budget: Raw의 `actual_runtime_{prefill,decode}.json`.
- 실제 제약: `scheduler_queues_*`, `KV_allocation_refusals_*`, `KV_lifetime_*`.
- Iso-work: `analysis/iso8_hybrid_heldout_confirmation_v1.json`.

Raw와 Analysis 경로는 `dense_ar_longbench400_2d_20261121` Namespace 기준이다.

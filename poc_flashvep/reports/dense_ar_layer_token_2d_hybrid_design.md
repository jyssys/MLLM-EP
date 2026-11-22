# Dense AR Layer×Token 2D Hybrid 실행 계약

캠페인: `dense_ar_longbench400_2d_20261121`.
이 문서는 설계와 정합성 검사를 구분한다.
Online 성능 판정은 별도 보고서에 기록한다.

## 계산 소유권

입력 길이는 `L`이다. 전체 Layer 수는 36이다.
Upper Layer 수는 `m`이다. Suffix 길이는 `s`이다.
Prefix 길이는 `T=L−s`이다. Lower Layer 수는 `k=36−m`이다.

| 계산 영역 | 실행 GPU | 선행 상태 |
|---|---|---|
| Prefix `0:T`, Lower `0:k` | P | 원래 입력 Token과 Position |
| Prefix `0:T`, Upper `k:36` | D | P의 Raw Hidden, Residual, Lower KV |
| Suffix `T:L`, Lower `0:k` | D | Prefix Lower KV |
| Suffix `T:L`, Upper `k:36` | D | Prefix Upper KV와 Suffix Lower Hidden |
| 생성 Token, 모든 Layer `0:36` | D | 전체 Prompt KV |

Prefix Upper 계산과 Suffix Lower 계산은 서로 다른 선행 조건을 갖는다.
Suffix Upper 계산은 두 계산의 완료를 요구한다.
Attention은 원래 Causal Mask와 절대 Position을 사용한다.

## 두 실행 순서

`UNALIGNED`는 실제 Scheduler Query Chunk를 유지한다.
한 Chunk가 Prefix/Suffix 경계를 가로지를 수 있다.
이때 Lower Layer는 Suffix Query Tail만 실행한다.
Upper Layer는 Prefix와 Suffix Query를 원래 순서로 결합한다.
각 Stage는 원래 KV Block Table에서 정확한 Attention Metadata를 만든다.

`PARTITION_ALIGNED`는 D의 Query Chunk를 `T`에서 끝낸다.
이 경계는 KV 할당 전에 Scheduler Token 수를 제한한다.
첫 Wave는 Prefix Upper Layer를 완료한다.
다음 Wave는 Suffix의 Full36 Layer를 실행한다.
동일 경계를 사용하는 원래 Full36 Single-GPU 실행이 독립 Teacher다.

두 실행 순서는 BI0에서 수치 결과가 다를 수 있다.
따라서 서로 다른 Query Geometry의 Bitwise Equality를 가정하지 않는다.
동일 Geometry Teacher에서는 전체 유효 KV와 최종 Hidden을 직접 비교한다.

## 작업량 보존

다음 식은 `STRUCTURAL_ACCOUNTING`이다.
이 식은 GPU 실행시간이나 FLOPs 동등성을 뜻하지 않는다.

- P Token-Layer Rows: `(36−m)(L−s)`.
- D Token-Layer Rows: `m(L−s)+36s`.
- 전체 Rows: `36L`.
- P Causal Attention Pairs: `(36−m)(L−s)(L−s+1)/2`.
- 전체 Attention Pairs: `36L(L+1)/2`.
- D Attention Pairs: 전체 Pairs에서 P Pairs를 뺀 값.

Native와 같은 Prompt 계산을 두 GPU에 재배치한다.
필요한 Token-Layer 계산을 생략하지 않는다.
공식 Native PD Proxy의 첫 Logit용 마지막 Prompt Query 재계산은 별도로 기록한다.
이를 Hybrid의 보존 작업량에 숨기지 않는다.

유효 Prefix KV Payload는 `4096×(36−m)(L−s)` Bytes다.
이는 Qwen3-8B BF16 KV의 구조적 값이다.
Block Padding과 Descriptor, Metadata는 실제 전송량에 추가된다.
`m>0`이면 Raw Hidden과 Residual은 합계 `16384×(L−s)` Bytes다.
실제 DMA 시간은 이 Byte 수로 추정하지 않는다.

## 순수 축 경계

- `m=0,s=0`: Native 경계.
- `m>0,s=0`: Layer-only 경계.
- `m=0,s>0`: Token-only 경계.
- `m>0,s>0`: Genuine Interior Hybrid.

Iso-work 비교에서는 `mL+(36−m)s`를 동일하게 맞춘다.
`Lm=36s`는 순수 Layer/Token 비교에만 적용한다.
Interior Hybrid에 이 순수 축 식을 그대로 적용하지 않는다.

## State와 Lifetime

P는 실제 Lower KV만 전송한다.
D는 같은 Prefix Position의 KV Slot에 이를 Import한다.
Raw Hidden과 Residual은 Final Norm 이전 값이다.
P의 Hidden Slot은 Remote Read 완료 ACK 후 반환한다.
D의 Hidden Slot은 Prefix Upper 소비 CUDA Event 후 반환한다.
전체 Request 종료 후 KV Credit 복원을 검사한다.

Mixed Batch는 요청별 시작 Layer를 갖는다.
Decode Query는 항상 Full36 Layer를 실행한다.
Prefill의 부분 Layer 상태를 Decode Query에 적용하지 않는다.
부분 Prefill Query가 1개여도 Full36 Decode Graph를 재사용하지 않는다.
이 Graph 경계 조건은 별도 실패 재현과 GPU 회귀 검사 대상이다.

## 현재 증거 범위

N8 Serial 검사에서 Layer2/4/8/12와 대응 Token이 Teacher를 통과했다.
Interior `m=4,s≈L/8`의 Aligned Hybrid도 통과했다.
각 요청의 전체 36-Layer 유효 KV Hash가 일치했다.
최종 Hidden과 Greedy 16 Tokens도 일치했다.
두 Pure-axis 경계도 같은 검사를 통과했다.

증거: [Full-state 비교](dense_ar_longbench400_2d_20261121/analysis/full_state_q4_comparisons.json).
Gate: [C0 N8 Gate](dense_ar_longbench400_2d_20261121/state/C0_full_state_q4_gate.json).
이는 `CORRECTNESS_ONLY`다.
Natural400 품질, C3 Continuous 성능, Decode Safety를 입증하지 않는다.

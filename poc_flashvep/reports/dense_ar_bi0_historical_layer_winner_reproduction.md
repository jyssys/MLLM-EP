# Historical Layer Winner와 Current Runtime Bridge

상태: `CLOSED`. Historical 승리는 확인했다. Current BI0의 Layer 우위는 재현하지 못했다.
기존 Commit은 `4595080c`다. 현재 BI0 기준 Commit은 `12e0f370a26ec544af81a90577874527ce8d4c5b`다.

## Historical 결과의 재계산

증거는 `HISTORICAL_MEASURED_ONLINE_REUSED`와 `TRACE_ANALYSIS`다.
Input8000/Output16, Last8, QPS6, N72, 실제 Batch1/1/1을 사용했다.
독립 Arrival Seed는 2026111920/1921/1922다. 각 Axis에 Server Start3회가 있다.
모든 Worker의 Scheduler 상한은 B8192/S1024다.

Raw Source ZIP, Config, Dataset SHA256과 정상 Worker 종료를 다시 검사했다.
다섯 핵심 실행 파일은 `4595080c`의 파일과 일치했다.
이 검사는 새 GPU 실행이 아니다.

| Mean Host Stage | Layer−Token |
|---|---:|
| Arrival→P Submit | +335.733ms |
| P Submit→Ready | +8.068ms |
| Ready→D Submit | −824.402ms |
| D Submit→First Worker | −14.379ms |
| Worker→Client | +0.011ms |
| Mean TTFT | −494.968ms |

다섯 Stage의 합은 Mean TTFT 차이와 일치한다. p95의 분해는 아니다.
TTFT p95의 Layer/Token 비율은 0.955705다.
Paired Start Bootstrap95% CI는 [0.935056, 0.971192]다. 독립 Start는 각3회다.
이 결과만으로 강한 Serving 성공 기준을 모두 통과했다고 판정하지 않는다.

## 같은 Historical Native와의 비교

같은3개 Seed의 Native Start도 다시 확인했다.
Layer/Native TTFT p95 비율은1.331144다.95% CI는 [1.315488, 1.346452]다.
Token/Native 비율은1.392840이다. CI는 [1.356337, 1.439279]다.
따라서 당시 두 Offloading 방식 모두 Native보다 TTFT가 나빴다.
Layer/Native ITL p99 CI 상한은1.055447이다. 현재 요구한1.05 Gate를 넘는다.
TPOT p95 CI 상한은1.043159다. 이 Gate는 통과한다.
Absolute34ms Decode SLO와 Native-relative5% Gate는 서로 다른 판정이다.

| Historical Host Demand | Native | Layer8 | Token8 |
|---|---:|---:|---:|
| P Submit→Ready | 291.846ms | 235.434ms | 227.366ms |
| D Submit→최종 Client Completion | 363.373ms | 435.267ms | 448.374ms |

이 Host 계약에서는 Native도 D 구간이 P보다 길었다.
P를 더 줄여도 D 구간을 늘리면 전체 Queue가 나빠질 수 있다.
이 결과를 Decode-safe Headroom이 확인된 Native 개선 사례로 사용하지 않는다.
Host Demand를 Exclusive GPU 사용률로 해석하지 않는다.

## 실제 Historical 계약

Historical은 BI1과 기존 Phase-stable Norm Correction을 사용했다.
실제 P Worker는 Compile0, Graph NONE과 Async False였다.
실제 D Worker는 Compile0, FULL_DECODE_ONLY Graph와 Async True였다.
Config의 요청값만으로 실제 Async 상태를 판정하지 않았다.
KV Tensor의 Block은16 Tokens다. KV Budget은 GPU당42GiB다.
Source Outstanding Window는32다. P Prefill은 D Reservation ACK 후에 제출된다.
D는 이전 요청의 마지막 Client Completion까지 다음 Suffix를 제출하지 않는다.
이 Guard는 Dense AR의 필수 제약이 아니다.

| Runtime 항목 | Historical | Current Bridge Primary |
|---|---|---|
| P Async Scheduling | False | True |
| P Graph 설정 | NONE | FULL_DECODE_ONLY. 실제 Prefill Graph 사용을 뜻하지 않음 |
| D Async/Graph | True / FULL_DECODE_ONLY | True / FULL_DECODE_ONLY |
| KV Block | 16 | 128 |
| KV Byte Coalescing | True | False. 별도 One-factor 진단에서만 True |
| P 앞 D Reservation ACK | 있음 | 없음 |
| Source Window/Credit32 | Pending·Active D를 기준으로 제한 | P Dispatch에서 획득하고 첫 D Client Token에서 반환 |

따라서 같은 숫자32를 같은 Lifetime Window라고 표시하지 않는다.
추가 감사에서 여섯 Historical Layer/Token Start의 Norm Correction Counter는 모두0이었다.
해당 Pair에서 Correction 연산이 활성화돼 이겼다는 설명은 기록과 맞지 않는다.
실제 Graph Replay 기록은1-Token Full Decode Shape였다.
Prefill Suffix Graph 가속을 과거 우위의 원인으로 가정하지 않는다.
[Runtime·Transport 감사](dense_ar_bi0_axis_rootcause_20261120/analysis/historical_current_runtime_transport_audit.json)에 실제 Config와 Counter를 보존한다.

## Bridge의 범위

A는 검증한 Historical3개 Paired Start를 재사용한다.
B는 Current Runtime의 BI1 진단이다. C는 같은 Current Runtime의 BI0다.
B/C는 같은 Historical Prompt IDs, Arrival, Cut과 Output16 계약을 사용한다.
Current BI0/BI1은 먼저 새로운 N8 State·Greedy Gate를 검사한다.
C의 목표는 각 Axis와 Native에 독립 Start3회다.
B의 초기 목표는 Axis별1회 진단이다. BI1 결과를 BI0 Main Table에 합치지 않는다.

Historical Window32와 Current Credit32는 반환 시점이 다르다.
Historical은 혼합 Token Suffix로 Aggregate Rows를 정확하게 맞췄다.
Current는 기존 nearest16 Prefix 규칙을 유지한다. 따라서 작은 Rows 오차를 별도로 기록한다.
A/B는 여러 Runtime 항목이 다르다. 이를 단일 변경의 인과 효과로 표현하지 않는다.

[Source·Guard 감사 JSON](dense_ar_bi0_axis_rootcause_20261120/analysis/historical_winner_source_guard_audit_v1.json),
[사전 Bridge 계약](dense_ar_bi0_axis_rootcause_20261120/configs/frozen_historical_bridge_v1.json)을 보존한다.
새 Bridge 결과와 미실행 조건을 별도로 보존했다.

## Current BI0·KV128의 독립3회 결과

같은 Historical72 입력, Output16, Last8과 QPS6을 사용했다.
Actual Batch는1/1/1이다. Source Credit32와 Hidden Slots32를 사용했다.
Seed20/21/22에서 Layer, Token과 Native를 각각 새로 실행했다.
현재 KV Block은128이다. Historical Block16과 혼동하지 않는다.

| Current BI0 비교 | 비율 | Paired95% CI |
|---|---:|---:|
| Layer/Token TTFT p95 | 1.160249 | [1.088786, 1.249619] |
| Layer/Token SLO-goodput | 0.962092 | [0.952115, 0.975201] |
| Layer/Native TTFT p95 | 0.537230 | [0.492386, 0.575226] |
| Token/Native TTFT p95 | 0.463030 | [0.394060, 0.528562] |

Current BI0에서는 Token이 Layer보다 빨랐다.
두 Axis 모두 같은 Current Native보다 TTFT가 낮았다.
두 Axis 모두 Native-relative ITL p99와 TPOT p95의5% Gate를 통과했다.
이는 Fixed-output C0 결과다. Continuous Serving이나 Sustainable Capacity 결과가 아니다.
Token의 Aggregate D Rows는 Layer보다0.1% 작다. Exact Matched Work라고 표시하지 않는다.

Current BI0의 Mean P Host Service는 Layer185.859ms, Token174.783ms였다.
D 준비 Host Stage도 Layer79.998ms, Token73.493ms였다.
Historical에서 관측한 D 준비 이점이 Current BI0에서 그대로 남아 있지는 않았다.
이 비교의 변화에는 여러 Runtime 요인이 함께 들어간다.
BI Flag 하나만의 효과나 Exclusive GEMM 원인으로 확정하지 않는다.

[Bridge 독립 Start 통계](dense_ar_bi0_axis_rootcause_20261120/analysis/historical_bridge_independent_start_statistics.json)를 참조한다.

## Current Runtime의 Numerical Mode 진단

Seed20의 같은72개 Request에서 BI0/BI1을 비교했다.
각 Axis·Mode는 독립 Start1회다. 성능 CI는 없다.
Actual Engine Option, P/D Query Geometry, GPU UUID와 핵심 Source Hash는 일치했다.
Config JSON에는 BI Flag 외에도 Dispatcher 적격성·보고용 Flag 차이가 있다.
이 차이도 숨기지 않고 감사 JSON에 보존했다.

| 같은 Current Runtime | P Submit→Ready | D Submit→First Worker | 후속15 Token Worker Residency | Mean Worker Interval |
|---|---:|---:|---:|---:|
| Layer BI0 | 188.597ms | 80.822ms | 108.046ms | 7.203ms |
| Layer BI1 | 238.994ms | 97.408ms | 345.076ms | 23.005ms |
| Token BI0 | 174.385ms | 77.819ms | 108.187ms | 7.212ms |
| Token BI1 | 225.472ms | 92.928ms | 347.829ms | 23.189ms |

BI0에서는 후속 Decode Residency가 크게 짧아졌다.
따라서 BI1의 D 병목과 같은 Stage Balance를 BI0에 가정할 수 없다.
Worker Emission Interval은 GPU-exclusive Kernel Time이 아니다.
BI0/BI1 Greedy Output Equality도 주장하지 않는다.
이 진단은 Historical→Current의 ACK·Block·Allocator 차이를 함께 식별한 실험이 아니다.

[Numerical Bridge 계약 감사](dense_ar_bi0_axis_rootcause_20261120/analysis/current_runtime_numerical_bridge_contract.json)에 모든 Config 차이를 보존한다.

## KV16의 유리했던 첫 Start를 별도로 반증

사전 One-factor KV16 진단의 Seed20에서는 Layer가 빨랐다.
TTFT p95는 Layer5093.638ms, Token5527.215ms였다.
이 유리한 Start를 Held-out Confirmation에 합치지 않았다.
Seed21/22/23에서 Native와 두 Axis를 각각 새로 실행했다.
Block16, Credit32, Hidden Slots32와 실제1/1/1을 유지했다.

Held-out3회 TTFT p95의 Layer/Token 비율은1.025844다.
95% CI는 [0.962237, 1.071844]다.
SLO-goodput 비율은0.960418이다. CI는 [0.904757, 1.007581]다.
두 Axis 모두 같은 KV16 Native 대비 Decode5% Gate를 통과했다.
그러나 첫 Start의 Layer TTFT 우위는 독립3회에서 확정되지 않았다.
이 결과를 유리한 Seed20만 남긴 Layer Winner로 표시하지 않는다.
또한 느린 Token/KV16만을 Token의 Best-tuned 기준선으로 사용하지 않는다.

## BI1 Transport Intervention의 반복 결과

동일 Current BI1 Runtime에서 Exact-byte Coalescing만 변경했다.
No-coalescing의 Layer/Token TTFT p95 비율은0.982512다.
95% CI는 [0.967079, 1.000334]다. 확정된 Axis Winner는 아니다.
Coalescing 적용 시 비율은0.953549다. CI는 [0.937964, 0.966709]다.
같은3개 Seed의 비율 변화는0.970521이다. CI는 [0.938857, 0.996444]다.
각 조건의3회에는 최초 Discovery1회와 추가 Seed2회가 포함된다.

BI Flag와 Model Math는 바꾸지 않았다.
그러나 실제 Descriptor는 두 설정 모두 입력1764개, 제출1764개였다.
의도한 물리적 병합은 실현하지 못했다. 해당 기전의 인과 검증은 실패했다.
따라서 이 비율 차이를 Historical Coalescing 이득의 재현으로 사용하지 않는다.
Coalescing은 Token의 Mean TTFT를303.965ms 늘렸다. Layer의 절대 개선은 불확실했다.
따라서 상대 비율의 개선을 두 방식 모두의 속도 향상으로 표시하지 않는다.
더 빠른 No-coalescing Token을 배제하지 않는다.
Current BI1 Native-relative Decode 보호는 측정하지 않았다.
따라서 `DIAGNOSTIC`으로 남긴다. BI0 Layer-specific Serving PASS로 판정하지 않는다.
Historical의 Canonical KV Allocation과 ACK 계약까지 재현한 실험도 아니다.

[Ready→D 인과 분해](dense_ar_bi0_ready_to_d_queue_root_cause.md)에 Intervention과 한계를 기록했다.

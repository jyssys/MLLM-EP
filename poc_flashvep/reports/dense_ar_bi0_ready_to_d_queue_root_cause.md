# Ready→D 대기와 Guard의 원인 분리

상태: `CLOSED / PARTIAL_ROOT_CAUSE_IDENTIFICATION`. Host Boundary와 GPU Kernel Time을 구분한다.

## Historical Last8의 정확한 대기 분해

Input8000/Output16, QPS6, N72, 실제 Batch1/1/1, 독립 Start3회를 재분석했다.
다음 값은 Request당 Host 시간의 Start 평균이다.
증거는 `TRACE_ANALYSIS`다.

| Ready→D 구간과 겹친 이전 요청의 상태 | Layer | Token |
|---|---:|---:|
| D 준비 Guard | 1184.114ms | 1531.730ms |
| Decode Residency Guard | 5284.635ms | 5760.954ms |
| 최종 Delivery Guard | 4.524ms | 4.735ms |
| 나머지 Host 대기 | 11.130ms | 11.386ms |
| 전체 Ready→D Submit | 6484.403ms | 7308.805ms |

차이는 −824.402ms다. 대부분의 대기는 이전 요청의 Guard와 겹친다.
겹친 Decode Residency 시간이 실제 Decode GPU 비용 차이와 같지는 않다.
앞선 D 준비 비용과 요청 도착 순서도 이 겹침을 바꿀 수 있다.
Guard를 제거한 실행의 이득으로 해석하지 않는다.

## 대기 구간의 감소와 실제 제출 시각의 감소

같은 Arrival 기준으로 Layer의 P Ready는343.801ms 늦었다.
P Ready가 늦어지면 Ready→D 대기가 짧아질 수 있다. 이것만으로 TTFT가 개선되지는 않는다.
실제 D Submit은480.601ms 빨랐다.
따라서 −824.402ms는 다음 두 경계 차이의 정확한 합이다.

| Ready→D 차이의 항등식 | Layer−Token |
|---|---:|
| D Submit 시각의 차이 | −480.601ms |
| P Ready가 늦어진 만큼의 대기 감소 | −343.801ms |
| Ready→D 대기 차이 | −824.402ms |

두 항을 독립적인 Intervention 효과라고 해석하지 않는다.

같은 FIFO D 실행 체인도 정확하게 풀어 계산했다.
각 이전 요청의 Service와 다음 Host Gap이 이후 요청들에 반복해서 영향을 준다.
Request 순서에 따른 실제 가중치를 사용했다. M/M/1 근사식은 사용하지 않았다.

| Mean D Submit 차이의 정확한 체인 분해 | Layer−Token |
|---|---:|
| 최초 D Submit Offset | +9.772ms |
| 이전 D 준비 구간의 누적 | −537.264ms |
| 이전 Decode Residency의 누적 | +43.140ms |
| 이전 최종 Delivery의 누적 | +0.122ms |
| P Ready 공급 부족으로 인한 누적 | 0.000ms |
| 실행 가능한 D의 Host Gap 누적 | +3.629ms |
| 합계 | −480.601ms |

Decode Residency의 실제 누적 차이는 Layer에 불리했다.
앞의 Guard 겹침 표에서 Decode 구간과 겹친 대기가 작았다는 사실과 다르다.
대기 구간의 겹침을 Decode Service의 인과 효과로 읽으면 잘못된 설명이 된다.
D 준비 구간의 누적 차이가 실제 제출 시각 차이의 주된 항이다.
이 준비 구간에는 Import, Transport, Suffix Forward와 Host Scheduling이 포함된다.
따라서 Exclusive D GEMM 비용만의 효과로 확정하지 않는다.

Historical Layer/Token의 모든71개 D 전환에서 다음 P Ready가 이미 존재했다.
반면 별도의 Current LongBench Last4 Natural Start에서는 공급이 늦은 전환이 약51.76%/49.25%였다.
이 두 Family의 차이는 P/D Critical Path가 달라질 수 있다는 근거다.
서로 다른 Cut과 Dataset의 차이를 단일 변수의 인과 효과로 사용하지 않는다.
여기서 공급 대기는 Host 경계다. Physical GPU Idle이나 Decode-safe Window가 아니다.

Historical D Submit→First Worker 평균은 Layer81.555ms, Token95.934ms다.
이후 Worker Decode Residency는 Layer353.424ms, Token352.161ms다.
따라서 Token의 후속 Decode 자체가 크게 느렸다고 설명할 근거는 없다.
D 준비 비용과 전체 Guard Cycle의 작은 차이가 Queue에서 누적될 수 있다.
하지만 이 Boundary는 GPU Compute, Import, Transport와 Host 비용을 포함한다.
Exclusive GEMM이 결정적 원인인지는 이 기록만으로 판정할 수 없다.

## Reservation ACK와 Current Runtime의 차이

Historical은 P 제출 전에 D Reservation ACK를 기다린다.
D Worker는 Engine Step 경계에서 Control Message를 처리한다.
Credit Reject0은 ACK 지연0을 뜻하지 않는다.
Current C0은 P 제출 전에 이 Reservation ACK를 기다리지 않는다.
Current C0도 D의 전체 요청 완료와 Device-idle Certificate를 기다린다.
두 Runtime 모두 대기를 만들지만, 경계와 Credit 반환 정책은 다르다.

Current Official Pull은 D Submit 후 KV 수신을 시작한다.
따라서 Current Ready→D만으로 전체 Transfer 비용을 측정할 수 없다.
KV Read API, Submit API와 Scheduler Receive Notice를 별도로 분석한다.
API Enclosure는 순수 DMA 시간도, CPU-exclusive 시간도 아니다.

## 같은 Historical Workload의 Current BI0 결과

Input8000/Output16·Last8·QPS6을 동일하게 유지한 Current BI0 독립3회를 비교했다.
KV128, Credit32와 실제 Batch1/1/1을 사용했다.
다음 표는 각 Start의 Request 평균 차이를 다시 평균한 값이다.

| Current BI0 Host Stage | Layer−Token |
|---|---:|
| Arrival→P Submit | +396.755ms |
| P Submit→Parent Ready | +11.076ms |
| Parent Ready→D Submit | −137.433ms |
| D Submit→Worker Accept | +1.327ms |
| Worker Accept→첫 D Prepare | +3.515ms |
| 첫 D Prepare→First Worker | +1.663ms |
| First Worker→Client | +0.064ms |
| Mean TTFT | +276.966ms |

Ready→D 이득은 현재도 있었지만 P 앞의 추가 대기를 만회하지 못했다.
Current BI0의 D Submit→First Worker 평균은 Layer79.998ms, Token73.493ms였다.
즉, Historical의 D 준비14.379ms Layer 이점도 이 Current 계약에서는 유지되지 않았다.
이 값에는 Import와 Host 단계가 포함된다. D GPU Forward만의 우열은 아니다.
Current Ready→D는 KV Transfer 전체 구간이 아니다. Official Pull은 D 제출 후 시작한다.
Historical과 경계가 다르므로 −824ms에서 −137ms로 바뀐 것을 단일 Runtime 변경의 효과라고 표현하지 않는다.
같은 Request, Cut과 Output을 맞춰도 Numerical Mode, KV Block과 Credit Lifetime은 다르다.

## Exact-byte Coalescing의 별도 반증 실험

Current BI1의 동일 Input8000/Output16·Last8에서 Byte Coalescing만 변경했다.
각 Factor와 Axis를 같은3개 Arrival Seed에서 실행했다.
최초 Discovery1회와 추가 Seed2회를 포함한다. 완전히 독립된 Held-out3회는 아니다.
Model Math, BI Flag, Cut, Graph와 KV128은 유지했다.

| BI1 Transport 조건 | Layer/Token TTFT p95 | Paired95% CI |
|---|---:|---:|
| Coalescing Flag OFF | 0.982512 | [0.967079, 1.000334] |
| Coalescing Flag ON | 0.953549 | [0.937964, 0.966709] |

두 Axis 비율의 Coalescing/No-coalescing 비율은0.970521이다.
Paired95% CI는 [0.938857, 0.996444]다.
그러나 후속 Descriptor 감사에서 의도한 병합은 확인되지 않았다.
모든 측정 Request의 Descriptor는 입력1764개, 제출1764개였다.
Coalescing Candidate의 Byte Count도 기록되지 않았다.
따라서 변경한 것은 Host Flag다. 실제 물리적 Descriptor 병합의 효과는 검증하지 못했다.
이 비율 차이를 DMA 병합의 인과 효과나 Historical 원인의 식별로 사용하지 않는다.
Model Math를 바꾸지 않았다는 사실도 이 Intervention의 실현 실패를 대신하지 못한다.
단, 이것은 양쪽을 모두 빠르게 만든 최적화가 아니다.
Layer의 Mean TTFT 변화는 −9.489ms다.95% CI는 [−360.607, 336.355]ms다.
Token의 Mean TTFT는303.965ms 늘었다. CI는 [125.352, 460.831]ms다.
Token의 Ready→D 대기 증가270.571ms와 D 제출 후 세 Host Stage 증가도 기록됐다.
이 실행에서 실제 Descriptor 병합이 빨랐거나 느렸다고 판정할 근거는 없다.
빠른 No-coalescing Token을 제외하고 느린 Token 설정만 비교 기준으로 사용하지 않는다.
다만 실행 순서와3개 Start만으로 모든 Runtime 변화의 원인을 식별하지는 못한다.
Native-relative BI1 Decode 보호는 `NOT_MEASURED`다. 증거 수준은 `DIAGNOSTIC`이다.
Intervention 실현 여부는 `NOT_REALIZED`다. 통계량과 실패한 기전 검증을 모두 보존한다.
이 결과를 BI0 Main Serving 성공이나 Decode-safe Window 검증으로 사용하지 않는다.

## 최종 해석의 한계

과거 Last8의 −824ms와 현재 Last4의 −45ms를 직접 빼지 않는다.
같은 Input8000/Output16·Last8의 Current Bridge로 비교한다.
Historical Guard의 겹침은 확인했다. 개별 Guard·ACK 변경의 인과 효과는 아직 미확정이다.
안전 계약을 제거하는 실험은 실행하지 않는다.

[Per-request Guard 감사](dense_ar_bi0_axis_rootcause_20261120/analysis/historical_winner_source_guard_audit_v1.json)를 참조한다.
[정확한 FIFO D 체인 분해](dense_ar_bi0_axis_rootcause_20261120/analysis/exact_destination_chain_decomposition_v1.json)도 보존한다.
[Coalescing Host Stage Intervention](dense_ar_bi0_axis_rootcause_20261120/analysis/coalescing_host_stage_intervention.json)에 절대 성능의 악화도 보존한다.

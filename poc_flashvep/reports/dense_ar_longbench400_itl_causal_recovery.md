# LongBench400 Decode Tail: Causal Recovery Checkpoint

작성일: 2026-11-21 KST. 이 문서는 진행 중인 검증의 체크포인트다.
최종 Serving 성공 판정이 아니다.

## 판정 계약

Native PD, Layer, Token, Hybrid의 Decode 보호 기준은 같다.
Native 대비 ITL p99와 TPOT p95 비율의 Paired 95% CI 상한은 1.05 이하여야 한다.
독립 Server Start가 CI의 표본 단위다.
한 Start의 Request 또는 Token을 독립 Start로 취급하지 않는다.

Absolute SLO도 유지한다.
TTFT는 5,000ms 이하여야 한다.
각 Request의 모든 Client-visible ITL은 34ms 이하여야 한다.
Request TPOT는 33ms/token 이하여야 한다.
세 조건을 모두 만족한 Request의 Output Tokens만 SLO-goodput에 포함한다.

측정 시각을 수정하지 않는다.
늦은 Token을 버퍼링해서 ITL을 평탄화하지 않는다.
Co-delivered Token의 실제 0ms 간격도 보존한다.

## 경쟁 가설과 개입

| 가설 | 반증 가능한 예측 | 현재 검사 |
|---|---|---|
| 큰 D Prefill이 Decode와 함께 실행된다 | Prefill 계산만 미루면 ITL이 줄어든다 | 실제 Scheduler의 pre-allocation throttle 경로 사용 |
| Compile 차이가 주원인이다 | 원래 compiled model의 Token도 Decode를 보호한다 | Stock Token 대조군 실행 |
| Host KV 준비가 남은 Decode 지연에 기여한다 | 같은 계산 정책에서 READ 준비 시점만 바꾸면 잔여 지연이 줄어든다 | 별도 정합성 Gate 진행 |
| 전체 Request 직렬화만 보호할 수 있다 | 연속 Decode를 유지한 compute-only 제어는 보호하지 못한다 | Whole-request admission과 compute-only 제어를 구분 |

Stock Token의 기본 Compile 경로만으로는 보호하지 못했다.
Discovery200/QPS1.8의 Cut8 ITL p99는 46.631ms였다.
QPS2.8에서는 61.975ms였다.
두 값은 각각 독립 Start 1회다.
이 결과는 Compile 차이만으로 원인을 설명하는 가설을 반박한다.

## 실제 개입 결과

아래 값은 `MEASURED_ONLINE` Discovery다.
모든 값은 독립 Start 1회다.
Dataset은 동결한 Discovery200이다.
Arrival Seed는 2026112291이다.
동일 QPS의 각 A/B는 동일 GPU Pair와 Source Capsule을 사용한다.
모든 Worker는 BI0/BF16/B8192/S1024/KV42GiB/Block128을 사용한다.

| QPS | Arm | TTFT p95 (ms) | ITL p99 (ms) | TPOT p95 (ms/token) | SLO-goodput (Tokens/s) |
|---:|---|---:|---:|---:|---:|
| 1.8 | Native strong | 3388.371 | 17.889 | 8.460 | 54.17 |
| 1.8 | Layer4 original | 2774.010 | 43.617 | 11.485 | 26.31 |
| 1.8 | Layer4 compute-only | 2914.253 | 10.728 | 8.451 | 55.801 |
| 1.8 | Stock Token4 compute-only | 2578.140 | 10.920 | 8.321 | 54.997 |
| 1.8 | Hybrid4/f0.125 compute-only | 2505.385 | 12.138 | 8.902 | 53.884 |
| 2.8 | Native strong | 10304.221 | 20.696 | 8.617 | 41.50 |
| 2.8 | Layer4 original | 7622.149 | 43.383 | 11.930 | 21.38 |
| 2.8 | Layer4 compute-only | 10161.226 | 9.900 | 8.566 | 43.747 |
| 2.8 | Stock Token4 compute-only | 6572.802 | 12.542 | 8.964 | 52.013 |
| 2.8 | Hybrid4/f0.125 compute-only | 9069.583 | 11.858 | 9.407 | 44.126 |

`Native strong`는 원래 Full-model Compile 경로와 제한 없는 Source admission을 유지한다.
Stock Token도 이 경로를 유지한다.
Layer와 Hybrid는 row-aware 부분 Layer 경로와 Source Credit4를 사용한다.
이 표는 동일 Actual Batch 또는 동일 GEMM Rows의 비교가 아니다.
Actual Geometry는 각 Raw Log와 Metrics에 별도로 저장한다.
Hybrid4/f0.125의 D 작업량은 Layer4 또는 Token4와 다르다.

Compute-only 제어는 큰 Prefill 계산만 미룬다.
Remote KV 처리와 연속 Decode Batching은 계속 허용한다.
Native의 마지막 Prompt Query 재실행도 유지한다.
이미 할당한 Token 수를 사후 수정하지 않는다.
원래 Scheduler의 Allocation 검사와 Lifetime 검사를 유지한다.

Whole-request D-quiescent 제어는 별도 대조군이다.
QPS2.8에서 Layer4의 ITL p99는 8.681ms였다.
그러나 TTFT p95는 10,969.776ms였다.
Compute-only 제어의 TTFT p95는 10,161.226ms였다.
전체 Request 직렬화를 일반 Continuous Serving으로 표시하지 않는다.

## 잔여 지연

Layer4/QPS2.8/compute-only의 Native-matched Point 한계 초과 간격은 26개였다.
그중 25개가 NIXL Host 구간과 겹쳤다.
큰 Prefill과 Decode가 같은 Host Step에서 실행된 간격은 없었다.
이 겹침은 `TRACE_ANALYSIS`다.
겹침만으로 NIXL의 인과적 기여율을 확정하지 않는다.
Host 구간을 GPU-exclusive 시간이나 순수 DMA 시간으로 표현하지 않는다.

Hybrid4/QPS2.8의 TPOT p95 비율은 Native strong 대비 약 1.092였다.
Point 기준으로도 5% 보호 기준을 넘는다.
따라서 Hybrid의 Decode 보호를 아직 PASS로 표시하지 않는다.
ITL p99가 줄었다는 사실만으로 TPOT Gate를 대체하지 않는다.

추가 개입은 D의 Remote-KV READ 준비를 Active Decode 이후로 미룬다.
계산 정책, Model, Output 및 Client Timestamp는 바꾸지 않는다.
ACK, Heartbeat 및 KV Retirement는 원래 경로를 유지한다.
Native, Layer, Stock Token, Hybrid에 같은 정책을 적용한다.
N1/N8의 동일 Geometry 검사와 Natural32/400 검사를 먼저 수행한다.
검사 전에는 이 정책의 성능을 적격 결과로 사용하지 않는다.

## 정합성 및 제한

기존 compute-only 제어는 네 Numerical Family에서 Natural400 Gate를 통과했다.
각 Family는 동일 Geometry의 N1/N8 Teacher 근거도 보존한다.
Import KV, Hidden, Query/Layer Coverage와 Lifetime을 별도로 검사한다.
BI0의 다른 Batch에서 발생한 Token 차이는 Task-quality Gate로 별도 평가한다.
모든 Batch에서 Greedy Token이 일치한다고 주장하지 않는다.

현재 수치는 독립 반복에 의한 Safety CI가 아니다.
Final Static 후보와 Runtime 정책은 Held-out 전에 동결한다.
이후 Native strong과 공통 정책 Native를 모두 대조군으로 사용한다.
Finite-cohort Goodput을 Sustainable Capacity로 표현하지 않는다.

## 재현 근거

- Raw namespace: `dense_ar_longbench400_2d_20261121/raw/ThrottleAB_Discovery200_*`.
- Source capsule: `compute_throttle_AB_discovery_v1_20261121.zip`.
- Config: `compute_throttle_AB_lane0_v1.json`, `compute_throttle_AB_lane1_v1.json`.
- Red-capable diagnostic: `python -m poc_flashvep.dense_ar_longbench400_pd.decode_tail_audit --red-fixture`.
- 잔여 Tail 분석: 각 Arm의 `*_decode_tail_audit.json`.
- READ 준비 정책: `remote_setup_deferral.py`와 `remote_setup_qualification.py`.
- Raw Log, 실패 실행 및 이전 Source Capsule은 삭제하지 않았다.

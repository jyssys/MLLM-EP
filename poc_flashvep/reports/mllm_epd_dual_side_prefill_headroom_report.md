# E/D 추가 Prefill Headroom: 실측 종합

판정은 `HOLD`다. 물리적 Idle과 메모리 적재는 관측했다. 안전한 추가 Prefill Quantum은 확정하지 못했다.
이 문서는 완료된 C25·C40·C41의 범위로 동결한다. 전체 캠페인의 종료 보고서는 아니다.

## 결과 요약

| 질문 | 실측 또는 미측정 결과 |
|---|---|
| E와 D의 Idle | 적격 QPS0.5 C25에서 물리적 Idle과 동시 Idle 관측. Prefill-heavy 조건은 아님 |
| E의 추가 Weight 적재 | Original Full Model·중복 Vision·KV8GiB를 E와 같은 GPU4에 실제 적재 |
| 추가 Prefill 실행 | E/D 128·512-token Prefix Probe를 독립 N3로 실행. 모든 조건의 전체 보호 Gate 미통과 |
| Imported-KV Suffix | Input3종×Quantum5종의 15개 Strict 정합성 통과. 실제 Query 작업량 보존 |
| Suffix 비용 | Clean/Profile/Clean Gate 미통과. 비용은 DIAGNOSTIC |
| E가 더 큰 Safe Quantum을 처리하는가 | NOT_MEASURED. E/D의 같은 Imported-KV Suffix 비교도 없음 |
| 실제 요청 완료 기여·Serving Speedup | Probe는 기여하지 않음. 이 표로 Speedup을 주장하지 않음 |
| Encode-safe·Decode-safe Budget | NOT_CONFIRMED. Encoder-exclusive Service Gate는 NOT_MEASURED |

## 물리적 Idle의 적용 범위

C25 중앙 Arrival 구간은 157.172988초다. 독립 초기화는 1회다.
E/D 동시 Kernel Idle은 그 구간의 74.237585%다.
P Busy와 동시 Idle의 교집합은 전체 구간의 9.994642%다.
같은 교집합은 P Busy 시간의 71.043991%다.
두 분모를 혼합하지 않는다. P Queue가 지속되는 구간의 비율로 대체하지 않는다.
이 QPS0.5 관측을 QPS2 Probe의 실제 미래 Window라고 가정하지 않았다.
고부하의 E 비병목·P 포화·Service Proximity·안전 여유의 동시 조건은 미확인이다.

## 실제 추가 작업과 보호 Gate

실제 Probe는 613회다. Prefix Token 합계는 195,200개다.
추가 Token-layer Rows는 5,465,600개다.
각 Probe의 1-token Sample은 폐기했다. Client Throughput·Goodput 분자에 더하지 않았다.
각 조건은 같은 Main100의 Natural EOS와 Equal-output Replay를 따로 실행했다.

| Helper | Output | Actual Calls | Prefix Tokens | Absolute Decode≥95% Every Start | Full HTTP Protection |
|---|---|---:|---:|---|---|
| E128 | natural | 61 | 7808 | False | False |
| E128 | replay | 61 | 7808 | True | False |
| E512 | natural | 59 | 30208 | True | False |
| E512 | replay | 58 | 29696 | True | False |
| D128 | natural | 92 | 11776 | False | False |
| D128 | replay | 95 | 12160 | False | False |
| D512 | natural | 95 | 48640 | True | False |
| D512 | replay | 92 | 47104 | True | False |

NativeStrong·NativeCommon·동일 초기화의 No-probe 평균을 각각 대조군으로 사용했다.
E HTTP p95·ITL p99·TPOT p95·TTFT p50/p95의 Paired CI 상한은 각각 1.05 이하여야 한다.
Absolute Decode는 요청별 All-ITL과 TPOT의 AND다. 각 Start에서 95% 이상이어야 한다.
추가 Probe가 없는 Control24개 중 4개도 Absolute Gate를 실패했다.
따라서 모든 드문 Client Gap을 Probe의 GPU 경쟁으로 단정하지 않는다.
실패한 전체 Gate는 유지한다. 다른 일부 Metric의 개선을 이유로 Gate를 제거하지 않는다.
E HTTP는 Encode Queue·Host·실제 Encode의 Enclosure다. Encoder-exclusive GPU Service가 아니다.
고정한 Probe 순서, 다른 Batch와 Natural Output 변화의 한계도 유지한다.

## 실제 메모리 적재

E는 원래 Vision Encoder다. 추가 Helper는 LLM만 적재한 경량 Worker가 아니라 원래 Full Model이다.
Logical Vision Parameters는 1,353,100,288 bytes다. 추가 Full Model Parameters는 16,584,333,312 bytes다.
그중 LLM은 15,231,233,024 bytes다. 중복 Vision은 1,353,100,288 bytes다.
Helper Unique KV Storage는 8,587,837,440 bytes다. Native D는 KV42GiB 설정을 유지했다.
서로 같은 KV Capacity라고 표현하지 않는다.

| Independent Block | Final GPU4 Device Free bytes |
|---:|---:|
| 0 | 19053608960 |
| 1 | 21605842944 |
| 2 | 20857159680 |

이는 실제 Workload와 Probe를 마친 뒤의 Device 표본이다. Process Lifetime Peak나 최대 Activation 표본은 아니다.
E와 Helper가 보고한 Device Used를 더하지 않았다. 같은 GPU의 메모리를 중복 집계하지 않았다.
Allocator Reserved는 Weight 크기가 아니다. Process Lifetime Peak도 Cohort별 Peak로 대체하지 않았다.
최대 Encoder Batch Capacity와 OOM 경계는 측정하지 않았다. 메모리 적재만으로 안전한 Prefill을 주장하지 않는다.
이전 C16의 제한된 입력 Memory Fit과 이번 Main100 N3의 표본은 별도 근거다.

## Prefix와 Suffix 비용의 구분

C40은 Prefix128/512다. Imported Prefix KV가 없는 실제 추가 작업이다.
C26은 Prefix128/256/512/1024/2048의 별도 E/D 고립 비용이다. 두 Arm의 전체 Observer Gate는 실패했다.
C41은 실제 P Prefix KV를 NIXL로 가져온 D Suffix다. Input은 4831·2961·27840 Tokens다.
15개 조합의 Strict 검사 뒤 Cell·단계별 9개 비용 표본을 측정했다. 독립 초기화는 1회다.
Clean-before HTTP p95의 Cell 범위는 [63.221, 201.262]ms다.
Clean-after HTTP p95의 Cell 범위는 [74.811, 224.265]ms다.
Profile Forward Enclosure p95의 Cell 범위는 [28.934, 154.926]ms다.
모든 비용은 전체 5% Gate 미통과로 DIAGNOSTIC이다. GPU-exclusive Kernel 또는 DMA 비용이 아니다.
HTTP는 KV Import·Embedding 소비·Admission·Sampling·Delivery를 포함한다. 겹치는 Forward 비용을 다시 더하지 않는다.
E의 Prefix 비용을 D의 같은 Suffix 비용으로 취급하지 않았다.
C25의 미래 Physical Gap과 비용을 비교한 Fit은 OFFLINE_ORACLE이다. 실제 안전 Window나 Speedup이 아니다.

## 후속 연구 Gate와 재현 자료

현재 결과로 ENCODE_SAFE_HEADROOM_CONFIRMED 또는 DECODE_SAFE_HEADROOM_CONFIRMED를 부여하지 않는다.
Request-level Elastic Offloading·Pipeline·Wavefront·Context Parallelism의 실제 3GPU 이득도 미검증이다.
추가 Runtime 정책·Actual Batch 감사는 별도 실험이다. 이 동결된 Probe 결과를 사후 변경하지 않는다.
원래 Vision·Processor·BF16 Weight·Full28·MRoPE·Causal Attention·Sampling을 유지했다.
Model/Processor Revision은 cc594898137f460bfe9f0759e9844b3ce807cfb5다.
Main100 SHA256은 910506f24984e8f83e433c169b0a03d06c89396e1ef8a5f66b514d97406844d1이다.
GPU4/5/6, B8192/S1024, Native P/D KV42GiB, Block128, Prefix Cache Off다.
analysis/dual_side_headroom_*_v0의 JSON·CSV에 Source SHA와 실제 작업·메모리 표를 저장했다.
N3의 CI·모든 실패 조합·실제 Probe 수락/거부는 원래 C40 CSV에 보존했다.
Full Task Quality는 PENDING이다. 이 보고서는 전체 캠페인의 종료나 Serving GO를 선언하지 않는다.

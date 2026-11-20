# C0 Controlled와 C3 Continuous Serving Bridge

상태: `CLOSED / HOLD`. 실제 Batch Matching과 정상 Continuous Serving을 구분한다.

## Baseline 결과

Full200, BI0, QPS2.35, Arrival Seed2071을 사용했다.
모든 Engine에 B8192/S1024와 KV128/42GiB를 명시했다.
각 설정의 독립 Start는1회다. 다음 표에는 Start-level CI가 없다.
네 Baseline 모두 사전 Task-quality CI Gate를 통과했다.

| 정상 Continuous Baseline | TTFT p95 | ITL p99 | TPOT p95 | Raw Output tokens/s |
|---|---:|---:|---:|---:|
| Native Compile0 | 17.905s | 13.945ms | 8.325ms | 45.050 |
| Native Default Compile/Graph | 15.152s | 18.153ms | 8.109ms | 47.047 |
| Agg DP2 Compile0 | 2.014s | 344.340ms | 261.570ms | 55.322 |
| Agg DP2 Default Compile/Graph | 1.877s | 322.287ms | 239.992ms | 54.222 |

Default는 Compile/Graph 선택을 뜻한다. B/S와 KV의 자동 기본값을 뜻하지 않는다.
Agg는 Full-model Replica2개와 Round-robin을 사용했다.
일반 Continuous Batching과 Chunked Prefill을 유지했다.
C3에는 외부 C0 Source Credit4 제한을 적용하지 않았다.
실제 Prefill/Decode Batch는 Runtime Query Log에 기록했다.
이를 C0의 실제1/1/1과 같은 Batch라고 표시하지 않는다.

## Layer Bridge의 차단 원인

첫 Layer N8은 D Hidden Slot1의 Credit Exhaustion으로 실패했다.
이는 Layer가 원래 Batch를 만들 수 없다는 근거가 아니다.
D Pool을8 Slots로 늘리는 Correctness-only Intervention을 실행했다.
이후 실행은 Partial-layer Prefill과 Full36 Decode의 Mixed-row Assertion에서 실패했다.
Assertion을 제거하지 않았다.
현재 Adapter는 이 Mixed Shape를 지원하지 않는다.
따라서 적격 C3 Layer 성능 비교는 `NOT_RUN`이다.

## Token Bridge의 범위

Token N8의 전송 Prefix KV Byte Equality를 검사했다.
N32 Lifetime/Query 검사와 N200 Task-quality 실행은 Correctness-only다.
C3의 전체 Same-geometry Hidden·Greedy Teacher는 아직 `NOT_RUN`이다.
Token의 진단 Timing을 적격 C3 Layer-vs-Token Main Table에 넣지 않는다.

## 판정

C0의 TTFT 차이만으로 C3의 Method 우위를 주장할 수 없다.
Native/Agg의 정상 Continuous Baseline은 확보했다.
Layer의 적격 Continuous Bridge는 현재 구현에서 차단됐다.
이 제한은 Layer-axis의 근본적인 불가능성을 뜻하지 않는다.

[Continuous Baseline 결과](dense_ar_bi0_axis_rootcause_20261120/analysis/C3_continuous_bridge_results_v1.json),
[Layer Follow-up](dense_ar_bi0_axis_rootcause_20261120/analysis/C3_layer_bridge_followup_v1.json)을 참조한다.

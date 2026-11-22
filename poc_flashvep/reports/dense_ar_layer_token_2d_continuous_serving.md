# LongBench400 Continuous PD: Runtime Contract

작성일: 2026-11-21 KST.
상태: Runtime 구현과 정합성 검사를 완료했다.
Strong Static의 원래 Main400/Heldout200 최종140 Starts를 완료했다.
이후 등록 수정의 Natural400 독립 확인도 QPS1.8/2.8에서 각각 완료했다.
이 문서의 Discovery 수치를 최종 Serving 성공으로 취급하지 않는다.

## C0와 C3

C0는 실제 P Prefill / D Continuation / D Decode Batch1/1/1 Control이다.
C3는 설치한 vLLM Scheduler의 Continuous Batching을 사용한다.
두 Protocol은 다른 실행 계약이다.
C0 성능을 C3 성능으로 옮겨 쓰지 않는다.

모든 Worker는 B8192/S1024를 명시한다.
이 값은 Scheduler 상한이다.
실제 GPU Batch와 같지 않다.
Actual Query, Padded Query, Prefill Request 및 Decode Request 수를 별도로 저장한다.
Chunked Prefill과 KV42GiB/Block128도 유지한다.

## Mixed-row 수정

기존 C3 실패는 Batch 전체에 같은 Partial-layer 시작점을 적용했다.
Partial Prefill과 Full36 Decode가 섞이면 그 가정이 성립하지 않는다.
새 Adapter는 Request별 Token 위치와 Layer 시작점을 사용한다.
각 Layer에서 실제로 계산해야 하는 Query Span을 만든다.
그 Span에 맞는 Attention Metadata도 새로 만든다.
원래 Assertion을 삭제해서 실행을 통과시키지 않았다.

Decode Query는 항상 Full36 Layer를 방문한다.
P에서 수행한 Prompt Token-layer Work를 D에서 중복하지 않는다.
Genuine Hybrid는 Prefix Upper Layers와 Suffix Full36의 의존 순서를 지킨다.
짧은 Hybrid의 별도 Fusion 경로도 독립 Teacher로 검사했다.
그 경로는 L≤8192에만 적용한다.

## Numerical 및 Runtime Family

| Family | 원래 Full-model Compile | Partial Prefill | Source Admission |
|---|---|---|---|
| NativeStrong | 유지 | 없음 | Uncapped Coordinator Admission |
| Stock Token | 유지 | Full36 Suffix | 기본 Strong Arm은 Uncapped |
| Row-aware Layer/Hybrid | Compile Mode0 | Row-aware Eager | 기본 Credit4 |
| Preserved-compile Layer | Full-model Call에 유지 | Row-aware Eager | 기본 Credit4 |

Full Decode의 CUDA Graph와 Partial Prefill의 Compile을 구분한다.
Row-aware Mode0도 Full Decode Graph를 사용할 수 있다.
Partial P Forward가 원래 Full-model Compiler와 같다고 주장하지 않는다.
이 차이는 실제 Runtime의 구현 한계다.
Matched C0 Control과 C3 Best-versus-best Serving을 별도로 평가한다.

BI0의 실행 Geometry가 다르면 모든 Token의 Bitwise 일치를 보장하지 않는다.
같은 Numerical 경로에서는 Strict KV/Hidden/Greedy 비교를 사용한다.
다른 경로는 State Coverage, Import 및 동결한 Task-quality Gate로 평가한다.
실패한 Strict 비교를 지우거나 Tensor를 덮어쓰지 않는다.
자세한 한계는 `dense_ar_longbench400_bi0_numerical_reference_audit.md`에 기록했다.

## Credit와 Hidden Pool

Source Credit은 실제 GPU Batch 수가 아니다.
Coordinator는 P Dispatch에서 Credit을 획득한다.
첫 Client Output에서 Credit을 반환한다.
P Ready 또는 KV ACK만으로 이 Credit을 반환하지 않는다.
C3의 Credit0은 별도 Coordinator Outstanding 제한을 끈 값이다.
Worker의 B/S, KV Capacity와 Admission 제한까지 없앴다는 뜻은 아니다.
또한 C0의 별도 Credit 설정을 0으로 바꿨다는 뜻도 아니다.

Hidden Source Slot은 별도의 자원이다.
Source Slot은 완료한 READ ACK 이후 재사용한다.
D Slot은 Continuation의 CUDA Event가 완료된 후 재사용한다.
기본 Layer/Hybrid의 Source/Destination Pool은 각각 4 Slots다.
Stock Token과 Native는 이 Hidden Pool을 할당하지 않는다.

추가 Credit 검사는 Pool16을 고정하고 Credit1/4/8/16을 변경한다.
같은 Axis 안에서 Credit과 Buffer 크기를 동시에 바꾸지 않는다.
Credit1은 직렬화 Control이다.
이를 기본 Serving 후보로 자동 선택하지 않는다.
Credit1도 첫 Output 이후 다음 P와 이전 Decode를 겹칠 수 있다.
따라서 End-to-end Serial 실행과 같지 않다.
Pool32는 실제 Free Memory와 동결한 2GiB Reserve를 먼저 검사한다.
Reserve가 부족하면 KV42GiB를 줄이거나 검사 없이 할당하지 않는다.

## Decode Tail 원인 검사

기존 C3는 큰 Continuation과 Decode를 같은 Host Step에 넣었다.
Original Stock Token의 Compile만 유지해도 ITL Tail이 남았다.
따라서 Compile 차이만으로 Tail을 설명하지 않는다.

첫 개입은 큰 Prefill 계산을 Active Decode 이후로 미뤘다.
설치한 Scheduler의 pre-allocation throttle 경로를 사용했다.
이미 할당한 Token 수는 사후 변경하지 않았다.
기존 Allocation 및 KV Lifetime 검사를 유지했다.

두 번째 개입은 Remote-KV READ 준비를 Active Decode 이후로 미뤘다.
Model, Sampling, 계산 작업량 및 Client Timestamp는 바꾸지 않았다.
ACK, Heartbeat 및 KV Retirement Metadata는 그대로 전달했다.
현재 설치한 Pull Worker의 READ 호출은 `start_load_kv`에서 수행한다.
Handshake Thread는 Ready Queue를 사용한다.
이 검사는 현재 Single-process Actor와 고정 Runtime에 한정한다.
다른 Worker Thread 구조의 일반적인 안전성을 입증하지 않는다.

Whole-request D-quiescent Admission은 별도 Control이다.
이는 Prefill 계산만 미루는 정책과 다르다.
D Running Cap1/2도 별도 Admission 개입이다.
Cap2가 모든 Decode Step의 Actual Batch2를 보장하지 않는다.

### 별도 Source의 Request 등록 제어

후속 Source는 새 Request의 D 등록 시점도 제어한다.
Output이 시작된 Running Request가 있으면 새 `engine.add_request` 호출을 미룬다.
IPC 수신, 기존 Decode, ACK, Heartbeat와 Retirement는 계속 처리한다.
Stop 전에 미룬 Request를 모두 처리한다.
CPU 검사는 FIFO 순서, Control 전달과 원래 Output 객체의 보존을 확인한다.
GPU 검사는 별도 Numerical Family의 Natural32/400 State, Coverage, Lifetime과 Task Quality를 확인한다.

미룬 Request는 별도 Host 등록 Queue에 남는다.
vLLM Queue가 비어 있다는 사실로 이 대기를 숨기지 않는다.
Receive→Release 대기는 Coordinator의 이전 대기 또는 IPC 전송 시간을 포함하지 않는다.
실제 Queue 길이와 대기는 별도 Event로 재구성한다.

원래 140-start Confirmation은 변경하지 않았다.
새 정책은 별도 Source와 새로운 Arrival Seed3개를 사용하는 Adaptive Follow-up이다.
QPS1.8과 2.8을 별도 검사한다. 두 Load의 동결한 Cut, Credit, Cap과 Fusion 설정은 서로 다를 수 있다.
따라서 두 Load의 차이를 QPS만의 인과 효과로 해석하지 않는다.
같은 Load의 Token 등록-only A/B는 다른 설정을 유지한다.

새 Source가 정합성을 통과해도 Decode 비열등성을 자동 통과하지 않는다.
NativeStrong과 같은 정책 Native를 모두 대조군으로 사용한다.
최종 성능 CI와 Absolute SLO 판정은 두 Load의 별도 결과에 기록한다.
이 보수적인 등록 정책은 미래 Window를 예측하는 Dynamic Split-size Scheduler가 아니다.

### Client 전달 지연의 남은 한계

일부 큰 Client ITL은 Worker Token 간격과 다르다.
원래 Source의 `C3Bridge_I3_O64_Cap1_s1_StockToken_v1`에서 한 Client 간격은 384.483039ms였다.
같은 두 Token의 Worker Notification 간격은 8.083349ms였다.
이 간격은 기록된 Mixed Step, Offload Prefill, NIXL Host 또는 Hidden READ와 겹치지 않았다.
따라서 이 384.483039ms를 D GPU 계산 시간으로 해석하지 않는다.

원래 Ingress는 역할별 Sender Thread와 Queue를 사용한다.
동기 Socket Send가 Coordinator의 직접 호출 경로를 막았다는 설명은 Source와 맞지 않는다.
현재 CPU Pressure Snapshot이나 충분한 Free Memory도 과거의 단일 지연 원인을 증명하지 않는다.
이 드문 Client-only 전달 지연의 구체적 원인은 `NOT_IDENTIFIED`다.

Primary ITL은 실제 Client Timestamp를 유지한다.
Worker Timestamp로 바꾸거나 큰 Interval을 제거하지 않는다.
N16에서 한 Request가 Absolute SLO를 위반하면 Attainment는 최대 15/16이다.
따라서 작은 Mechanism Cell의 p99 Gate와 Absolute Attainment를 둘 다 보고한다.
새 등록 정책의 p99 개선도 모든 Interval의 34ms 상한을 보장한다는 뜻이 아니다.

근거는 해당 Raw의 `requests.jsonl`과 다음 분석에 있다.
[Token Delivery Tail](dense_ar_longbench400_2d_20261121/analysis/C3Bridge_I3_O64_Cap1_s1_StockToken_v1_decode_tail_audit.json).

## 완료한 정합성 검사

- C0의 26개 Pure/Interior Action은 N8 State Teacher 검사를 통과했다.
- C3 Layer, Token, Hybrid는 Natural400 Coverage, Import 및 Lifetime 검사를 통과했다.
- Native Fallback과 여러 Axis를 섞은 Natural400도 별도로 통과했다.
- Compute-only 제어의 네 Numerical Family는 Natural400 Gate를 통과했다.
- READ 준비 제어의 Native, Layer, Stock Token, Hybrid도 Natural400 Gate를 통과했다.

검사 단위와 Numerical 경로는 각 Gate에 명시한다.
Normal Completion만 검증했다.
Cancellation 및 Remote Failure 복구는 아직 검증하지 않았다.
PoC를 Production-ready Fault-tolerant Runtime으로 표현하지 않는다.

## 최종 성능 판정 계약

Static 후보는 Discovery200에서 선택한다.
각 Family에서 정확도와 두 Decode Point 검사를 통과한 후보를 비교한다.
NativeStrong과 같은 D 정책의 Native를 둘 다 대조군으로 사용한다.
최종 선택은 Held-out200 Latency 실행 전에 동결한다.

선택한 새 Source 조합은 Full400 정합성을 다시 검사한다.
이후 독립 Server Start 3회의 Paired CI를 계산한다.
ITL p99와 TPOT p95 CI 상한이 각각 1.05 이하여야 한다.
Absolute SLO도 바꾸지 않는다.
Request 또는 Token 수를 독립 Start 수로 사용하지 않는다.

Held-out200의 Primary 확인은 GPU0/1에서 수행한다.
Main400의 추가 Start는 다른 동등 H100 Pair를 사용할 수 있다.
그 경우 GPU Pair와 Arrival Seed Block을 명시한다.
Natural Output 수가 달라지면 실제 생성 수와 Task Score를 함께 보고한다.

### Pair와 Clock의 해석 범위

모든 Pair는 같은 H100 SKU와 NVLink Topology를 사용한다.
하지만 이를 모든 실행 시점의 동일한 Clock 또는 Service Time으로 해석하지 않는다.
Clock과 Power 상태는 초기화, 계산 및 대기 단계에 따라 달라진다.
운영 Snapshot에서 모든 Card의 Power Limit은 700W였다.
일부 P Card에는 Software Power Cap이 표시됐다. Clock이나 Power 설정은 변경하지 않았다.
이 Snapshot은 과거 지연의 원인 검사 또는 Pair 간 성능 동등성 증명이 아니다.

초기 간섭 Witness는 GPU0/1의 Serial1회와 병렬3회를 비교했다.
선언한 5% Threshold는 통과했다. 그러나 3회 대 3회의 CI는 아니다.
이 검사로 이후 모든 Natural Workload의 간섭이 0이라고 주장하지 않는다.
최종 Paired 비교는 같은 Arrival Block에서 같은 GPU Pair를 사용한다.
C0의 Input Bucket은 서로 다른 Pair를 쓸 수 있다.
따라서 서로 다른 Input 행의 차이를 Input Length만의 순수 인과 효과로 단정하지 않는다.

근거: `analysis/pair_interference_preflight_v1.json`과 `state/operational_clock_sample_20261122_0726.json`.
두 경로는 `dense_ar_longbench400_2d_20261121` 내부 경로다.

Static Runtime의 Decode 보호와 Dynamic Scheduling 이득은 다른 결과다.
현재 Active-decode Safe Window Predictor는 검증하지 않았다.
따라서 그 Predictor를 사용하는 위험한 Online 삽입은 실행하지 않는다.

## 재현 근거

- Runtime: `adapter.py`, `stock_token_adapter.py`, `compiled_full_adapter.py`.
- Policy: `decode_prefill_throttle.py`, `remote_setup_deferral.py`.
- Lifetime: `hidden_transport.py`, Request별 NIXL/Hidden ACK Log.
- CPU: `cpu_final_tests.py`, `cpu_regression.py`.
- Final 설계: `configs/final_continuous_confirmation_contract_v1.json`.
- Final 선택기: `final_continuous_selection.py`.
- Final 실행기: `final_confirmation_jobs.py`, `final_confirmation_execute.py`.
- Final 분석기: `final_confirmation_analysis.py`.

Config와 Raw 경로는 `dense_ar_longbench400_2d_20261121` Namespace 기준이다.
최종 성능 수치와 판정은 별도 Summary에 기록한다.
원래 실패와 별도 수정 Source의 결과를 혼합하지 않는다.

## Whole-prefill Deflection의 Host 경로

강한 Whole-prefill Deflection 대조군도 최종 Table에 유지한다.
일부 Request는 P를 거치지 않고 D에서 Full36 Prefill을 수행한다.
그 Request에는 P 제출, P 계산과 P→D Handoff 경계가 없다.
이 단계를 0ms로 만들어 고정된 6-stage 표에 넣지 않는다.

Route-aware 분석은 `P_FULL36_THEN_D`와 `D_LOCAL_FULL36`을 구분한다.
각 Route의 실제 경계 합은 같은 Client TTFT와 일치해야 한다.
Reference도 동일 Request ID 집합으로 비교한다.
Route 선택은 현재 P/D 상태를 사용한다. 조건부 Route 비교는 무작위 배정의 인과 효과가 아니다.

기존 CPU 분석기는 모든 Request에 P 제출이 있다고 가정했다.
실제 완료 Raw에서 `KeyError(q1,prefill)`을 재현했다.
새 회귀 검사는 200개 중 P→D141개와 D-local59개의 경계를 모두 확인한다.
이는 CPU 보고 경로의 수정이다. GPU 결과, Output과 Timestamp는 변경하지 않았다.
근거: `analysis/deflection_host_cpu_red_green_v1.json`.

## Natural400의 수정 후 범위

Request 등록 수정 후 QPS1.8과2.8은 각각 독립 Start3회를 완료했다.
Layer, Stock Token과 Hybrid는 두 Native 기준의 상대 Decode5%와 매 Start의 Absolute Decode≥95%를 통과했다.
두 Native 기준은 원래 NativeStrong과 정확한 공통 D 정책 Native다.
그러나 같은 Natural400에서 Layer와 Hybrid의 TTFT는 강한 Stock Token보다 느렸다.
이는 Safety 회복이다. Layer-specific 또는 Hybrid-specific Serving Win은 아니다.

고정 Output16/Cap1의 후속 검사에서는 Layer와 Hybrid가 원래 NativeStrong의 ITL 기준을 실패했다.
Natural400의 PASS를 모든 고정 Output 또는 모든 실행 계약의 PASS로 확대하지 않는다.
D Compile만 복원하는 별도 실험은 Strict KV·Hidden Gate를 실패했다.
이를 빠른 Decode 구현으로 채택하거나 실패한 성능 Gate를 덮어쓰지 않았다.
후속 개입은 Compile에 따른 IR Norm Provider 우선순위 차이를 확인했다.
두 Norm Provider를 원래 CUDA 우선순위로 맞추면 두 Axis의 Strict N1/N8과 Mixed32·Natural400이 통과했다.
두 Norm 각각의 수치 기여율은 분리하지 못했다. 이 정합성 통과를 Serving 속도 향상으로 해석하지 않는다.
Partial-prefill만 바꾼 별도 Source의 Hybrid는 Strict State를 실패했다.
그 Source의 예정 성능21회는 첫 실행 전에 NOT_RUN으로 마감했다.
최종 제한적 경로는 모든 Prompt를 Eager/CUDA로 유지하고 Decode-only를 원래 Native Compiled Model로 실행한다.
이 경로의 독립 반복은 별도 Source와 보고서로 판정한다.

최종 반복21회는 모두 완료했다. 수정 Hybrid는 두 Native와 Absolute Decode Gate를 통과했다.
하지만 수정 Layer의 Common Native ITL CI 상한은1.065921로 남았다.
수정 Hybrid도 강한 Token보다 TTFT에서 느렸다. 따라서 모든 조건의 Decode 복원이나 Layer/Hybrid-specific Strong GO를 선언하지 않는다.

근거:

- [Natural400 QPS1.8 등록 수정](dense_ar_longbench400_request_registration_tail.md).
- [Natural400 QPS2.8 등록 수정](dense_ar_longbench400_request_registration_tail_q28.md).
- [동일 조건의 Controlled Bridge](dense_ar_longbench400_registration_controlled_bridge.md).
- [D Compile의 State 검사와 Negative Control](dense_ar_longbench400_decode_compile_residual.md).
- [최종 Prefill/Decode Kernel 정책의 독립 검증](dense_ar_longbench400_eager_prefill_native_decode_results.md).

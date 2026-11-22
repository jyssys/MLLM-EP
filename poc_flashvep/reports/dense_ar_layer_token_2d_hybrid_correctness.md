# Genuine Layer×Token Hybrid 정합성

## 2026-11-21 04:15 KST Checkpoint

이 문서는 새 LongBench400 캠페인의 결과다. 이전 연구 결과를 새 실측으로 재사용하지 않는다.
현재 판정은 `CORRECTNESS_ONLY`다. Online 성능과 Decode Safety는 별도 Gate다.

모델은 Qwen3-8B다. Revision은 `b968826d9c46dd6066d109eabc6255188de91218`이다.
계산은 Original BF16, BI0, Exact Causal Attention이다.
모든 Worker는 B8192/S1024를 명시한다. KV Budget은 GPU당 42 GiB다.

## 독립 Reference와 Work Conservation

P는 Prefix의 Lower Layers를 계산한다. D는 Prefix의 Upper Layers와 Suffix의 Full36 Layers를 계산한다.
Decode는 Full36 Layers를 계산한다. 각 Request의 Query Interval과 Layer Visit를 기록한다.
검사기는 Interval의 누락과 중복을 모두 거부한다. Padding Row는 유효 Query로 계산하지 않는다.

Reference는 원래 Full36 모델을 실행한다. Reference에는 Partition Adapter와 GPU 간 Handoff가 없다.
Token Boundary가 달라지는 경우, 같은 Boundary로 원래 Full36 모델을 실행한다.
BI0의 다른 Query Geometry에서 발생한 수치 차이를 계산 오류로 단정하지 않는다.

동일 Geometry에서는 다음 항목을 검사한다.

- 모든 유효 Prompt KV Byte의 Digest 일치
- 마지막 Prompt Query의 Final Hidden 일치
- Greedy Output 16 Tokens 일치
- P/D Query와 Layer Coverage
- NIXL ACK, KV Retirement와 Hidden Buffer Lifetime

## 완료된 검사

| 검사 | 범위 | 결과 | 증거 수준 |
|---|---|---|---|
| Q4 Pure Axis | Last2/4/8/12, 각 Axis N8 | 모든 KV·Final Hidden·Greedy16 일치 | CORRECTNESS_ONLY |
| Q4 Hybrid | m4/s≈L/8, m0/s≈L/8, m4/s0, 각각 N8 | 내부 분할과 Pure Boundary 일치 | CORRECTNESS_ONLY |
| Q7 Family | 26 Actions × N8, 총 208개 실행 | 208개 모두 일치 | CORRECTNESS_ONLY |
| Q5 Natural32 | Layer4, Token4, Hybrid4/s≈L/8 | Coverage·Lifetime와 Task Quality Gate 통과 | CORRECTNESS_ONLY |
| Q9 GPU Digest | 독립 Boundary Teacher와 Hybrid4, 각각 N8 | 모든 KV·Final Hidden·Greedy16 일치 | CORRECTNESS_ONLY |

Q7은 서로 다른 원본 Prompt 8개를 각 Action에서 실행했다. 원본 Context는 Main400과 겹치지 않는다.
208개 실행은 독립 Prompt 208개나 독립 Server Start 208회가 아니다.
48개 Full36 Boundary Case를 Reference로 사용했다.
Q7 Gate는 C0 Serial 실행만 적격으로 판정한다. C3 Mixed-row 적격성을 자동으로 부여하지 않는다.

Q5의 Mixed Batch 수는 Layer 31개, Token 19개, Hybrid 59개다.
Q5는 모든 Request의 Full-KV Byte Capture를 실행하지 않았다.
Full-KV 검사는 Q4와 Q9의 별도 검사에서 확보했다.
Q5의 Task Quality는 동결한 2 percentage point 비열등성 기준을 사용한다.
그 CI의 단위는 Task 안의 Context Cluster다. Server-start 성능 CI가 아니다.

## Query1 CUDA Graph 오류와 수정

8192-token Chunk 뒤에 1-token Partial Prefill이 남으면 오류가 발생했다.
기존 Dispatch는 이 Query1을 Full36 Decode Graph로 실행했다.
그 실행은 필요한 Partial-layer Hidden Metadata를 만들지 못했다.
이 오류는 Assertion 제거로 수정하지 않았다.

새 Guard는 Partial Prefill의 Full36 Graph Replay를 막는다.
Full36 Decode의 Graph Replay는 유지한다.
Layer Prefix8193, Token Prefix8193, Hybrid Prefix8193을 각각 GPU에서 검사했다.
수정 전 Layer 최소 재현은 실패했다. 수정 후 세 경우는 모든 KV·Final Hidden·Greedy16이 일치했다.

## Full400 Observer 실패와 대체 검사

`C3Q6_Layer4_Natural400_FullState_v1`은 Serving Timeout 2100초에서 실패했다.
완료 Request는 237/400개다. 이 결과를 Full400 PASS로 처리하지 않는다.
실패 로그와 중간 State는 보존한다. 이 실행의 지연을 Primary 성능으로 사용하지 않는다.

새 Observer는 GPU에서 4096-byte Leaf마다 SHA256을 계산한다.
CPU는 길이와 Domain을 포함한 Leaf Digest의 Root를 계산한다.
Algorithm 이름은 `SHA256_OF_SHA256_4096B_LEAVES_V1`이다.
이 값은 Raw Byte Stream의 SHA256과 다르다.

Synthetic Byte·Dtype·비연속 Tensor 검사를 통과했다.
Q9에서는 실제 모델 Tensor의 모든 Leaf를 CPU `hashlib`와 비교했다.
모든 KV와 Hidden/Residual 검사가 일치했다.
이 Observer는 Tensor를 읽는다. 모델 Tensor나 Output Token을 수정하지 않는다.
Full400 대체 검사는 별도 v2 Namespace에서 진행한다.

Q9 v1은 Observer의 잘못된 KV Shape 가정에서 실패했다.
실제 유효 KV는 `[L,8,256]` BF16이며 Token당 4096 bytes다.
v2는 이 실제 ABI와 Byte Count를 검사한다.
이전 Raw-KV 검사는 이미 이 전체 Byte 구간을 읽었다.
Observer Assertion 실패를 기존 KV 누락의 증거로 해석하지 않는다.

## 현재 제한

Full400 Continuous 정합성은 이 Checkpoint에서 미확정이다.
별도 추가 Cut와 Heterogeneous Mixed-row 검사는 진행 중이다.
208개 Serial 검사의 성공은 Hybrid Online 이득을 입증하지 않는다.
Decode-safe Window와 Native-relative 5% 비열등성도 이 문서의 완료 검사에 포함되지 않는다.

## 재현 자료

- [Q4 Gate](dense_ar_longbench400_2d_20261121/state/C0_full_state_q4_gate.json)
- [Q7 비교](dense_ar_longbench400_2d_20261121/analysis/family_q7_full_state_comparisons.json)
- [Q7 Gate](dense_ar_longbench400_2d_20261121/state/C0_family_q7_gate.json)
- [Query1 GPU 비교](dense_ar_longbench400_2d_20261121/analysis/query1_boundary_full_state_comparisons.json)
- [Q9 비교](dense_ar_longbench400_2d_20261121/analysis/gpu_digest_q9_comparison_v2.json)
- [Full400 실패 Receipt](dense_ar_longbench400_2d_20261121/raw/C3Q6_Layer4_Natural400_FullState_v1/dispatcher_result.json)
- [Full400 Observer 변경 계약](dense_ar_longbench400_2d_20261121/configs/continuous400_observer_revision_v2.json)

## 2026-11-22 추가 검증

위의 미확정 문장은 11월21일 04:15 Checkpoint 범위다.
후속 GPU Digest 검사에서 Layer4, Token4, Hybrid4+L/8의 Natural400 Gate를 통과했다.
기존 Timeout 실패는 그대로 보존했다.

Row-aware C3는 같은 Batch의 Partial Prefill과 Full36 Decode를 지원한다.
Request별 Query Span과 필요한 Layer 구간을 따로 계산한다.
같은 Batch의 다른 요청에 같은 Cut을 적용하지 않는다.
전체 Span이 이미 맞는 경우에는 원래 Attention Metadata를 재사용한다.
부분 Span일 때만 해당 Span의 새 Metadata를 만든다.

Native와 여러 Axis가 섞인 Heterogeneous Natural32/400도 별도 Gate를 통과했다.
보수적 Compute Throttle을 사용한 검사에서는 Meaningful Mixed Prefill/Decode Batch가 0개였다.
따라서 이 후속 검사를 실제 Mixed Decode+Prefill GPU 비용의 증거로 사용하지 않는다.
원래 Mixed-row Gate와 새로운 Safety Policy Gate는 서로 다른 증거다.

Compute/READ 정책은 Native, Layer, Stock Token, Hybrid의 Natural32/400에서 검증했다.
정상 완료 시 Pending READ, KV Credit과 Hidden Pool Lifetime을 확인했다.
이를 Cancellation이나 Transport Failure 복구의 검증으로 확대하지 않는다.

최종 Serving Source Capsule은 `a2d5ebc4cb39eff66a8fbba3b4c0b83a8957d79e50c11ff94a529b2cdb2ca6e3`이다.
선택한 Runtime Composition마다 이 Source의 새 Natural400 Gate를 요구한다.
같은 Source와 같은 의미적 Signature가 있을 때만 앞선 Proof를 재사용한다.
정합성 Proof의 Latency로 Static 후보를 고르지 않는다.

Full400의 Import-prefix Digest와 Query/Layer Coverage는 모든 유효 State의 수명 검증이다.
여러 Batch Geometry의 모든 Output이 같은 Greedy Sequence라는 뜻은 아니다.
Strict Boundary Teacher와 Task-quality Non-inferiority를 구분한다.

별도 Preserved-compile Layer는 Stage-matched N1/N8/N32/N400 검사를 통과했다.
Preserved-compile Hybrid는 엄격한 Hidden 비교에서 실패했다.
실패한 Hybrid의 성능은 Primary에서 제외했다.
추가 Fractional Stock Token 후보도 엄격한 Teacher Gate를 통과하지 못했다.
이 실패를 숨기거나 Token을 덮어써서 일치시키지 않았다.
BI0 Numerical 차이인지 실제 계산 오류인지 미확정인 범위도 별도로 남겼다.

근거:

- [C3 Action Gate](dense_ar_longbench400_2d_20261121/state/C3_continuous_actions_gate_v1.json)
- [Heterogeneous400 Gate](dense_ar_longbench400_2d_20261121/state/C3_heterogeneous_native_axis400_gate_v1.json)
- [공통 READ Gate](dense_ar_longbench400_2d_20261121/state/D_remote_setup_deferral_gate_v1.json)
- [BI0 Numerical Audit](dense_ar_longbench400_bi0_numerical_reference_audit.md)

최종 Latency와 Decode 5% Gate는 이 정합성 문서와 별도로 판정한다.

## Request 등록 정책의 별도 정합성

새 등록 정책은 별도 Source Capsule을 사용한다.
SHA256은 `07cdfb84be94fc4ae3b1e778d58c8c1d03ff7a99452b0cd3ee667966600447cd`다.
원래 Source에서 바뀐 기존 파일은 Worker의 Connection 연결 한 곳이다.
새 `request_admission_deferral.py`가 등록 Queue를 제어한다.
Model, Tensor, Sampling 결과나 Client Timestamp를 덮어쓰지 않는다.

다음 QPS1.8 Composition은 각각 Natural32/400 Gate를 통과했다.
이 표의 상태는 모두 `CORRECTNESS_ONLY`다.

| Composition | Actual Source Credit | Natural32/400 |
|---|---:|---|
| NativeRegistration | 0 | PASS |
| NativeCompletionRegistration | 0 | PASS |
| LayerRegistration Last8 | 4 | PASS |
| TokenRegistration Last8, Stock Compile | 0 | PASS |
| HybridRegistration Last4+L/8, Short Fusion | 4 | PASS |

Credit은 GPU Batch가 아니다. 이 Composition은 Credit-matched Mechanism Control도 아니다.
각 Gate는 Import KV, Query/Layer Coverage, Normal Lifetime과 Task Quality를 확인한다.
Full400의 모든 Geometry가 하나의 Strict Greedy Sequence와 같다는 뜻은 아니다.
등록 Queue가 종료 시 모두 비었는지도 별도 확인한다.
CPU 검사는 FIFO, Cleanup Control과 원래 Output 객체의 전달을 확인한다.
Cancellation과 Remote Failure의 안전성은 여전히 `NOT_RUN`이다.

QPS2.8은 같은 Source라도 Credit, Pool, Cap과 Fusion Signature가 다를 수 있다.
정확히 같은 Signature의 Proof만 재사용한다.
나머지 QPS2.8 Composition은 새 Natural32/400 Gate가 필요하다.
위 QPS1.8 PASS를 QPS2.8 전체의 자동 PASS로 사용하지 않는다.

근거는 `state/request_registration_*_gate_v1.json`과 동결한 두 등록 정책 Plan에 있다.
모든 경로는 `dense_ar_longbench400_2d_20261121` 내부 경로다.

### 후속 Signature 검사 완료

QPS2.8의 등록 정책 Composition도 Signature별 Natural32/400 검사를 완료했다.
이후 Controlled Bridge가 요구하는 10개 Signature도 모두 통과했다.
그중 2개는 같은 Source와 정확히 같은 Signature의 QPS2.8 Proof를 재사용했다.
나머지 8개는 GPU6/7에서 새 Natural32 및 Natural400을 실행했다. 새 Server Start는16회다.
다른 Signature나 단순히 같은 Method 이름의 Proof를 대신 사용하지 않았다.

모든 Proof의 Source SHA256은 위 등록 정책 Source와 같다.
각 Proof의 Task Quality, State, Full36 Decode Coverage와 Lifetime을 별도로 확인했다.
이 상태는 `CORRECTNESS_ONLY`다. 고정 Output16/64의 Latency와 Decode 비열등성 통과는 별도 검사다.
Natural400 Proof 통과만으로 아직 실행하지 않은 성능 Cell을 PASS로 표시하지 않는다.

근거: [등록 수정 후 Bridge의 정확한 Signature와 Proof 경로](dense_ar_longbench400_2d_20261121/configs/request_registration_controlled_bridge_frozen_v1.json).

## 최종 Prefill/Decode Kernel 정책의 정합성

최종 V2 Source SHA256은 `2b004c881d5c2f7caf66d6c467957d43079e27a15694cae54b8ee90971504a9f`다.
모든 Prompt Stage를 Eager/CUDA Norm으로 유지한다. Decode-only는 원래 Native Compiled Model을 사용한다.
P의 기존 Mode0 경로는 바꾸지 않는다. Tensor와 Token을 보정하지 않는다.

Layer와 Hybrid는 각각 Strict N1/N8, 실제 Mixed32, Guarded Natural32 및 Natural400을 통과했다.
Strict Teacher는 같은 Stage별 Numerical 계약의 독립 Full36 Single-GPU 모델이다.
이 Teacher에는 Partition Adapter, NIXL과 Imported KV가 없다.
Full KV, Last Hidden과 Greedy16의 엄격한 비교를 먼저 통과했다.
후속 Natural400은 State·Import·Coverage·Lifetime·Task-quality를 검사했다. 성능 Gate가 아니다.

같은 최종 Source의 NativeStrong, NativeCommon, Layer, Hybrid와 Stock Token 대조군도 Natural400을 통과했다.
기존 경로의 Numerical Family Proof와 새 Source의 실제 Proof를 구분했다.
모든 조건의 Relative Decode CI는 별도 독립 성능 반복으로만 판정한다.

Teacher의 원시 Scope 문자열 일부는 이전 Prefix-only Helper에서 상속됐다.
최종 Config와 실제 Event는 모든 Prompt Stage의 Eager 실행을 명시한다.
원시 문자열을 수정하지 않았다. 별도 Scope 감사에 그 Hash와 실제 계약을 보존했다.

V1은 Cold Dispatch의 CPU 검사에서 실패했다. GPU 실행은0회다.
그 버전의 예정38개는 모두 NOT_RUN으로 보존했다. 별도 V2만 실제 GPU로 검사했다.
Partial-only Source의 Hybrid Strict 실패와 예정 성능21개의 NOT_RUN도 지우지 않았다.

- [최종 V2 정합성 설계와 모든 Gate](dense_ar_longbench400_2d_20261121/configs/eager_prefill_native_decode_correctness_frozen_v2.json).
- [독립 Teacher의 실제 Scope](dense_ar_longbench400_2d_20261121/analysis/eager_teacher_effective_scope_v2.json).
- [최종 성능의 별도 판정](dense_ar_longbench400_eager_prefill_native_decode_results.md).

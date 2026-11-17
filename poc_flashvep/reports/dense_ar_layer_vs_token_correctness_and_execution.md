# Dense AR Layer/Token 이전의 실행·정합성 계약

## 현재 판정

실제 2GPU Token-axis P→D Continuation을 구현했다.
Suffix 512/1024/2048/4096 Tokens는 각각 N8과 N32 정합성 검사를 통과했다.
인접한 1776/1792 및 2656/2672 Token 경계도 N8과 N32를 통과했다.
결과는 `CORRECTNESS_ONLY`다.
이 결과만으로 Online 성능 우위를 주장하지 않는다.
새 Prompt 128개에서도 Suffix2048/2672/4096의 전체 Prompt KV·Hidden과 Output16이 일치했다.
Suffix2304/2400/2560은 별도의 N8과 N32 전체 State 검사를 통과했다.
Output16의 최종 Token2400 후보도 새 Prompt N128 전체 State Gate를 통과했다.
Output64의 최종 Token1024 후보도 새 Prompt N128 전체 State Gate를 통과했다.
이 두 별도 검사는 Output2를 사용했다.
최종 성능 실행은 Output16과 원래 Output Reference를 사용한다.

Layer Last0/2/4/8/12도 새 Evidence Namespace에서 N32를 통과했다.
새 Held-out Prompt 128개에서도 모든 Layer Cut의 전체 Prompt State와 실제 Layer 방문을 통과했다.
이 강제 Cut 실행은 Output2 정합성 시험이다.
강제 Cut 정합성 시험은 Online Dynamic 선택 결과가 아니다.

## 공통 환경

Qwen/Qwen3-8B Revision `b968826d9c46dd6066d109eabc6255188de91218`을 사용했다.
Original BF16, Exact Causal Attention과 Full36 Decode를 유지했다.
Primary GPU는 H100 SXM 80GB GPU0/1이다.
vLLM 0.26.0+cu129, PyTorch 2.11.0+cu129와 NIXL 1.4.1을 사용했다.
GPU별 KV Budget은 42GiB다.
Prefix Cache는 비활성화했다.
Greedy와 Thinking-disabled Prompt를 사용했다.

## 실제 Token-axis 실행

Prompt 길이는 8000 Tokens다.
P는 앞의 `8000−suffix` Tokens를 36개 Layer에서 계산한다.
P가 생성한 Prefix Sample은 최종 출력으로 사용하지 않는다.
P는 계산을 완료한 Prefix의 전체 Layer KV만 전송한다.
D는 원래 전체 Prompt와 정확한 Prefix Match 길이를 받는다.
D는 Prefix 이후의 Query만 36개 Layer에서 계산한다.
그 뒤 정상적인 Full36 Decode를 수행한다.

예를 들어 Suffix2048이면 P는 `[0,5952)`를 계산한다.
D는 `[5952,8000)`을 계산한다.
D의 Suffix Attention은 이전 Query의 전송된 KV를 사용한다.
두 Token 구간이 독립적이라고 가정하지 않는다.
P→D Dependency 때문에 같은 Request의 두 Prompt 구간은 직렬 의존성을 갖는다.
서로 다른 Request의 P와 D 계산은 겹칠 수 있다.

이 직렬 의존성은 이번 Completed-prefix 구현의 경계다.
Token-axis 자체가 항상 전체 Prefix 완료를 기다려야 한다는 뜻은 아니다.
구조적으로 D의 Suffix Layerℓ는 P의 Prefix KVℓ와 D의 Suffix Hiddenℓ−1에 의존한다.
P의 Prefix Layerℓ+1은 뒤쪽 Suffix Token에 의존하지 않는다.
따라서 Layer별 KV Fence가 있으면 두 GPU의 Token Pipeline이 가능할 수 있다.
이 Pipeline의 실제 구현과 성능은 `NOT_RUN`이다.

현재 설치된 NIXL의 `wait_for_layer_load`는 No-op이다.
Pull Worker는 Request 단위 KV Read와 완료를 관리한다.
검증한 Token 구현은 이 Request 단위 경로를 재사용한다.
Layer별 전송·완료·Partial Forward 계약은 새로 구현하지 않았다.
따라서 이 비교를 모든 Streaming Token Pipeline의 상한으로 해석하지 않는다.

구현은 기존 NIXL Pull 경로를 재사용한다.
Destination의 Remote Match만 실제 완료된 Prefix로 제한한다.
기존 경로처럼 전체 Prompt가 Remote-complete라고 표시하지 않는다.
Source Export의 Token 수와 Block 수를 검사한다.
Position과 RoPE는 원래 Request의 절대 위치를 유지한다.
ACK 이전에 Source KV를 해제하지 않는다.
새 KV Manager를 구현하지 않았다.
Token Arm의 Layer Cut Histogram에는 Last0이 표시된다.
이는 Layer-axis 이전을 하지 않는다는 뜻이다.
Token Migration0을 뜻하지 않는다.
실제 Suffix Tokens와 D Token-layer Rows는 별도 Work Accounting에 기록한다.

## Layer-axis 실행

P는 모든 Prompt Token에 대해 앞의 `36−m` Layer를 계산한다.
P는 해당 Layer의 KV와 Hidden/Residual을 전송한다.
D는 남은 m Layer를 계산한다.
이후 D는 전체 36개 Layer로 Decode한다.
P와 D는 모두 전체 Model Weight를 보유한다.
Weight를 영구 분할하는 PP2가 아니다.
중간 State를 무시하는 Optional Input 경로를 사용하지 않는다.
Compiled BF16 Boundary의 기존 부적격 경로도 사용하지 않는다.

## 검증 범위

각 Request에서 다음 항목을 확인했다.

- 원래 독립 Reference의 Greedy Output Token.
- 36개 Layer의 전체 유효 Prompt KV Hash.
- 실제 P/D Query의 연속성과 중복·누락 여부.
- P Prefix와 D Suffix를 결합한 전체 Final Hidden의 Byte Equality.
- 실제 Layer 방문 또는 검증된 Full36 Graph 정의와 Replay.
- NIXL ACK와 정상 KV Retirement.

Layer 검사는 기존 독립 Full36의 KV/Hidden과 비교했다.
Token 검사는 새 Native Whole-prompt State와 비교했다.
두 경우 모두 기존 200-token Original Output Reference도 검사했다.
Token Continuous B512 경로는 N32, Output64에서 전체 Prompt State를 통과했다.
정합성 Capture의 CPU Tensor 복사는 성능을 크게 교란한다.
이 Latency를 Clean Serving 결과에 사용하지 않는다.

## 동일 작업량 Control

Source Group은 Coordinator가 관리하는 Macro다.
Source Group8이 GPU의 한 Batch8 실행을 뜻하지는 않는다.
P slots4와 Token Budget32000에서는 8000-token Prompt를 최대4개씩 실행한다.
Source Macro8은 이 GPU 실행 두 개를 포함할 수 있다.
D Cohort와 실제 GPU Batch도 별도로 기록한다.
Geometry 일치 여부는 Raw Execution을 기준으로 판단한다.

`STRUCTURAL_ACCOUNTING`이다.
동일 GEMM Token-layer Rows를 정확히 맞추는 Token Plan을 작성했다.
Request144개와 FIFO Source Group4개를 한 주기로 사용한다.
각 Group 내부의 Suffix 길이는 동일하다.

| Layer Cut | 인접한 Token Suffix | 전체 144개에서의 조건 |
|---|---|---|
| Last2 | 432/448 | `36×sum(suffix)=144×8000×2` |
| Last4 | 880/896 | `36×sum(suffix)=144×8000×4` |
| Last8 | 1776/1792 | `36×sum(suffix)=144×8000×8` |
| Last12 | 2656/2672 | `36×sum(suffix)=144×8000×12` |

이 Control은 논리적인 Prefix KV Byte도 맞춘다.
Layer에는 Request별 131,072,000 Bytes의 Hidden/Residual이 추가된다.
Token Suffix는 늦은 Causal Query를 담당한다.
따라서 같은 GEMM Rows라도 Token의 D-side Attention Pair 수는 더 많다.
Attention FLOPs, 실제 GPU 시간, Handoff 완료시간까지 같다는 뜻은 아니다.
실제 실행 Group이 같은지는 별도 Online Geometry 검사로 확인해야 한다.

## 초기 실패와 검증기 수정

두 인접 Token 경계를 섞은 첫 N8 실행에서 Output·KV·Hidden은 일치했다.
검증기는 모든 Request에 하나의 Scalar Prefix를 적용했다.
그 결과 두 번째 Group의 Query Partition을 잘못 실패 처리했다.
원래 실패 결과와 당시 Source Hash를 보존했다.
검증기를 Request별 `suffix_for`로 수정했다.
동일 Raw Evidence를 다시 검사했다.
N8과 N32의 두 Plan이 모두 통과했다.
실패한 검증 기록을 삭제하지 않았다.

초기 Dynamic State Gate 네 개에는 검증 범위를 과도하게 넓힌 설명 문장이 있었다.
실제 Hash와 Layer 방문 검사는 실행한 Cut에만 적용됐다.
해당 Gate를 모든 Cut의 검증으로 해석하지 않는다.
별도 정정 파일에 실제 Cut 집합을 기록했다.
원래 Gate와 Raw Evidence는 수정하지 않았다.
모든 Last0/2/4/8/12를 확인한 N32·N128 시험은 별도로 존재한다.

## 근거

- [Token 구현](../dense_ar_layer_token_pd/token_axis.py).
- [Token 검증기](../dense_ar_layer_token_pd/qualification.py).
- [Layer N32 전체 State Gate](dense_ar_layer_token_dynamic_headroom_20261117/analysis/Layer_all_cuts_N32_full_state_gate.json).
- [새 Prompt Layer N128 전체 State Gate](dense_ar_layer_token_dynamic_headroom_20261117/analysis/Layer_N128_full_state_gate.json).
- [새 Prompt Token N128 전체 State Gate](dense_ar_layer_token_dynamic_headroom_20261117/analysis/Token_N128_state_qualification.json).
- [선정된 Token2400 N128 State Gate](dense_ar_layer_token_dynamic_headroom_20261117/analysis/selected_Token2400_state_N128_O2_v1_hash_state_gate.json).
- [선정된 Token1024 N128 State Gate](dense_ar_layer_token_dynamic_headroom_20261117/analysis/selected_Token1024_state_N128_O2_v1_hash_state_gate.json).
- [Token Pipeline Dependency와 Runtime 경계](dense_ar_layer_token_dynamic_headroom_20261117/analysis/token_pipeline_dependency_audit.json).
- [실제 D Prefill Batch8의 N32 State Gate](dense_ar_layer_token_dynamic_headroom_20261117/analysis/Layer12_actualD8_full_state_qualification.json).
- [초기 Gate 설명 범위 정정](dense_ar_layer_token_dynamic_headroom_20261117/analysis/state_gate_scope_corrections.json).
- [Token2048 N32 전체 State Gate](dense_ar_layer_token_dynamic_headroom_20261117/analysis/Token2048_state_N32_O16_v1_state_gate.json).
- [Token4096 N32 전체 State Gate](dense_ar_layer_token_dynamic_headroom_20261117/analysis/Token4096_state_N32_O16_v1_state_gate.json).
- [Continuous Token B512 N32/Output64](dense_ar_layer_token_dynamic_headroom_20261117/analysis/standard_Token_continuous_B512_state_N32_O64_v1_state_gate.json).
- [인접 Token 경계 N32](dense_ar_layer_token_dynamic_headroom_20261117/analysis/Token_plan_2656_2672_state_N32_v1_state_gate.json).
- [원래 검증 실패 보존](dense_ar_layer_token_dynamic_headroom_20261117/analysis/plan_gate_scalar_prefix_failure_preserved.json).
- [정확한 구조적 계산](dense_ar_layer_token_dynamic_headroom_20261117/analysis/matched_GEMM_rows_structural_accounting.json).

N128의 성능 실행에는 Output Gate를 적용한다.
모든 N128 성능 실행에서 전체 KV·Hidden을 복사한다고 주장하지 않는다.
성능과 전체 State Capture는 별도의 실행이다.
Held-out 성능 실행의 정상 종료 Credit도 별도 검사했다.
P/D KV Free Blocks는 초기값으로 돌아가야 한다.
Layer Reservation과 Hidden Pool은 실제 사용한 경로에서만 검사한다.
일반 Token 경로에 없는 Hidden Pool을 실패로 세지 않는다.
초기 Audit의 이 범위 오류도 별도 파일로 보존했다.
정상 종료 검사는 Cancellation이나 GPU Failure 복구의 검증이 아니다.

- [Held-out 정상 KV·ACK·Credit 검사](dense_ar_layer_token_dynamic_headroom_20261117/analysis/heldout_normal_lifetime_credit_audit.json).

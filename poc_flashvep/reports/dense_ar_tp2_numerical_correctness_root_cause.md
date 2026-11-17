# Qwen3-8B TP2 수치 정합성 원인 진단

## 판정

`NUMERICAL_NON_EQUIVALENCE_IDENTIFIED`다.
TP1 Strict Parity는 통과하지 못했다.
TP2는 이번 Primary 성능 Table의 적격 기준선이 아니다.

같은 Prompt와 같은 Token Prefix에서 최초 차이를 찾았다.
최초 Tensor 차이는 0-based Layer 0의 `self_attn.o_proj` 출력이다.
그 전의 Embedding, Input Norm, Q/K/V, Q/K Norm, RoPE와 Attention 출력은 수집한 Rows에서 일치했다.
이 경계의 TP2 BF16 부분합을 더하면 실제 TP2 출력과 정확히 일치한다.
동일 입력의 GPU Primitive Replay는 실제 TP1과 TP2 출력을 각각 정확히 재현했다.
따라서 이 경계의 차이는 잘못된 Weight Shard나 누락된 Rank 합산으로 설명되지 않는다.

이 결론은 전체 Runtime의 모든 오류 가능성을 배제하는 증명은 아니다.
TP2의 전체 Prompt KV 정합성은 별도 미측정 항목이다.
정상적인 BF16 TP 경로에서 TP1 Strict Parity를 회복하는 최소 수정은 확보하지 못했다.
Token을 덮어쓰거나 FP32를 Main 성능 경로에 적용하지 않았다.

## 환경과 비교 범위

Model은 Qwen/Qwen3-8B다.
Revision은 `b968826d9c46dd6066d109eabc6255188de91218`다.
Original BF16, Exact Causal Attention과 Full36을 사용했다.
vLLM 0.26.0+cu129, PyTorch 2.11.0+cu129와 H100 GPU0/1을 사용했다.
모든 Tensor Capture는 Eager, 순차 실행과 같은 정합성 보정을 사용했다.
Tensor 복사 실행은 `CORRECTNESS_ONLY`다.
이 실행의 Latency를 Clean Serving 성능으로 사용하지 않는다.
실제 TP Group은2 Ranks와 PYNCCL All-reduce를 사용했다.
Engine Log의 `disable_custom_all_reduce`는 참이었다.
Worker의 `BF16_reduced_precision_reduction`은 거짓이었다.
따라서 Local GEMM의 BF16 출력 반올림을 BF16 내부 누산과 혼동하지 않는다.
Collective 입출력 Tensor는 BF16이었다.
NCCL 내부 누산 Precision과 Ring/Tree의 실제 Kernel 선택은 `NOT_MEASURED`다.
PYNCCL 결과는 기록한 BF16 부분합의 수동 합과 일치했다.

원문 Prompt는 기존 Discovery Pool의 Request 4다.
Input은 8000 Tokens다.
기존 순차 N8 로그에서 가장 이른 Output Position 차이는 Request4의 Position6이었다.
Request ID 순서로 첫 불일치는 Request0의 Position40이었다.
기존 N8의 Position 분석은 `TRACE_ANALYSIS`다.
새 Tensor Capture를 8개 Request 전체의 재실행으로 표시하지 않는다.
출력 8개를 실제 Greedy 계산으로 생성했다.
0~6번 Step의 입력 Prefix는 두 실행에서 같다.
6번 Step에서 처음 Output이 달라진다.
그 뒤의 서로 다른 Prefix를 최초 원인 비교에 사용하지 않았다.
TP1과 TP2 Capture의 실제 출력 Prefix는 각자의 기존 비계측 Reference와도 일치했다.
이 확인은 출력 재현성에 한정한다.
Capture의 시간 간섭이 없었다는 뜻은 아니다.

| 항목 | TP1 | TP2 |
|---|---|---|
| 실제 출력 | 14374, 21517, 315, 279, 11789, 271, 1986, 2197 | 14374, 21517, 315, 279, 11789, 271, 785, 2197 |
| 6번 Step의 Token1986 Logit | 28.125 | 27.875 |
| 6번 Step의 Token785 Logit | 28.000 | 27.875 |
| Top1−Top2 Margin | 0.125 | 0 |

TP2에서는 두 Logit이 동률이 된다.
실제 Greedy Output은 Token785다.
Top-k 도구가 동률을 표시하는 순서는 실제 Argmax 규칙과 구분한다.

## 최초 Tensor 차이 — CORRECTNESS_ONLY

아래 값은 최초 Prefill Forward의 마지막 8개 Query Rows다.
Tensor Shape는 `[8, 4096]`이다.
전체 32768 Elements 중 12029개가 달랐다.

| Metric | TP1 대비 TP2 |
|---|---:|
| Max Absolute Error | 0.015625 |
| RMS Error | 0.00020561653946060687 |
| Relative L2 Error | 0.003151108743622899 |
| Max BF16 ULP 차이 | 14476 |
| Mean BF16 ULP 차이 | 10.311431884765625 |

큰 ULP 수치는 0 부근의 Cancellation 값에도 영향을 받는다.
이를 큰 절대오차와 같은 의미로 해석하지 않는다.
Max Absolute Error와 Relative L2를 함께 보고한다.
같은 경계는 후속 공통 Prefix Step 1~6에서도 최초 차이였다.

논리적인 TP Shard를 올바른 축으로 결합했다.
Q/K/V와 Gate/Up의 서로 다른 Packed Segment를 따로 결합했다.
Q/K Norm은 Head 축으로 결합했다.
RoPE Module은 Layer 간에 공유된다.
초기 Hook은 이 공유 Module의 마지막 호출을 덮어썼다.
이를 최초 원인으로 사용하지 않았다.
수정된 Hook은 실제 Layer Context를 기록한다.
Key별 쓰기 횟수가 1회인지 검사한다.
수정 후에도 최초 유효 차이는 Layer 0의 Output Projection이었다.

## 동일 입력 Primitive Replay — CORRECTNESS_ONLY

실제 Capture의 Projection 입력과 Weight를 사용했다.
두 TP2 Weight Shard를 K 축으로 합치면 TP1 Weight와 정확히 일치했다.
TP2 입력 Shard를 합쳐도 TP1 입력과 정확히 일치했다.

Prefill의 M=8 Replay와 원래 Shape인 M=8000 Replay를 모두 수행했다.
M=8000에서는 독립적인 입력 Rows를 반복했다.
마지막 8개 Rows의 실제 Operand를 유지했다.
Decode의 M=1은 Step 1과 6에서 검사했다.

네 Replay 모두 다음 검사를 통과했다.

- Full-K BF16 GEMM 결과가 실제 TP1 출력과 Byte Equality를 통과했다.
- 각 K/2 BF16 GEMM 결과가 실제 TP2의 Collective 입력과 Byte Equality를 통과했다.
- 두 BF16 부분합의 합이 실제 TP2 출력과 Byte Equality를 통과했다.
- 실제 Collective 출력이 두 부분합의 수동 합과 Byte Equality를 통과했다.

TP1과 TP2가 다른 결과를 만드는 위치를 재현했다.
각 부분 Projection은 Collective 전에 이미 BF16 출력으로 반올림된다.
Collective의 합산 순서만 바꾸어 이미 사라진 부분합 정보를 복구할 수 있다고 가정하지 않는다.

진단용 FP32 Replay의 Full-K와 Split-K 차이는 훨씬 작았다.
Prefill M=8의 Max Absolute Error는 `4.76837158203125e-07`이었다.
M=8000에서는 `3.5762786865234375e-06`이었다.
FP32도 Bitwise Equality를 통과한 것은 아니다.
이 진단을 Original BF16 성능 기준선으로 사용하지 않았다.

## 정합성 Gate와 남은 한계

| Gate | 판정 |
|---|---|
| TP1 Strict Greedy Parity | FAILED |
| 같은 TP2 경로의 Output 재현 | 기존 N8 반복 및 이번 Capture에서 확인 |
| 최초 Tensor 경계 | CORRECTNESS_ONLY: Layer 0 Output Projection |
| 같은 Operand의 수치 원인 Replay | CORRECTNESS_ONLY: 통과 |
| 첫 Layer의 Row Projection Weight/Input Shard | Byte Equality 통과 |
| 전체 TP2 Prompt KV와 전체 Hidden의 Within-TP2 비교 | NOT_MEASURED |
| TP2의 통일된 Coordinator Timestamp 성능 평가 | NOT_RUN |
| TP1 Strict Parity를 유지하는 정상 TP2 성능 기준선 | BLOCKED_CORRECTNESS |

정상적인 TP2 수치 차이를 무조건 잘못된 Transformer 계산으로 판정하지 않는다.
반대로 Within-TP2 Output 반복만으로 TP1과 동일한 Primary 기준선이라고 판정하지 않는다.
Primary 비교는 정합성이 검증된 TP1 기반 1P1D와 DP2를 사용한다.
남은 예산은 Layer/Token 이전과 Dynamic Headroom 검증에 사용한다.

## 근거

- [Tensor 비교](dense_ar_layer_token_dynamic_headroom_20261117/analysis/TP1_TP2_first_divergence.json).
- [TP1 실제 Capture](dense_ar_layer_token_dynamic_headroom_20261117/raw/TP1_first_tensor_witness_req4_v5/capture_export.json).
- [TP2 실제 Capture](dense_ar_layer_token_dynamic_headroom_20261117/raw/TP2_first_tensor_witness_req4_v4/capture_export.json).
- [실제 TP Rank 설정](dense_ar_layer_token_dynamic_headroom_20261117/raw/TP2_first_tensor_witness_req4_v4/worker_reports.json).
- [Byte-exact GPU Primitive Replay](dense_ar_layer_token_dynamic_headroom_20261117/raw/TP_row_projection_byte_exact_replay_v2/primitive_result.json).
- [Capture와 기존 출력의 비교](dense_ar_layer_token_dynamic_headroom_20261117/analysis/TP_capture_against_frozen_outputs.json).
- [기존 N8의 최초 Output 분기 위치](dense_ar_layer_token_dynamic_headroom_20261117/analysis/TP2_prior_N8_first_output_divergences.json).
- [실행 계약](dense_ar_layer_token_dynamic_headroom_spec.md).

각 Raw Directory에 Immutable Config, Source ZIP, Source Hash와 실제 Command를 보존한다.
Tensor `.pt` 원본도 보존한다.
큰 원본을 Archive로 공개할 경우 원본별 SHA256을 검증한다.
초기 Hook 오류와 Resource Guard의 Prelaunch 실패도 삭제하지 않는다.

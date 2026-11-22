# LongBench400 BI0 Numerical Reference Audit

작성일: 2026-11-21 KST.
이 문서는 Numerical 정합성의 범위를 기록한다.
Serving 성능 또는 Decode 보호의 최종 판정이 아니다.

## 두 종류의 불일치

첫 번째 불일치는 KV Digest 형식의 차이였다.
한 Observer는 Raw SHA256을 사용했다.
다른 Observer는 4,096-Byte Leaf SHA256의 SHA256을 사용했다.
이 두 Digest를 직접 비교할 수 없다.
수정한 비교기는 Algorithm 또는 Leaf 크기가 다르면 비교를 중단한다.
이 수정은 Model Tensor를 바꾸지 않는다.
이전 실패 기록은 보존한다.

두 번째 불일치는 실제 BF16 Tensor의 차이였다.
READ 준비 시점을 변경한 Native 및 Stock Token의 N8 검사에서 발생했다.
P에서 이미 차이가 발생했다.
해당 Request의 유효 P KV와 D Import KV는 정확히 일치했다.
따라서 이 차이를 Remote Transfer의 KV 손상으로 확정하지 않았다.

## 원래 Model 경로의 A/B 검사

동일한 Source와 GPU Pair에서 독립 Native N1을 네 번 실행했다.
원래 Model과 Full36 Layer를 사용했다.
Control 두 번과 READ 준비 제어 두 번을 비교했다.
정확도 진단용 Kernel Capture를 사용했다.
이 실행의 Latency를 Primary 성능으로 사용하지 않는다.

| 실행 | P KV 경로 | D KV 경로 | D Hidden 경로 |
|---|---|---|---|
| Original0 | B | B | B |
| READ 제어0 | A | A | A |
| Original1 | B | B | B |
| READ 제어1 | B | B | B |

경로 A와 B는 서로 다른 실제 BF16 값을 생성했다.
원래 Native Control에서도 경로 B가 발생했다.
따라서 경로 B를 READ 정책에만 생기는 계산 오류로 해석하지 않는다.

관측한 P Attention Kernel 이름과 실행 순서의 Hash는 네 실행에서 같았다.
이 Collector는 Grid Geometry를 저장하지 않았다.
Kernel 이름이 같다는 사실만으로 내부 Reduction 순서가 같다고 주장하지 않는다.
경로를 선택한 정확한 Operator는 `NOT_IDENTIFIED`다.
다른 Start 또는 Batch에서 Bitwise Invariance가 확보됐다고 주장하지 않는다.

## 독립 Teacher와의 검사

원래 Model의 경로 B를 사용하는 N8 Teacher를 별도로 실행했다.
Native Teacher는 원래 Full36 Prefill과 Full36 Decode를 수행했다.
Token Teacher는 원래 Full36 Model을 Single GPU에서 수행했다.
Token Teacher는 합법적인 Prefix Chunk 경계만 적용했다.
Teacher는 Partial-layer Adapter를 사용하지 않았다.

다음 두 비교는 `CORRECTNESS_ONLY` PASS다.

- Native N8의 Full Prompt KV, Final Hidden, Greedy Output이 독립 Native Teacher와 정확히 일치했다.
- Stock Token N8의 같은 세 항목이 독립 Full36 Token-boundary Teacher와 정확히 일치했다.

N1은 별도의 원래 경로 A Teacher와 정확히 일치했다.
이 결과는 경로 A와 B가 서로 같다는 뜻이 아니다.
다른 경로를 비교했던 이전 Strict 실패도 그대로 남긴다.
Token 또는 Tensor 값을 Teacher 값으로 덮어쓰지 않았다.

## FP32 진단

원래 Checkpoint를 FP32로 변환한 Transformers Model도 별도로 실행했다.
Prompt는 Qasper의 실제 3,987 Tokens였다.
Attention은 PyTorch SDPA MATH의 Exact Causal 경로였다.
TF32는 사용하지 않았다.
Missing, Unexpected, Mismatched Weight는 없었다.

| 진단 항목 | 경로 A | 경로 B |
|---|---:|---:|
| Final Hidden Relative RMS Error | 0.01106690 | 0.01349115 |
| Final Hidden Cosine | 0.99993879 | 0.99990916 |
| 공통 FP32 LM Head의 Argmax | 8973 | 8973 |

FP32 Reference의 첫 Logit Top1−Top2 Margin은 2.79922485였다.
위 Argmax는 측정한 BF16 Hidden에 공통 FP32 Head를 적용한 결과다.
실제 vLLM BF16 LM Head Logit Trace가 아니다.
후속 첫 Token Divergence의 Margin도 아니다.
N1의 작은 Error만으로 모든 Request의 정확도를 입증하지 않는다.
FP32 실행시간을 BF16 Baseline 또는 Serving Speedup으로 사용하지 않는다.

첫 FP32 Process는 Tensor 저장 후 JSON 직렬화에서 실패했다.
그 실패와 Tensor를 모두 보존했다.
독립 v2 실행은 정상 종료했다.
실패한 v1 Process를 PASS로 바꾸지 않았다.

## Natural32/400의 별도 Gate

READ 준비 제어의 Native, Layer, Stock Token, Hybrid는 Natural32/400을 검사했다.
각 Family는 유효 Import KV, Query/Layer Coverage 및 Lifetime 검사를 통과했다.
Normal Completion에서 Pending READ가 모두 종료됐다.
Task-quality Gate도 별도로 통과했다.
Cancellation 및 Worker Failure 복구는 이번 Gate에서 검증하지 않았다.

Native Natural400은 C1 Reference와 323/400개의 Greedy Output이 정확히 일치했다.
Task Macro Score 차이는 +0.4161 Percentage Points였다.
Context-cluster Bootstrap 95% CI는 [−0.0472, +1.1019] Percentage Points였다.
동결한 비열등성 Margin은 −2.0 Percentage Points다.
이 CI는 정확도 검사다.
Server Start 단위의 성능 CI가 아니다.

BI0의 다른 실행 Geometry는 허용한 State 및 Task-quality 계약으로 판정한다.
모든 Batch에서 Strict Token Match를 달성했다고 주장하지 않는다.
Natural Output의 길이도 실행별로 기록한다.
Dataset의 C1 Output Bucket을 모든 Runtime의 불변 Output Length로 표현하지 않는다.

## 재현 파일

- `analysis/compiled_original_reference_variability_v1.json`
- `analysis/alternate_original_compiled_path_teacher_results_v1.json`
- `references/fp32_original_native_q0_v2/result.json`
- `state/D_remote_setup_{Native,Layer,StockToken,Hybrid}_gate_v1.json`
- `state/D_remote_setup_deferral_gate_v1.json`
- Code: `qualification_gate.py`, `compiled_reference_variability.py`, `alternate_original_teachers.py`.

위 경로는 `dense_ar_longbench400_2d_20261121` Namespace 내부 경로다.
실제 Model Revision, GPU UUID, Config 및 Source SHA256은 각 Raw Folder에 저장했다.

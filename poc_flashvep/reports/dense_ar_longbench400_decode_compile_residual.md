# LongBench400: Fixed Output에서 남은 Decode Tail

Natural400의 안전성 결과를 모든 고정 Output에 확대하지 않는다.
Output16/Cap1은 같은 I3 Prompt16개와 독립 Start3회를 사용했다.
등록 수정 후 Layer와 Hybrid는 Common-policy Native를 통과했다.
하지만 두 Arm은 더 빠른 원래 NativeStrong의 ITL 기준을 통과하지 못했다.
Layer/NativeStrong ITL p99 CI 상한은1.063700이다.
Hybrid/NativeStrong ITL p99 CI 상한은1.077669이다.
동일 기준의 Stock Token 상한은1.044486으로 통과했다.
상대 기준은1.05다. 이 실패를 반올림으로 제거하지 않는다.

## 같은 실행 기록의 Host 경계

아래 값은 각 독립 Start의 통계량을 산술 평균한 값이다.
Worker와 Client의 Timestamp를 사용한다. GPU-exclusive Kernel 비용은 아니다.

| Arm | D Compile Mode | Client ITL p99(ms) | Worker Emission p99(ms) | Delivery p99(ms) | Pure Decode Step p50(ms) | Pure Decode Step p99(ms) | Host Gap p99(ms) |
|---|---|---:|---:|---:|---:|---:|---:|
| NativeStrong | 3 | 8.696131 | 8.639488 | 0.439422 | 8.110242 | 27.925320 | 0.541473 |
| NativeCommon | 3 | 8.978127 | 8.843194 | 0.463721 | 8.061749 | 36.086558 | 0.767414 |
| Layer | 0 | 9.178339 | 9.101229 | 0.453675 | 8.355442 | 102.326927 | 0.862280 |
| StockToken | 3 | 8.908710 | 8.808133 | 0.442896 | 8.104595 | 320.472683 | 0.723753 |
| Hybrid | 0 | 9.216511 | 9.095739 | 0.478717 | 8.371034 | 199.568986 | 0.666533 |
| TokenMode0Control | 0 | 9.341177 | 9.024102 | 0.772507 | 8.399580 | 324.755422 | 0.614734 |

## D-only Compile 검사

P의 Mode0와 Partial-prefill 계산은 유지했다.
D의 Full36 경로만 원래 Compiled Model로 복원하는 별도 Source를 만들었다.
원래 실행 중인54개 Bridge는 변경하지 않았다.
새 경로는 Strict N1/N8과 Natural32/400 State·Lifetime·Quality를 순서대로 검사한다.
N1의 V1 Harness는 연속 Source Window를 C0에 잘못 상속했다.
기존 Assertion이 LLM 초기화 전에 이를 차단했다. 실패 로그를 보존했다.
V2의 Serial State 검사만 해당 Window를0으로 정정했다.
V2의 Natural Continuous 검사는 원래 Window8을 유지한다.

| 검사 | Terminal | Correctness PASS | 실패 |
|---|---|---|---|
| Layer/V1 | True | False | {'name': 'DecodeCompileQ_Layer_StrictN1_v1', 'error': "AssertionError('DecodeCompileQ_Layer_StrictN1_v1')", 'status': 'FAILED'} |
| Hybrid/V1 | True | False | {'name': 'DecodeCompileQ_Hybrid_StrictN1_v1', 'error': "AssertionError('DecodeCompileQ_Hybrid_StrictN1_v1')", 'status': 'FAILED'} |
| Layer/V2 | True | False | {'name': 'DecodeCompileQ_Layer_StrictN1_v2', 'error': "AssertionError(('DecodeCompileQ_Layer_StrictN1_v2', {'full_KV_exact': False, 'last_hidden_exact': False, 'greedy_exact': True}))", 'status': 'FAILED'} |
| Hybrid/V2 | True | False | {'name': 'DecodeCompileQ_Hybrid_StrictN1_v2', 'error': "AssertionError(('DecodeCompileQ_Hybrid_StrictN1_v2', {'full_KV_exact': False, 'last_hidden_exact': False, 'greedy_exact': False}))", 'status': 'FAILED'} |
| Layer/V3 | True | False | {'name': 'DecodeCompileQ_Layer_StrictN1_v3', 'error': "AssertionError(('DecodeCompileQ_Layer_StrictN1_v3', {'full_KV_exact': False, 'last_hidden_exact': False, 'greedy_exact': True}))", 'status': 'FAILED'} |
| Hybrid/V3 | True | False | {'name': 'DecodeCompileQ_Hybrid_StrictN1_v3', 'error': "AssertionError(('DecodeCompileQ_Hybrid_StrictN1_v3', {'full_KV_exact': False, 'last_hidden_exact': False, 'greedy_exact': True}))", 'status': 'FAILED'} |

V3는 D에서 공식 Custom-op 설정을 명시적으로 `all`로 유지한다.
이 경로도 엄격한 KV·Hidden 기준을 통과하지 못했다. Layer와 Hybrid의 Greedy16은 일치했다.
Greedy 일치만으로 KV·Hidden Gate를 통과한 것으로 처리하지 않는다.
새 경로의 N8/Natural32/Natural400과 성능은 BLOCKED_CORRECTNESS다.
이는 계산 오류가 확정됐다는 뜻은 아니다. Numerical Path 차이와 구현 오류를 아직 모두 분리하지 못했다.

V2는 원래 Eager-prefix/Compiled-suffix Stage Teacher를 사용했다.
V3는 Mode0 Full36 Boundary Teacher를 사용했다. 두 Teacher의 계약은 동결한 Plan에 남긴다.
새 Teacher 선택으로 KV·Hidden 불일치를 통과 처리하지 않았다.
기존 Source071에서 실패한 성능 Gate도 유지한다.
D-only Source SHA256은 `49e494730ff48f3682d476d993122b4ba5b610b44e84621f484b4e207787bf87`이다.
Custom-op Source SHA256은 `17021999bdb7436ac256e758f59ecf5e862dfb4c679ea75e15313bb4fd970dd0`이다.

### 현재 Mode0의 Negative Control

같은 새 Source와 같은 Serial Harness에서 D Compile Flag만 모두 껐다.
이 검사는 앞선 엄격한 불일치가 기존 경로에도 있었는지 확인한다.
관측 후 새로운 Tolerance를 정해 실패한 Gate를 PASS로 바꾸지 않는다.

| Axis | 현재 Control/원래 Teacher KV 일치 | Hidden 일치 | Greedy 일치 |
|---|---|---|---|
| Layer | True | True | True |
| Hybrid | True | True | True |

| Axis / 새 경로 | Hidden Max Abs | Hidden RMS | Hidden Relative L2 | KV Sample RMS | Control 대비 Greedy16 일치 |
|---|---:|---:|---:|---:|---|
| Layer/v2 | 0.625000 | 0.026854603 | 0.007491116 | 0.009134335 | True |
| Layer/v3 | 0.500000 | 0.026263759 | 0.007326299 | 0.009410494 | True |
| Hybrid/v2 | 1.000000 | 0.043451414 | 0.012120811 | 0.017429400 | True |
| Hybrid/v3 | 0.500000 | 0.038646389 | 0.010780445 | 0.017164638 | True |

새 D 경로와 현재 Control의 전체 KV Hash, Hidden 오차와 KV Sample 오차는 JSON에 저장한다.
KV Sample은 모든 KV Tensor의 수치 오차 분포가 아니다.
Layer는 P가 계산한 Lower28 KV가 일치했다. 불일치는 D가 계산한 Layer28에서 시작했다.
Hybrid는 D가 Suffix의 모든 Layer를 계산하므로 전체 Prompt KV의 Layer0부터 차이가 있었다.
이 차이는 D 경로 개입으로 발생했다. 개별 Operator의 정확한 수치 원인은 아직 미확정이다.
Custom-op `all`만으로 원래 State를 복원한다는 가설은 이 검사에서 반증됐다.
관측 후 Tolerance를 정하거나 State를 수정하지 않았다.
이 진단 결과를 적격 성능 개선으로 사용하지 않는다.

## 판정 범위

Compiler와 Adapter의 차이는 원인 후보다. 위의 Arm 간 상관관계만으로 인과성을 확정하지 않는다.
정합성을 통과하지 못한 경로는 성능 또는 안전성 증거로 사용하지 않는다.
구조적 수치 차이와 실제 KV Ownership 오류도 구분해야 한다.
D-only 경로의 최종 성능 판정은 별도 적격 A/B가 필요하다.
C3의 GPU-exclusive Operator 비중과 SM/HBM은 이 검사에서 NOT_MEASURED다.

- [원래 수정 후 Bridge](dense_ar_longbench400_registration_controlled_bridge.md)
- [Host 경계와 새 정합성 결과](dense_ar_longbench400_2d_20261121/analysis/decode_compile_fixed_output_residual_v1.json)

## 후속 IR Provider 개입과 최종 Scope

위 표의 V1–V3 실패는 그대로 유지한다. 아래 결과는 이후의 별도 개입이다.
위 Host 표의 Pure Step은 Query의 DECODE Label 기준이다. 첫 출력 전 대기를 포함할 수 있다.
별도 최종 보고서는 첫 Worker 출력 이후의1-query Host Step을 추가 분리한다.
어느 표도 GPU-exclusive Decode Kernel 시간으로 사용하지 않는다.
설치 Source의 `vllm/platforms/cuda.py`는 Compile 여부에 따라 IR Norm Provider 우선순위를 바꾼다.
Mode0는 `vllm_c,native`, 기본 Inductor Compile은 `native`를 사용했다.
`custom_ops=all`은 이 IR 우선순위를 복원하지 않았다.
D의 RMSNorm과 Fused-add RMSNorm 우선순위만 함께 원래 값으로 바꿨다.
이 Global IR 개입에서는 Layer와 Hybrid의 Strict N1/N8 KV·Hidden·Greedy가 모두 일치했다.
이후 실제 Mixed32, Guarded Natural32와 Natural400 State·Quality Gate도 모두 통과했다.
이 결과는 적어도 검사한 엄격한 불일치에서 Norm Provider Bundle의 인과적 근거다.
두 Norm Operator의 개별 기여율이나 모든 BI0 수치 차이의 원인을 확정한 것은 아니다.
Global IR 경로의 적격 Serving 속도 향상은 측정하지 않았다.

Partial-prefill만 CUDA Norm으로 제한한 다음 Source에서는 Layer가 전체 Gate를 통과했다.
Hybrid는 Strict N1의 KV·Hidden Gate를 실패했다. Greedy16은 일치했다.
이 Source의 예정 성능21회는 첫 실행 전에 대체했다. 모든21회를 NOT_RUN으로 보존한다.
실패한 State를 사후 Tolerance 또는 Tensor Correction으로 통과시키지 않았다.

마지막 제한적 수정은 모든 Prompt Stage를 원래 Eager/CUDA Norm으로 유지한다.
Decode-only Full36은 원래 Native Compiled Model을 사용한다.
Cold Dispatch의 CPU Assertion 실패는 GPU 시작 전에 고쳤다. V1의 예정38회는 모두 미실행이다.
V2는 독립 Full36 Teacher와 같은 Numerical Stage 계약을 사용한다.
P의 계산과 Token·Timestamp 및 Frozen SLO는 바꾸지 않았다.
이 최종 Source의 정합성과 성능 판정은 별도 보고서에 기록한다.

- [Global IR 정합성 설계](dense_ar_longbench400_2d_20261121/configs/decode_IR_priority_correctness_frozen_v1.json)
- [Partial-only Source의 미실행 성능21회](dense_ar_longbench400_scoped_decode_compile_results.md)
- [최종 Eager Prefill / Native Decode 결과](dense_ar_longbench400_eager_prefill_native_decode_results.md)
- [Teacher의 실제 Scope와 Raw Hash](dense_ar_longbench400_2d_20261121/analysis/eager_teacher_effective_scope_v2.json)

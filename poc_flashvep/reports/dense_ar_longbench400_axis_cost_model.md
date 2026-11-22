# LongBench400: Axis Work and Service Cost

## 측정 범위

이 문서는 Structural Accounting, 고립 GPU 계측, Offline Replay를 구분한다.
세 근거를 실제 Continuous Serving Speedup으로 합산하지 않는다.
Main Numerical Mode는 BI0다.
Model은 Original BF16 Qwen3-8B, Revision `b968826d9c46dd6066d109eabc6255188de91218`이다.
모든 Decode는 Full36 Layers를 실행한다.

## 정확한 2D 작업량

Prompt Length를 `L`, D의 Upper Layer 수를 `m`, D의 Suffix Length를 `s`로 둔다.
Prefix Length는 `p=L-s`다.

| 항목 | P | D |
|---|---|---|
| Transformer GEMM Token-Layer Rows | `(L-s)(36-m)` | `(L-s)m+36s` |
| 유효 Causal Attention Pairs | `(36-m)p(p+1)/2` | `36L(L+1)/2 - P_pairs` |

P/D Rows의 합은 항상 `36L`이다.
Hybrid D Rows를 `Lm+36s`로 계산하면 `ms`를 중복 계산한다.
실제 식은 `Lm+(36-m)s`다.

이 Rows에는 LM Head가 포함되지 않는다.
RMSNorm, Residual, Attention, Kernel Launch와 Transfer도 같은 단위가 아니다.
같은 Transformer Rows를 전체 GEMM 비용 또는 전체 FLOPs가 같다고 해석하지 않는다.

Native 경계는 `m=s=0`이다.
현재 Native Proxy는 D에서 마지막 Prompt Query를 다시 계산한다.
이 Query는 보존식 밖의 명시적 Runtime 추가 작업이다.
실제 Query Log에서 별도로 센다.

예를 들어 `L8192/m4/s1024`의 P Rows는 229,376이다.
D Rows는 65,536이다.
이는 같은 L의 Layer8 D Rows와 정확히 같다.
동일 Rows라도 Attention과 Hidden Payload는 다르다.
실제 Token Suffix는 정수와 Alignment 제약을 받는다.
실행별 Request Manifest에서 Aggregate Error를 다시 계산한다.

## Iso-work Attention 배치

D Rows 비율을 `r`로 두면 순수 Layer의 P Attention 비율은 대략 `1-r`이다.
순수 Token에서는 대략 `(1-r)^2`다.
따라서 Token은 같은 Rows에서 P의 Attention을 더 많이 줄인다.
대신 D에 더 많은 Attention을 배치한다.

Hybrid의 Layer 비율을 `a=m/36`으로 두면 P Attention 비율은 대략 `(1-r)^2/(1-a)`다.
이 값은 두 순수 Axis 사이에 있다.
이 식은 큰 L에서 선형항을 제외한 Structural Accounting이다.
GPU 시간 비율이 아니다.

Layer의 긴 Query GEMM과 적은 Layer 호출은 D 비용을 줄일 수 있다.
Token의 짧은 P Prefix는 P Attention 비용을 줄일 수 있다.
Hybrid는 두 효과와 추가 Dependency Stage 비용을 함께 갖는다.
따라서 Interior Hybrid가 반드시 더 빠르다는 결론은 나오지 않는다.

## 적격 고립 Operator 계측

아래 값은 `ISOLATED_KERNEL_MEASUREMENT`다.
실제 Serving API와 Kernel 경로를 사용했다.
원본 Prompt 16개의 길이는 20,006–31,133 Tokens다.
평균 길이는 25,722 Tokens다.
Last4와 Aggregate Rows-matched Token을 비교했다.
Actual Batch는 C0의 1/1/1이다.
Output은 고정 8 Tokens다.

각 Axis는 독립 Clean/Profile/Clean Triplet 3개를 사용했다.
두 Role의 Observer Gate는 모두 동결한 5% 기준을 통과했다.
각 Triplet은 모든 측정 Prompt Shape를 미리 Warmup했다.
이 Warmup은 실제 Online Policy가 아니다.

| Role / 비용 | Layer (ms/request) | Token (ms/request) |
|---|---:|---:|
| P Prompt Kernel Union | 791.907 | 763.103 |
| P Attention | 287.453 | 258.186 |
| P Projection GEMM | 432.476 | 432.905 |
| D Continuation Kernel Union | 94.329 | 126.338 |
| D Attention | 32.926 | 65.107 |
| D Projection GEMM | 50.534 | 52.069 |

Projection GEMM은 QKV/O/Gate-Up/Down이다.
LM Head는 제외했다.
P Kernel Union 차이는 Layer−Token 약 +28.804ms였다.
P Attention 차이는 약 +29.267ms였다.
P Projection GEMM 차이는 약 −0.429ms였다.
이 고립 조건에서는 P Attention 배치가 큰 차이를 설명한다.

D Kernel Union 차이는 약 −32.009ms였다.
D Attention 차이는 약 −32.181ms였다.
D Projection GEMM 차이는 약 −1.535ms였다.
이 조건의 D 이득을 GEMM 효율만의 이득으로 설명할 수 없다.
Operator 구간은 Stream에서 겹칠 수 있다.
각 구간의 합을 Exclusive Total로 표시하지 않는다.

LM Head의 실제 Projection Shape는 `M1/K4096/N151936`이었다.
Prompt Chunk 수가 다르면 LM Head 호출 수도 달랐다.
Layer P의 LM Head는 약 1.605ms/request였다.
Token P에서는 약 1.497ms/request였다.
Layer D는 약 1.605ms/request였다.
Token D는 약 0.443ms/request였다.
Transformer Rows 매칭은 이 차이를 제거하지 않는다.
이 값은 `operator_repeat_observer_analysis_v3.json`의 같은 Role끼리 다시 확인했다.

짧은 Input의 Operator Capture는 별도다.
기존 API Capture와 새로운 Kernel-only Output1 Capture 모두 Observer 실패를 보존했다.
새 Output1의 세 Triplet CI도 네 Axis/Cut 조건에서 모두 5% Gate를 통과하지 못했다.
이 값은 `DIAGNOSTIC`으로만 남긴다.
긴 Input의 적격 결과를 짧은 Input의 실측 원인으로 대체하지 않는다.
SM 활용률과 HBM Traffic은 `NOT_MEASURED`다.

## Serial 비용과 Online TTFT

P와 D가 Pipeline으로 실행되면 Serial P+D 합계만으로 TTFT를 정할 수 없다.
작은 P Service 차이도 P Queue가 지속되면 누적될 수 있다.
D Continuation이 빠르더라도 이미 증가한 P Queue를 상쇄하지 못할 수 있다.
반대로 D Admission과 Residency가 병목이면 짧은 D Continuation이 유리할 수 있다.

이 설명은 M/M/1 공식을 적용한 결과가 아니다.
실제 C0 Replay는 P Queue, D Queue와 Credit4를 Event마다 갱신한다.

### 단순 Demand Rule의 반증 검사

같은 Serial Calibration을 사용해 세 가지 단순 Rule을 비교했다.
새 Winner Label로 비용 계수를 다시 학습하지 않았다.
다만 Rule 형태는 기존 Matrix 관측 후 평가했다. 이를 사전 맹검 검증이라고 부르지 않는다.

`P`는 추정 P Host Service다. `D`는 D First Service와 후속 Decode Residency의 추정 합이다.
두 값 모두 GPU-exclusive 시간이 아니다.

| Rule | 명확한 Winner 38개 중 정답 | Accuracy | 유효 59개 Cell의 평균 TTFT Regret | 최대 Regret |
|---|---:|---:|---:|---:|
| 작은 `P+D` 선택 | 16/38 | 42.11% | 2.149% | 11.530% |
| 작은 `max(P,D)` 선택 | 34/38 | 89.47% | 0.794% | 13.863% |
| 추정 Utilization 0.5에서 Sum/Bottleneck 전환 | 28/38 | 73.68% | 1.749% | 15.264% |

Regret은 작은 실측 Start-mean TTFT p95 대비 손실이다.
Sustainable Goodput 손실이 아니다. Safety Invalid 5개 Cell은 이 요약에서 제외했다.
불확실한 21개 Cell에는 명확한 Winner Label을 주지 않았다.

단일 Score `max(P_L,D_L)-max(P_T,D_T)`는 이 C0 Grid에서 Serial 합보다 Winner를 잘 설명했다.
그러나 최대 손실은 더 컸다. 완전한 Rule은 아니다.
이 Score는 여러 Calibration 입력과 알려진 Controlled Output을 압축한다.
자연 EOS는 Admission 시점에 알 수 없다. 따라서 이를 검증된 단일 Online Observable Knob이라고 주장하지 않는다.
C3의 Packing, KV 상태와 Active Decode Safety도 별도 검증이 필요하다.

명확한 반례도 남겼다.
I0/Output8/Cut2, I1/Output8/Cut4, I1/Output16/Cut4에서는 실제 Layer Winner를 Token으로 예측했다.
I0/Output8/Cut8에서는 실제 Token Winner를 Layer로 예측했다.
마지막 Cell의 Regret은 13.863%였다.
따라서 높은 전체 Accuracy만으로 이 Rule을 안전한 Online Axis Selector로 사용하지 않는다.

근거: [단순 Demand Rule 감사](dense_ar_longbench400_2d_20261121/analysis/minimal_calibrated_demand_rules_v1.json).
Credit은 P Dispatch에서 획득한다.
D의 첫 Client Output에서 반환한다.
Credit과 Actual GPU Batch는 같은 값이 아니다.
C3의 P Submit→Ready는 다른 Request의 작업도 포함한다.
이를 Exclusive Per-request GPU Service라고 표현하지 않는다.

## Offline Replay 검증

Host Service Model은 Serial Calibration 880개 관측을 사용했다.
그중 624개는 8개 Prompt × 26개 Action × 3 Starts다.
나머지 256개는 Native의 16개 Length/Output Pilot이다.
880개를 서로 다른 Prompt 880개라고 표현하지 않는다.

Tail Model은 실제 Client ITL Block을 사용했다.
긴 Output에서는 Block을 반복하는 가설을 사용했다.
Calibration의 Finish Gap은 880개 모두 0ms였다.
Model은 모든 후보에서 P/D Queue와 Credit을 다시 갱신했다.

고유 Held-out C0 Trace 432개에서 다음 오차를 얻었다.

| 지표 | Median Absolute Error | p90 Absolute Error |
|---|---:|---:|
| TTFT p95 | 3.301% | 11.371% |
| ITL p99 | 2.954% | 6.143% |
| TPOT p95 | 3.249% | 8.974% |
| Joint SLO | 0 percentage points | 0 percentage points |
| SLO-goodput / 실제 Raw Throughput 단위 오차 | 0.165% | 3.200% |

이 Model Extension은 Posthoc다.
새 Tail Model의 출력 전에 허용 오차를 동결했다.
Validation Trace는 Fitting에 사용하지 않았다.
그러나 연구자는 기존 Axis 결과를 이미 확인했다.
이를 Blind Preregistration 또는 Production Generalization이라고 표현하지 않는다.
C3 Mixed Batching과 Active-decode Safe Window는 이 Model의 검증 범위가 아니다.

## Bounded Offline Oracle

순수 Layer/Token과 Interior Hybrid를 포함한 25개 Action을 탐색했다.
Fixed Action은 Discovery Prompt에서 먼저 선택했다.
Held-out에서는 Future Arrival과 Controlled Output Length를 사용했다.
두 번의 Coordinate Sweep을 실행했다.
이 탐색은 전역 최적 상한이 아니다.
검색은 4개 Common Template Draw를 사용했다.
최종 평가는 별도 16개 Common Draw를 사용했다.

8개 Input/Output 조건의 평균 Predicted Goodput 이득은 모두 5% 미만이었다.
가장 큰 조건은 Input20K–36K/Output64였다.
세 Arrival Trace를 합친 Oracle/Discovery-fixed Goodput 비율은 약 1.0442였다.
해당 조건의 Fixed Action은 Hybrid2/f0.125였다.
한 Trace에서는 약 1.0959가 나왔다.
나머지 두 Trace에서는 약 1.0071과 1.0241이었다.
좋은 Trace 하나를 전체 Dynamic 이득으로 표시하지 않는다.

일부 조건에서는 최종 Draw를 바꾸자 검색 결과가 Fixed보다 나빴다.
그 결과도 보존했다.
Calibration 불확실성을 전파한 Confidence Interval은 없다.
Arrival Trace를 독립 GPU Server Start라고 부르지 않는다.
이 Oracle은 `OFFLINE_ORACLE`이며 실제 Online Speedup이 아니다.

## Knob에 대한 현재 결론

우선적인 설명 변수는 P/D Service Balance와 Decode Residency다.
Input Length와 Cut은 이 비용을 바꾼다.
Credit과 Admission은 Queue 증폭을 바꾼다.
Input Length 하나로 모든 승자를 설명할 수 있다는 근거는 없다.
검증된 1–2 Knob Online Selector도 아직 없다.

C0 Cost Model은 후속 Lookup 후보로 사용할 수 있다.
현재 결과만으로 C3 Admission에 이 Model을 사용하지 않는다.
실제 Decode-safe Window와 강한 Fixed 후보 대비 추가 이득이 확인될 때만 확장한다.

## 재현 근거

- `dense_ar_longbench400_2d_20261121/analysis/operator_repeat_observer_analysis_v3.json`.
- `dense_ar_longbench400_2d_20261121/analysis/short_forward_kernel_only_observer_v1.json`.
- `dense_ar_longbench400_2d_20261121/analysis/c0_tail_replay_heldout_validation_v1.json`.
- `dense_ar_longbench400_2d_20261121/analysis/c0_bounded_axis_hybrid_oracle_v1.json`.
- `geometry.py`, `c0_queue_model.py`, `c0_tail_replay.py`, `c0_oracle_search.py`.
- PNG/PDF Figure와 각 Figure의 JSON Manifest를 같은 Namespace에 저장했다.
# 추가 검증: Held-out Axis Prediction과 Regret

증거 수준은 `OFFLINE_REPLAY`다. 기존 Serial Calibration Model은 Winner Label로 재학습하지 않았다.
Model은 P/D Queue와 Credit 반환 상태를 갱신한다. 다만 전체 Controlled Arrival Trace와 고정 Output 길이를 안다.
따라서 이 검사를 Online 관측 변수 한 개의 검증이라고 부르지 않는다.

독립 Start CI와 Decode Gate가 명확한 Winner를 준 Cell은 38개였다.
Model은 그중 32개를 맞췄다. Accuracy는 32/38, 즉 84.21%였다.
Tie Cell에는 Winner Label을 부여하지 않았다. Safety가 Invalid인 5개 Cell도 제거하지 않고 Ledger에 남겼다.

유효한 59개 Cell의 평균 Latency Regret은 0.841%였다.
p95 Regret은 5.966%였다. 최대 Regret은 11.530%였다.
Regret의 기준은 두 축 중 더 낮은 실측 Start-mean TTFT p95다. Goodput Regret은 아니다.

틀린 명확한 Winner 6개는 모두 실제 Layer 우위를 Token 우위로 예측한 경우였다.
Cell은 I0/O8/Last2, I0/O16/Last2, I1/O8/Last4, I1/O8/Last12, I1/O16/Last4, I2/O64/Last2다.
특히 I0/O16/Last2의 Latency Regret은 11.530%였다.
따라서 전체 TTFT 오차가 작다는 사실만으로 Axis 경계를 정확히 예측한다고 주장하지 않는다.

이 검사는 C0 Batch1/1/1과 Source Credit4에 한정한다.
C3 Continuous Batching, Natural EOS, 새로운 Source Credit의 예측 성능은 검증하지 않았다.
최소 1–2개 Online Knob으로 일반화했다는 판정은 `NOT_ESTABLISHED`다.

근거: [Held-out Prediction Audit](dense_ar_longbench400_2d_20261121/analysis/c0_axis_winner_prediction_audit_v1.json).

## 적격 Capture의 실제 Projection Shape와 Launch

긴 Input/Last4의 적격 Capture만 사용했다. Axis별 독립 CPC Triplet3개다.
각 Profile은 실제 원문16개다. Shape Warmup에는 해당 Prompt의 사전 정보가 사용됐다.
아래 Call과 Kernel 수는 Start당 합계의 평균이다. 시간은 Request당 GPU Union의 평균이다.
API Call 수와 GPU Kernel Launch 수를 구분한다.
Shape는 실제 Serving API의 Input과 Weight에서 읽었다. 구조적 모델 차원만으로 추정하지 않았다.

| Axis / Role | Projection | 실제 M 범위 | 실제 N/K | API Calls/Start | GPU Kernels/Start | GPU Union(ms/request) | 유효 Dense TFLOP/s |
|---|---|---|---|---:|---:|---:|---:|
| Layer/prefill | GEMM_QKV | 1036–8192 | [(6144, 4096)] | 1856.000 | 1856.000 | 58.804864 | 704.505 |
| Layer/prefill | GEMM_O_PROJ | 1036–8192 | [(4096, 4096)] | 1856.000 | 1856.000 | 38.562028 | 716.220 |
| Layer/prefill | GEMM_GATE_UP | 1036–8192 | [(24576, 4096)] | 1856.000 | 1856.000 | 223.955514 | 739.938 |
| Layer/prefill | GEMM_DOWN | 1036–8192 | [(4096, 12288)] | 1856.000 | 1856.000 | 111.153578 | 745.424 |
| Layer/decode | GEMM_QKV | 1036–8192 | [(6144, 4096)] | 232.000 | 232.000 | 6.846314 | 756.397 |
| Layer/decode | GEMM_O_PROJ | 1036–8192 | [(4096, 4096)] | 232.000 | 232.000 | 4.420511 | 780.985 |
| Layer/decode | GEMM_GATE_UP | 1036–8192 | [(24576, 4096)] | 232.000 | 232.000 | 26.248081 | 789.167 |
| Layer/decode | GEMM_DOWN | 1036–8192 | [(4096, 12288)] | 232.000 | 232.000 | 13.018923 | 795.540 |
| Token/prefill | GEMM_QKV | 1398–8192 | [(6144, 4096)] | 1944.000 | 1944.000 | 58.828657 | 704.218 |
| Token/prefill | GEMM_O_PROJ | 1398–8192 | [(4096, 4096)] | 1944.000 | 1944.000 | 38.638715 | 714.796 |
| Token/prefill | GEMM_GATE_UP | 1398–8192 | [(24576, 4096)] | 1944.000 | 1944.000 | 224.059097 | 739.594 |
| Token/prefill | GEMM_DOWN | 1398–8192 | [(4096, 12288)] | 1944.000 | 1944.000 | 111.378402 | 743.918 |
| Token/decode | GEMM_QKV | 2224–3456 | [(6144, 4096)] | 576.000 | 576.000 | 6.980877 | 741.818 |
| Token/decode | GEMM_O_PROJ | 2224–3456 | [(4096, 4096)] | 576.000 | 576.000 | 4.747575 | 727.188 |
| Token/decode | GEMM_GATE_UP | 2224–3456 | [(24576, 4096)] | 576.000 | 576.000 | 26.974611 | 767.911 |
| Token/decode | GEMM_DOWN | 2224–3456 | [(4096, 12288)] | 576.000 | 576.000 | 13.366102 | 774.877 |

TFLOP/s는 구조적 Dense Linear FLOPs `2MNK`를 관측 Kernel Union으로 나눈 유효 Rate다.
SM 활용률, Tensor Core 활용률 또는 HBM Traffic 실측값이 아니다.
Attention, Norm, Residual, LM Head와 Transport 비용은 포함하지 않는다.
Category가 다른 GPU 구간은 겹칠 수 있다. 표의 시간을 Exclusive Total로 합산하지 않는다.
모든 Projection의 실제 M 합과 Logical Token-Layer Rows 일치 여부도 JSON에 보존했다.
기본 B8192 때문에 긴 Layer Query도 여러 Model Call로 나뉜다.
전체 Input Length를 각 GEMM의 M 값으로 대입하지 않는다.
이 결과는 고립된 C0 경로다. 강한 Compiled Stock Token의 Natural C3 Operator 비용을 대신하지 않는다.

- [실제 모든 Shape, Kernel 이름과 Raw Hash](dense_ar_longbench400_2d_20261121/analysis/qualified_projection_shape_and_launch_audit_v1.json)

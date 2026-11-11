# Loop Split과 Dense AR Layer Split의 차별성 분석

## 1. 결론

현재 Whole-prompt Loop Split에서 **일반적인 Logical Layer Split이 재현할 수 없는 성능 이점은 확인하지 못했다.**
사용자가 제기한 반론은 타당하다.
넓은 Token 병렬성, GEMM 효율, Attention 계산량 차이는 Loop에만 있는 특성이 아니다.

다만 **Recurrent Weight Sharing에 의존하는 실제 구조적 차이**는 있다.
서로 다른 Depth의 작업이 같은 Core Weight를 사용한다.
따라서 이 작업들을 하나의 Recurrent Core Batch로 묶을 수 있다.
Untied Dense AR의 서로 다른 Layer는 일반적으로 서로 다른 Weight를 사용한다.

그러나 이 차이에는 세 가지 제한이 있다.

1. 이 기능은 이미 CDB의 핵심 기법이다.
2. 현재 Loop Migration이 이 기능의 추가 Serving 이득을 입증하지 못했다.
3. Looped 모델의 Token Split도 같은 기능을 사용할 수 있다.

따라서 판정은 `NO_CONFIRMED_LOOP_SPLIT_EXCLUSIVE_MECHANISM`이다.
이는 Loop Split이 모든 조건에서 무가치하다는 뜻은 아니다.
**특정 조건의 유효한 Depth Offloading**과 **Loop-specific한 새 기여**를 구분해야 한다.

## 2. 이번 분석의 범위

요청한 순서대로 기존 Discovery 캠페인을 먼저 종료했다.
그 캠페인의 최종 원격 SHA는 `0efa68913f038bc2b9cab501d5c1b6347d130bbf`다.
[Discovery 최종 보고서](https://github.com/jyssys/MLLM-EP/blob/0efa68913f038bc2b9cab501d5c1b6347d130bbf/poc_flashvep/reports/ouro_huginn_prefill_heavy_decode_safe_loop_offloading_final_summary.md)

이번 후속 분석은 CPU와 코드·문헌 확인만 사용했다.
추가 GPU Benchmark는 실행하지 않았다.
GPU 0~7의 승인된 Burner는 유지했다.

검토한 근거는 다음과 같다.

- 실제 Qualified RLT의 Ouro/Huginn Forward와 Scheduler.
- 기존 Loop/Token 128·256·512 비교의 동결된 결과.
- 기존 D-local / Same-depth / Cross-depth Microbenchmark와 실패 판정.
- 로컬 `ref_loopserve`의 Loop 실행 및 Non-loop 배치 규약.
- CDB, ExactLoop, FluidPD의 원문.
- 동결된 Checkpoint의 Safetensors Header와 기존 GPU Parameter Inventory.

[기계 판독용 분석 근거](ouro_huginn_loop_specific_vs_dense_ar_layer_split_analysis_20261112/audit.json)에 Source Hash와 Checkpoint Revision을 기록했다.
원본 Weight, Runtime, 이전 보고서는 수정하지 않았다.

## 3. 먼저 세 가지 주장을 구분해야 한다

| 주장 | 현재 근거 | 판정 |
| --- | --- | --- |
| 일부 조건에서 Loop Cut이 Token Suffix보다 효율적인가? | 실제 Serial GPU 비교가 있다 | 조건부로 확인 |
| 그 이유가 Weight Sharing을 요구하는가? | 기존 Width·Attention 이득은 요구하지 않는다 | 현재 주요 설명은 Generic Depth Split |
| 새 Loop-aware PD가 최적 Layer/Token 방식보다 Serving에서 유리한가? | 동일 조건의 강한 Online 비교가 부족하다 | 미입증 |

첫 번째 주장이 참이어도 세 번째 주장이 자동으로 참이 되지 않는다.
반대로 Generic 기법이라는 이유만으로 실제 시스템 이득을 무효화할 필요도 없다.
다만 논문에서 주장할 기여의 범위가 달라진다.

## 4. 현재 Loop Cut은 어떻게 Layer Cut으로 표현되는가?

Ouro는 24개 물리적 Transformer Layer를 네 번 사용한다.
`recurrent_prepared()`는 매 호출에서 같은 `self.model.layers`를 실행한다.
각 Loop 끝의 Norm도 다음 Loop 입력에 포함된다.

Huginn은 Prelude 2개, Recurrent Block 4개, Coda 2개를 사용한다.
Recurrent Block과 Adapter를 32번 반복한다.
각 반복은 바뀐 Recurrent State와 고정된 Prelude Injection을 받는다.
`recurrent_prepared()`는 같은 `transformer.adapter`와 `transformer.core_block`을 호출한다.

소스 위치는 다음과 같다.

| 소스 | 관련 위치 | 확인 내용 |
| --- | --- | --- |
| RLT `models/ouro.py` | 224–295행 | 동일 Layer 객체 재사용, Loop 내부 Norm |
| RLT `models/huginn.py` | 201–313행 | Prelude/Core/Coda, Adapter 재사용, State와 Injection 전달 |
| RLT `core/scheduling_policy.py` | 162행부터 | Prefill/Decode Recurrent Phase의 Mixed Batch 지원 |
| RLT `core/scheduler.py` | 701행부터 | 요청별 Phase·Position·Depth와 공통 실행 예산 |
| `ref_loopserve/.../loop_spec.py` | LoopBody 및 NonLoopOp | Loop 본체와 Entry/Exit 연산의 분리 |

정확한 소스 파일 Hash는 분석 근거에 있다.
외부 Runtime의 기존 Diff는 그대로 보존했다.

고정 Depth 계산을 다음과 같이 쓸 수 있다.

```text
S(r+1) = F(theta, S(r), Injection, KV_plane(r))
```

이를 R번 펼치면 일반적인 유한 계산 그래프가 된다.
Loop c 뒤의 Cut은 펼친 그래프에서 같은 지점의 Cut이다.
State, Injection, KV, 실행 순서와 Ownership을 유지하면 같은 계산을 이어갈 수 있다.

| Loop Cut | 펼친 그래프의 대응 경계 |
| --- | --- |
| Ouro R1/R2/R3 뒤 | Transformer Layer 적용 24/48/72회 뒤와 해당 Loop Norm 뒤 |
| Huginn R24 뒤 | Prelude 2개 뒤, Core Layer 적용 96회와 Adapter 24회 뒤 |
| Huginn R28 뒤 | Prelude 2개 뒤, Core Layer 적용 112회와 Adapter 28회 뒤 |
| Huginn R31 뒤 | Prelude 2개 뒤, Core Layer 적용 124회와 Adapter 31회 뒤 |

이 경우는 `EQUIVALENT_MECHANISM`이다.
이는 Untied AR 모델과 Weight 값이나 실제 실행시간까지 같다는 주장이 아니다.
**동일한 Looped 계산 그래프를 Layer 관점으로 표현해도 Cut이 사라지지 않는다**는 뜻이다.

Huginn의 대응 그래프는 Injection 같은 Side State도 보존해야 한다.
단순한 Hidden Tensor 하나만 전달하면 충분하다고 가정하지 않는다.
일반적인 Graph/Layer Scheduler도 이 Side State를 명시적으로 전달할 수 있다.

일반 Pipeline 실행도 Model Stage와 Microbatch를 분리한다.
그 사실이 이번 Adaptive PD Protocol의 완성이나 동일 성능을 증명하지는 않는다.
[PyTorch Pipeline 문서](https://docs.pytorch.org/docs/2.15/distributed.pipelining.html)

## 5. 기존 Loop/Token 이득의 어디까지가 Generic한가?

### 5.1 Serial 이득은 실제로 존재한다

다음 값은 기존 GPU 측정값이다.
각 Cell은 한 Process Start의 Serial 요청 8개다.
독립 Start 8회나 Online Serving 비교가 아니다.

| 모델 | Loop 작업 | Token 작업 | Loop D Prefill p50 | Token D Prefill p50 |
| --- | --- | --- | --- | --- |
| Ouro | 4096 tokens × 1 Loop | 1024 tokens × 4 Loops | 40.034 ms | 48.064 ms |
| Huginn | 2048 tokens × 2 Loops | 128 tokens × 32 Loops | 41.944 ms | 81.367 ms |
| Huginn | 2048 tokens × 4 Loops | 256 tokens × 32 Loops | 74.439 ms | 96.770 ms |
| Huginn | 2048 tokens × 8 Loops | 512 tokens × 32 Loops | 137.780 ms | 151.044 ms |

양쪽 모두 가능한 Remaining Loop를 한 번의 D Submission으로 묶었다.
Token 쪽에 불필요한 Loop별 Host Submission을 강제한 비교가 아니다.
다만 이 시간에는 Transfer와 별도 First-output 생성이 포함되지 않는다.
실제 Kernel Union, Service TTFT, Decode 간섭도 아니다.
[기존 짧은 비교 보고서](ouro_huginn_loop_token_suffix_128_256_512_quick_comparison.md)

기존 GEMM 탐색은 유한한 후보 탐색이다.
전역 최적 Kernel의 하한을 증명하지 않는다.
Huginn Token512의 Tuned Dense 합산 비율은 세 Start에서 0.984, 1.012, 1.054였다.
즉, 큰 Token Suffix에서는 Dense 차이가 거의 없어질 수 있다.

Huginn Token128의 P1920 Prefix는 별도 Tuned Entry가 없었다.
그 Cell의 음수 P Saving을 본질적인 Token 방식의 한계로 해석하지 않는다.

### 5.2 Attention 차이는 GEMM 교체만으로 사라지지 않는다

Prompt 길이를 N, 전체 Loop를 R, D Loop를 m, Token Suffix를 S라고 하자.
Recurrent Rows를 맞추면 `mN = RS`다.
같은 Core Attention Layer 하나의 Causal Query-Key Pair 수는 다음과 같다.

```text
Loop cut:    m × N(N+1)/2
Token cut:   R × [S(N-S) + S(S+1)/2]
```

Token Suffix는 매 Depth에서 이미 계산한 Prefix에도 Attend한다.
따라서 Dense Projection FLOPs가 같아도 Attention Work는 같지 않다.

| 모델 / Token Suffix | Loop Causal Pairs | Token Causal Pairs |
| --- | --- | --- |
| Ouro / 1024 | 8390656 | 14682112 |
| Huginn / 64 | 2098176 | 4129792 |
| Huginn / 128 | 4196352 | 8128512 |
| Huginn / 256 | 8392704 | 15732736 |
| Huginn / 512 | 16785408 | 29368320 |

이 표는 `STRUCTURAL_ACCOUNTING`이다.
실측 FLOPs, HBM Bytes, Kernel Time이 아니다.
Prelude, Coda, Padding, Transfer 비용도 제외한다.

좋은 GEMM은 Projection 비용을 줄일 수 있다.
그러나 필요한 Causal Dependency나 Query-Key Pair 자체를 없애지는 않는다.
그럼에도 이 장점은 Dense AR의 **Whole-prompt Layer Cut**에서도 나타난다.
따라서 Loop-specific 이유로 사용할 수 없다.

### 5.3 긴 순차 Depth와 넓은 Token 병렬성도 Generic하다

64 Tokens를 32번 반복하면 비선형 State Dependency가 32단계 남는다.
2048 Tokens의 마지막 한 Loop는 Token 축에서 넓게 실행할 수 있다.
Kernel Tuning만으로 앞 단계의 Hidden 결과를 미리 알 수는 없다.

그러나 Dense AR의 서로 다른 Layer 사이에도 같은 순차 Dependency가 있다.
Token 축 대신 Layer 축으로 나누면 같은 종류의 병렬성 이득을 얻을 수 있다.
이 역시 `EQUIVALENT_MECHANISM`이다.

## 6. 실제로 Weight Sharing이 필요한 차이

### 6.1 가장 강한 후보: Depth-independent Core Compatibility

서로 다른 Depth의 두 준비된 State를 X와 Y라고 하자.
동일한 Projection Weight W를 사용하면 다음 실수 연산 항등식이 성립한다.

```text
concat(X, Y) W = concat(XW, YW)
```

Untied Dense AR에서는 일반적으로 `W_layer_a != W_layer_b`다.
이 경우 두 작업을 동일한 W의 한 Dense GEMM으로 바꿀 수 없다.
Grouped GEMM은 지원할 수 있지만 서로 다른 Weight가 사라지지는 않는다.
Block-diagonal 표현도 공짜 Weight 공유를 만들지는 않는다.

따라서 **최적 GEMM을 사용해도 남는 구조적 차이**는 있다.
차이는 Depth라는 이름이 아니라 동일한 Weight Identity다.

그 효과는 개별 GEMM의 속도가 아니라 준비된 작업의 호환 범위에서 발생할 수 있다.
서로 다른 Depth의 작은 작업이 함께 Batch를 만들 수 있다.
Untied Layer에서는 같은 Layer를 기다리거나 서로 다른 Weight 연산을 별도로 처리해야 할 수 있다.

다만 이 항등식은 BF16 Bitwise Equality를 보장하지 않는다.
Batch Shape가 달라지면 Reduction과 Kernel 선택도 달라질 수 있다.
Qualified Numerical Fix와 실제 State Qualification이 여전히 필요하다.

CDB는 이미 서로 다른 Loop Depth의 작업을 Shared Core로 묶는다.
따라서 이 기능 자체를 새 발명으로 주장할 수 없다.
[CDB 원문, Method 및 Related Work](https://arxiv.org/html/2608.09444v2)

또한 Looped 모델의 Token Suffix도 같은 Weight Sharing을 가진다.
이 구조만으로 Loop Split이 최적 Token Split보다 좋다고 결론 낼 수 없다.
Weight Alias를 이해하는 Generic Logical Layer Scheduler도 같은 기능을 구현할 수 있다.
이것은 **Looped Architecture의 차이**이지, 이름만 다른 Scheduler의 독점적 기능이 아니다.

### 6.2 작은 Parameter Residency는 실제 구조적 차이지만 불공정한 비교를 주의해야 한다

이번 CPU 분석은 로컬 Checkpoint Header의 Shape를 읽었다.
모델 Tensor Payload는 읽거나 GPU에 적재하지 않았다.
계산한 Unique Parameter 수를 이전 실제 Runtime Inventory와 대조했다.
두 모델 모두 정확히 일치했다.

| 모델 | Runtime Unique Parameters | 반복 Core Parameters | 실제 BF16 Weight 크기 | Core를 물리적으로 R번 복제한 가상 크기 |
| --- | --- | --- | --- | --- |
| Ouro R4 | 1434652673 | 1233324032 | 2.6722488421946764 GiB | 9.563983919098973 GiB |
| Huginn R32 | 3564976800 | 1637349120 | 6.6402867436409 GiB | 101.18409901857376 GiB |

Huginn의 반복 Core에는 Adapter도 포함했다.
Tied LM Head는 중복 집계하지 않았다.
Frequency Buffer는 Parameter에서 제외했다.
Checkpoint의 저장 Dtype와 Runtime BF16 크기도 구분했다.

마지막 열은 `STRUCTURAL_ACCOUNTING`이다.
같은 품질의 실제 Dense AR 모델이 아니다.
동일 Weight를 복제했다면 원래처럼 Alias할 수 있으므로, 복제 크기를 필수 비용으로 주장할 수 없다.
서로 다른 Weight를 학습한 Untied 모델은 별개의 모델이다.
Parameter 수, 유효 Depth, FLOPs와 품질을 모두 같다고 가정할 수도 없다.

현재 1P1D는 양쪽 GPU가 전체 물리적 모델을 갖는다.
Dense AR도 전체 모델이 각 GPU에 들어가면 Layer Cut마다 Weight를 다시 가져올 필요가 없다.
따라서 작은 Residency만으로 이번 Loop Migration의 새 이득을 증명하지 못한다.
Looped 모델의 Token Split도 같은 Residency 이점을 가진다.

### 6.3 Weight Sharing이 제공하지 않는 것

- 서로 다른 Depth의 KV가 같다는 보장은 없다.
- Hidden State가 같거나, Loop를 생략해도 된다는 보장은 없다.
- 이미 시작한 Full-prompt Kernel의 선점 기능은 없다.
- 추가 Handoff, ACK, Admission 비용이 없어지지 않는다.
- Shared Weight 전체가 L2 Cache에 남는다는 보장도 없다.

각 Depth는 다른 Hidden을 받는다.
따라서 같은 W라도 `K_r = H_r W_K`는 일반적으로 서로 다르다.
이번 Exactness Contract에서는 Depth별 유효 KV를 유지해야 한다.

Huginn의 고정 Injection은 반복 사용되지만 재사용 최적화는 별도 문제다.
Adapter를 분해하거나 미리 계산하면 BF16 Reduction 순서가 바뀔 수 있다.
Ouro 공통 장점도 아니며, 현 Loop Split의 검증된 이득도 아니다.

## 7. 기존 Cross-depth 실험은 무엇을 보여줬는가?

이 방향은 아직 한 번도 실험하지 않은 아이디어가 아니다.
기존 PoC가 이미 Separate, Same-depth, Cross-depth를 비교했다.
강한 Same-depth 대조군은 필요한 정렬 Step의 비용도 포함했다.

| 모델 | Separate Core | Same-depth Batch | Cross-depth Batch |
| --- | --- | --- | --- |
| Ouro | 11.705 ms | 6.599 ms | 6.594 ms |
| Huginn | 4.392 ms | 2.348 ms | 2.348 ms |

Batch 자체의 절감은 컸다.
그러나 Same-depth도 같은 수준의 절감을 얻었다.
Full-core Same/Cross의 Attention Metadata는 다를 수 있다.
동일 Graph 비교는 별도의 Dense Label Control이다.
동일 Tensor·Weight Pointer·Graph를 사용하는 Dense Label Control에서는 Depth Label이 연산 입력이 아니었다.
실제 Weight Read Bytes는 `NOT_MEASURED`다.

강한 C248 Same-depth 정렬 대조군과의 비교도 중요하다.

| 모델 / GPU | Same-align Local First Token | Cross-depth Local First Token | Same-align Cohort | Cross-depth Cohort |
| --- | --- | --- | --- | --- |
| Ouro / 0 | 625.108 ms | 628.350 ms | 1626.912 ms | 1623.394 ms |
| Ouro / 1 | 627.327 ms | 631.179 ms | 1636.086 ms | 1632.973 ms |
| Huginn / 4 | 1379.397 ms | 1383.800 ms | 3323.412 ms | 3368.357 ms |
| Huginn / 5 | 1375.161 ms | 1379.972 ms | 3313.577 ms | 3358.123 ms |

이 표는 Background Decode 2개와 Local Prompt 1개의 Closed-loop Control이다.
Online PD Goodput이나 새 Arrival Seed의 N128 Serving 결과가 아니다.
Local Micro Request는 Output2다.
Background Output은 Ouro32와 Huginn16이다.

기존 C32 3-start Readiness Control에는 작은 차이도 있었다.
다음 CI는 당시의 탐색적 Paired-start Bootstrap이다.

| 모델 | Cross/Same Cohort Median, 95% CI | Cross/Same Local First Token Median, 95% CI |
| --- | --- | --- |
| Ouro | 0.9997080176828023 [0.9980102017992614, 0.9998925467579365] | 0.9951774963165806 [0.9948953373867597, 0.9957052495577717] |
| Huginn | 0.9946584882920199 [0.994598082531539, 0.9948079297780755] | 0.9898247607321896 [0.9894935022013346, 0.9898336188146467] |

이 수치는 독립 Start 3회의 제한된 Control에만 해당한다.
새로운 Dense 메커니즘이나 Online Capacity의 CI가 아니다.
당시 Phase C Gate는 두 모델 모두 실패했다.
Cross-depth Online N128 확장은 실행하지 않았다.

[기존 최종 보고서](ouro_huginn_loop_aware_d_local_prefill_cross_phase_batching_final_summary.md), [기존 Gate 및 CI](ouro_huginn_loop_aware_d_local_prefill_cross_phase_batching/phase_c_gate.json)

따라서 현재 근거로는 “Cross-depth를 추가하면 해결된다”고 권고할 수 없다.
새 이득을 찾으려면 기존 강한 대조군을 이기는 다른 Ready-work 분포가 필요하다.

## 8. 남아 있는 가장 설득력 있는 가설

가설은 **Weight-sharing-aware Work Supply**다.
단순히 남은 Loop 수를 최대한 많이 이전하는 방식과 다르다.

현재 Decode와 같은 Core 연산으로 합칠 수 있는 Prefill 작업을 공급한다.
Cut 선택은 D의 Deadline과 Compatible Ready Rows를 함께 고려한다.
목표는 Weight를 새로 읽는 별도 실행보다 낮은 추가 비용으로 유효 Prefill을 진행하는 것이다.

이 가설은 다음 조건에서만 유망할 수 있다.

1. Ready Decode Batch가 작고 실제로 Weight/Launch 비용을 충분히 상각하지 못한다.
2. Depth가 달라서 Same-depth Control이 실제 대기 또는 정렬 비용을 낸다.
3. Cross-depth Batch의 Attention/KV/패킹 비용이 작다.
4. 늘어난 Quantum이 Native-relative Decode Tail 기준을 통과한다.
5. Transfer와 Checkpoint를 포함한 P Critical-path 순절감이 남는다.

다만 이것은 `UNTESTED_HYPOTHESIS`다.
이번 분석은 새 Scheduler를 구현하거나 Speedup을 예측하지 않았다.
기존 CDB와 구분되는 새로운 PD 결합 효과도 아직 없다.
Token Split이 같은 Work Supply를 제공하면 Loop Split의 독자적 우위는 다시 사라진다.

최소한의 후속 검증은 다음처럼 설계해야 한다.

| 대조군 | 고정할 것 | 구분할 원인 |
| --- | --- | --- |
| Same-depth + 정렬 비용 포함 | 동일 Ready State와 Deadline | 단순 정렬 대기 제거 효과 |
| Cross-depth Shared Core | 동일 Placement와 추가 Work | Weight Identity 기반 호환성 |
| 강한 Token Suffix + 같은 Cross-depth 기능 | 동일 Guard·Graph·KV·Arrival | Loop Cut의 추가 Work Supply 가치 |
| Weight Alias-aware Logical Layer Scheduler | 동일 Looped 그래프와 모든 상태 | Loop라는 이름 이상의 새 Scheduling 기여 |
| Untied AR Grouped GEMM | 실제 다른 Weight, 명시적 모델 차이 | Architecture 차이와 Kernel 선택 차이 |

마지막 비교는 동일 모델 품질을 자동으로 맞출 수 없다.
따라서 Architecture 진단과 Serving 승리 주장을 분리해야 한다.
새 대규모 캠페인을 바로 시작할 근거는 아직 부족하다.

## 9. 선행연구와 Exactness 경계

CDB는 Shared Core의 Cross-depth Batch와 Loop별 Scheduling을 이미 다룬다.
그 논문의 KV 공유 실험은 별도의 Layout/품질 선택이다.
우리의 R32 Exact KV Contract에 그대로 적용할 수 없다.
[CDB, KV Layout 및 평가](https://arxiv.org/html/2608.09444v2)

ExactLoop는 Dead KV Slot, Kernel, Graph, Handoff 비용을 Exactness 등급으로 구분한다.
고정 Batch의 값 보존과 Batch Composition 변경은 같은 보장이 아니다.
그 논문의 Packed Prefill이나 Noise 변경을 엄격한 Exactness 조건에 자동으로 도입하면 안 된다.
[ExactLoop 원문](https://arxiv.org/html/2611.05150v1)

FluidToken의 Decode Slack 활용과 Pressure-aware 부분 Prefill 이전은 Generic P/D Load Balancing이다.
우리의 Token 구현은 공식 FluidPD 실행 결과가 아니라 FluidToken-style 재현이다.
이전 단위를 Loop로 바꾸는 것만으로 그 원리가 새로워지지는 않는다.
[FluidPD 원문](https://arxiv.org/html/2610.06917v1)

## 10. 최종 연구 판단

**확인한 것:** Loop Cut의 넓은 Token 병렬성과 Attention Work 차이는 실제 유리할 수 있다.
하지만 이 설명은 Dense AR Layer Cut에도 적용된다.

**구조적으로 다른 것:** Weight Sharing은 서로 다른 Depth를 같은 Core 연산의 Batch로 만들 수 있게 한다.
Kernel Tuning은 서로 다른 Weight를 동일 Weight로 만들 수 없다.

**아직 없는 것:** 이 구조가 현재 Loop Split에서 강한 Token/Layer 대조군을 이기는 추가 Online 이득을 낸다는 증거다.
이전 Cross-depth 대조 실험도 그 주장을 지지하지 않았다.

따라서 지금 논문 주장을 “Loop Split이 Layer Split과 본질적으로 다르다”로 잡는 것은 권하지 않는다.
더 정확한 후보는 “Weight Sharing을 이용한 Deadline-aware Prefill/Decode Work Co-scheduling”이다.
그 경우에도 CDB와 강한 Token 방식 대비 추가 이득을 새로 입증해야 한다.

이번 결론은 새로운 GPU NO-GO Benchmark가 아니다.
현재 코드, 기존 실측, 구조적 계산과 선행연구를 합친 Novelty Audit이다.
Dense AR의 실제 Online Layer Offloading 결과는 `NOT_RUN`이다.
새 Loop-specific TTFT/Goodput 이득은 `NOT_MEASURED`다.

## 11. 재현 및 보존

분석 코드: `poc_flashvep/loop_specific_layer_audit/audit.py`.
실행 명령: `python -m poc_flashvep.loop_specific_layer_audit.audit`.
동일 Evidence Namespace는 덮어쓰지 않는다.
재실행에는 새 Namespace가 필요하다.

[Audit JSON](ouro_huginn_loop_specific_vs_dense_ar_layer_split_analysis_20261112/audit.json)에는 Header Hash, Runtime Source Hash와 CPU 검사를 기록했다.
[Configuration](ouro_huginn_loop_specific_vs_dense_ar_layer_split_analysis_20261112/configuration.json)은 GPU 미사용 상태와 모델 Revision을 명시한다.
CPU 검사는 Integer 연산의 Cut/Concat 관계만 확인한다.
이는 원본 모델의 BF16 정합성이나 새 Serving 구현 검증을 대신하지 않는다.

최종 게시에는 별도 Branch를 사용한다.
기존 Staged Index, 외부 Runtime Diff와 Burner 상태를 다시 확인한다.

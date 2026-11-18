# Dense AR Serving Projection API 계측 감사

## 판정

`STRUCTURAL_ACCOUNTING`: 기존 고립 GEMM과 현재 Serving은 같은 호출 경로가 아니다.
기존 측정은 `torch.nn.functional.linear`를 호출했다.
Serving의 Unquantized Linear는 `VLLM_BATCH_INVARIANT=True`일 때 `linear_batch_invariant`를 직접 호출한다.
설치된 H100 분기의 `enable_batch_invariant_mode`는 `aten::linear`를 교체하지 않는다.
따라서 H100에서 두 호출을 같은 Projection API로 취급할 수 없다.

기존 고립 측정값은 삭제하거나 변경하지 않는다.
그 값은 해당 API의 실제 측정이다.
다만 현재 Serving Projection의 직접 비용으로 대체해서 사용하지 않는다.
이 감사는 Model Weight, Precision 또는 Serving Forward를 변경하지 않았다.
기존 Client-visible Latency와 실제 Continuation Forward 측정도 변경하지 않았다.

## 재현과 검사

CPU 재현은 기존 고립 소스가 Serving API를 호출한다는 조건에서 실패했다.
실패 메시지는 `RED: legacy isolated projection does not call the explicit Serving GEMM API`이다.
새 대조는 원래 고립 함수에서 Projection 호출 한 곳만 교체했다.
회귀 검사는 나머지 함수 소스가 같다는 조건도 검사한다.
`test_projection_api`는 1개 검사에서 통과했다.

설치된 `UnquantizedLinearMethod.apply`와 Batch-invariant 소스의 SHA256을 저장했다.
Serving Worker의 실제 `VLLM_BATCH_INVARIANT=True` 조건은 기존 Ready 기록으로 검사한다.
새 고립 Trace의 Kernel 이름은 `matmul_kernel_persistent`이다.
Clean Online Serving의 모든 Projection Kernel을 직접 Capture했다는 뜻은 아니다.

## 새 고립 측정

실행 이름은 `isolated_serving_projection_API_20261118_v1`이다.
GPU0/1에서 19개 M Shape와 4개 Projection을 각각 측정했다.
총 측정 행은 152개이다.
M은 1/2/3/4/432/448/880/896/1776/1792/2656/2672/5328/5344/6208/6224/7104/7120/8000이다.
원래 BF16 Layer0 Weight와 보존된 Activation 행을 사용했다.
각 Shape의 준비 실행은 5회이고, 기본 시간 Sample은 30개이다.
측정값은 CUDA Event 경과시간이다. Exclusive Kernel Duration의 합이 아니다.

M2672와 M8000의 별도 고립 A/B/A Profiler 검사는 총 16개이다.
16개 모두 해당 고립 연산의 5% Gate를 통과했다.
이 Gate는 Online Serving의 계측 간섭 Gate가 아니다.
Online Kernel Busy/Idle이나 SM/HBM Counter는 이 측정으로 확인하지 않았다.

M432/448은 이전 소규모 Last2 측정의 API 경로를 점검하기 위한 Shape이다.
Last2 Online Sweep을 재개한 것이 아니다.

## 동일 Rows에 적용한 비용 모델

아래 표는 D GPU인 GPU1의 고립 측정에 기반한다.
원래 Layer0의 4개 Projection 중앙값을 Shape와 Layer 수에 따라 확장했다.
Token은 N72의 인접 Suffix 조합으로 GEMM Rows를 정확하게 맞췄다.

**아래 시간은 파생 비용 모델이다. Full Forward, TTFT 또는 Online GEMM 실측 시간이 아니다.**

| Pair | Layer Projection 모델(ms) | Token Projection 모델(ms) | Token / Layer 모델 비율 |
|---|---:|---:|---:|
| Last4 / Token≈889 | 22.8293 | 25.2129 | 1.1044 |
| Last8 / Token≈1778 | 45.6586 | 49.0766 | 1.0749 |
| Last12 / Token≈2667 | 68.4879 | 71.8138 | 1.0486 |

이 모델은 Cut이 커질수록 Layer의 Projection 비용 이득이 줄어드는 경향을 보인다.
이 모델 자체는 큰 Cut에서 Token 우위가 발생한다고 예측하지 않는다.
실제 Layer Weight, Cache Residency와 Online Launch 시점은 고립 측정과 다를 수 있다.
독립 Start 반복 CI도 없다.

기존 F.linear 측정과 새 측정의 실행 시점은 다르다.
두 측정의 시간 차이 전체를 API 변경의 인과 효과로 해석하지 않는다.

## Online 원인 분석의 제한

GEMM 모델과 전체 Forward의 차이를 Attention 시간으로 표시하지 않는다.
고립 중앙값의 합과 Online Forward 경과시간은 같은 계측 경계가 아니다.
Norm, Activation, Launch, Attention, Transfer와 Queue를 따로 검사해야 한다.

현재 Batch1 Serving은 별도 Protocol로 진행한다.
실제 배치와 Client-visible TTFT를 검사한 뒤 Online 이득을 판단한다.
작은 Cut의 GEMM 모델 이득이 SLO-goodput 이득으로 이어진다고 가정하지 않는다.
큰 Cut의 역전이 없으면 그 결과도 그대로 남긴다.

## Evidence 위치

- 원시 데이터: `dense_ar_matched_axis_crossover_20261117/raw/isolated_serving_projection_API_20261118_v1/`.
- API 감사: 위 폴더의 `projection_API_audit.json`.
- Shape별 시간과 Trace Gate: 위 폴더의 `operator_geometry.json`.
- 파생 모델: `dense_ar_matched_axis_crossover_20261117/analysis/actual_projection_API_geometry_model.json`.
- 배치 일치 계획: [배치 일치 Protocol](dense_ar_batch_matched_axis_tradeoff_protocol.md).

원시 측정은 `MEASURED`인 고립 연산이다.
위 표는 `STRUCTURAL_ACCOUNTING`이다.
Exclusive Online GEMM·Attention 시간과 SM/HBM Counter는 `NOT_MEASURED`이다.

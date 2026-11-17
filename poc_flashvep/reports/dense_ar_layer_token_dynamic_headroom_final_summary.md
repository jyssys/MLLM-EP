# Dense AR TP2·Layer/Token·Dynamic 심층 PoC 최종 요약

## 최종 판정

TP2의 최초 수치 차이는 식별했다.
그러나 TP1 Strict Parity를 회복하지 못했다.
따라서 TP2는 Primary 성능 기준선에서 제외했다.

실제 2GPU Token-axis Continuation은 구현과 State 정합성을 통과했다.
같은 이전 작업량의 Last12 Control에서는 Layer의 조건부 이점을 확인했다.
이 결과를 모든 Token Split이나 모든 Workload에 대한 우위로 해석하지 않는다.
새 문서의 Best-vs-Best 결과와 Dynamic의 독립 기여는 아래 Table에서 별도로 판단한다.

이 캠페인은 Sustainable Capacity 또는 논문 Novelty를 입증하지 않았다.
N128/N144 완료율은 Finite-cohort Drain Rate다.
물리적인 Kernel Busy/Idle, SM/HBM Counter는 `NOT_MEASURED`다.

최종 판정은 부분 성공이다.

| 연구 목표 | 판정 | 범위 |
|---|---|---|
| TP2 수치 원인 | NUMERICAL_NON_EQUIVALENCE_IDENTIFIED | 수집한 최초 경계와 같은 Operand Replay |
| TP1 Strict Parity의 TP2 Baseline | BLOCKED_CORRECTNESS | 전체 Within-TP2 State도 별도 미측정 |
| 실제 Token-axis Continuation | CORRECTNESS_ONLY: PASS | N8·N32와 선정 후보 N128 Prompt State |
| Layer-axis Serving 이점 | MEASURED: 조건부 확인 | Output16의 동결 후보와 같은 작업량 Control |
| Strong Static을 넘는 Dynamic Primary 기여 | NO_GO_ON_TESTED_GRID | SLO Rate CI와 Native-relative 보호의 동시 우위 미확보 |
| Sustainable Capacity·전역 Novelty | NOT_MEASURED / NOT_IDENTIFIED | 이번 결과의 주장 범위 밖 |

NO-GO는 이번 모델·Runtime·탐색 Grid의 증거 부족을 뜻한다.
모든 Dynamic 정책의 효과가0이라는 증명은 아니다.

탐색 시작은 2026-11-16T16:33:39+00:00다.
GPU 실험 Admission 종료는 2026-11-17T01:20:41+00:00다.
이 구간의 Wall-clock 연구 시간은8.784시간이다.
GPU 시작·정합성 Capture·CPU 분석·Lease 대기를 포함한다.
Exclusive GPU Compute Time이 아니다.
이후 Archive·보고·Push 시간은 별도다.

새 문서의 독립3회에서도 다음 조건부 TTFT 우위를 확인했다.

| 조건 | Static Layer/Token TTFT p95, 95% CI | TTFT 감소 | Native 대비 Decode5% |
|---|---|---:|---|
| Output16, QPS6.4 | 0.906937 [0.859896, 0.941932] | 9.306% | PASS |

이 우위는 동결한 Completed-prefix Token 후보와 Static 후보의 비교다.
Streaming Token Pipeline이나 전역 최적 Token Scheduler를 이긴 결과가 아니다.

반대로 Token이 더 빠른 조건도 확인했다.

| 조건 | Static Layer/Token TTFT p95, 95% CI |
|---|---|
| Output64, QPS4.8 | 1.209814 [1.152767, 1.248021] |
| Output64, QPS6.4 | 1.098026 [1.058497, 1.133938] |

Dynamic의 Strong Static 대비 SLO Rate 우위와 Native-relative Decode 보호를 함께 통과한 Primary 조건 수는 0개다.
이 수치만으로 Matched Geometry의 독립 정책 기여까지 통과했다고 판정하지 않는다.

## 환경·보존 계약

- Model: `Qwen/Qwen3-8B`.
- Revision: `b968826d9c46dd6066d109eabc6255188de91218`.
- Original BF16, Exact Causal Attention, Full36 Decode, Greedy와 Thinking Disabled.
- H100 SXM 80GB GPU0/1의 같은 2GPU.
- GPU0 UUID: `GPU-f217c8a0-1142-20f4-d84b-af29f3a47a0d`.
- GPU1 UUID: `GPU-a77f3471-67d4-20b0-9fab-e502d4de5adb`.
- vLLM0.26.0+cu129, PyTorch2.11.0+cu129, NIXL1.4.1.
- GPU당 KV42GiB. Prefix Cache 비활성화.
- Publication Parent: `f0e80395aef488a7b3c6148d4155784f87291df2`.
- 새 Branch: `flashvep/dense-ar-layer-token-dynamic-headroom-20261117`.

기존 User Checkout과 Staged Index를 사용해 Commit하지 않았다.
별도 Index와 Commit Tree로 새 Branch를 공개한다.
이전 Source 435개와 외부 Runtime의 보존 검사를 수행한다.
추가 Worker 의존 Source140개도 Publication Parent와 Byte Equality를 확인했다.
원래 Weight Shard 5개의 SHA256은 Content-addressed 원본과 일치했다.
실험별 Config, Command, Model Revision, GPU Map, Source ZIP과 Hash를 보존한다.
실패한 실행도 삭제하지 않는다.

P와 D는 모두 전체 Model Weight를 보유한다.
Layer Weight를 영구 분할하는 PP2가 아니다.
D는 모든 방법에서 정상적인 Full36 Decode를 수행한다.

## 데이터·SLO·통계

Primary Input은 8000 Tokens다.
Output16/64는 `ignore_eos=True`로 고정했다.
새 LongBench 원문128개는 기존 Discovery 원문 ID와 첫7900-token Hash가 겹치지 않는다.
Discovery의 최적 설정을 고른 뒤 Candidate 평가 전에 동결했다.
같은 Output/QPS/Seed에서는 Prompt와 Arrival Trace를 모두 맞췄다.
Seeds는 2026111740/2026111741/2026111742다.
한 번만 실행한 탐색 QPS에는 독립 반복 CI를 만들지 않는다.

Mean TBT는 Coordinator가 전달한 연속 Token 간격의 평균이다.
TPOT는 Request별 `(last−first)/(Output−1)`이다.
First ITL을 포함한다.
요청의 TTFT·TPOT·모든 ITL이 통과해야 SLO Token에 포함한다.
Worker 간격이 정상이어도 Client SLO 실패를 통과로 바꾸지 않는다.

| Output | Regime | TTFT, ms | ITL, ms | TPOT, ms |
|---|---|---:|---:|---:|
| 16 | TTFT-priority | 2000 | 54 | 52 |
| 16 | Balanced | 5000 | 34 | 33 |
| 16 | Decode-priority | 10000 | 30 | 29 |
| 64 | TTFT-priority | 2000 | 55 | 53 |
| 64 | Balanced | 5000 | 35 | 33 |
| 64 | Decode-priority | 10000 | 31 | 29 |

이 Threshold는 기존 Frozen SLO다.
이번 결과를 보고 변경하지 않았다.
Primary 선정은 Decode-priority의 SLO-qualified Output Token Rate를 사용했다.
다른 두 Regime도 원자료와 통계 파일에 함께 보존했다.

아래 값은 Start별 Metric의 평균이다.
Tail을 모든 Token에 Pool한 값이 아니다.
95% CI는 Arrival Seed를 짝지은 독립 Server Start Bootstrap이다.
n=3은 작은 표본이다.
CI는 Pointwise이며 다중 비교 보정은 적용하지 않았다.
Native-relative ITL p99와 TPOT p95의 CI 상한이 모두1.05 이하여야 보호를 통과한다.
Absolute SLO와 Relative Gate는 별도 판정이다.

## 새 문서 Best-vs-Best — MEASURED

### Input8000/Output16, Offered QPS6.4

| 방법 | 독립 Start | Mean TTFT, ms | TTFT p95, ms | Mean TBT, ms | ITL p99, ms | TPOT p95, ms | Raw Output tokens/s | SLO Output tokens/s | Joint, % | 최대 ITL, ms |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Native | 3 | 9058.148 | 16226.447 | 23.292 | 24.123 | 23.329 | 55.422 | 30.041 | 54.1667 | 25.458 |
| Colocated | 3 | 829.852 | 1794.113 | 163.509 | 297.207 | 281.026 | 96.310 | 4.253 | 4.4271 | 298.274 |
| Token | 3 | 3991.341 | 6354.998 | 23.387 | 24.697 | 23.781 | 76.703 | 76.503 | 99.7396 | 31.575 |
| Static | 3 | 3674.296 | 5763.580 | 23.300 | 24.426 | 23.736 | 78.594 | 78.594 | 100.0000 | 27.245 |
| Dynamic | 3 | 4106.705 | 6209.982 | 23.326 | 24.954 | 23.741 | 77.679 | 77.679 | 100.0000 | 27.372 |

비율의 방향은 Candidate/Reference다. 대괄호는 95% CI다.

| Reference → Candidate | n | TTFT p95 비율 | ITL p99 비율 | TPOT p95 비율 | SLO Output 비율 | Native 대비 Decode5% Gate |
|---|---:|---|---|---|---|---|
| Native → Token | 3 | 0.391644 [0.345962, 0.434882] | 1.023811 [1.012733, 1.034344] | 1.019401 [1.005930, 1.035840] | 2.546625 [2.313220, 2.886841] | PASS |
| Native → Static | 3 | 0.355197 [0.313325, 0.404002] | 1.012560 [0.991172, 1.040000] | 1.017458 [1.001997, 1.036132] | 2.616229 [2.352952, 2.996361] | PASS |
| Native → Dynamic | 3 | 0.382707 [0.354537, 0.409417] | 1.034447 [1.000057, 1.065044] | 1.017677 [0.999591, 1.037738] | 2.585777 [2.292580, 2.981407] | NOT_ESTABLISHED |
| Token → Static | 3 | 0.906937 [0.859896, 0.941932] | 0.989010 [0.969527, 1.009073] | 0.998094 [0.996089, 1.000280] | 1.027332 [1.017158, 1.037848] | 해당 없음 |
| Static → Dynamic | 3 | 1.077452 [0.995131, 1.200123] | 1.021616 [1.001731, 1.043295] | 1.000215 [0.996957, 1.003233] | 0.988360 [0.970581, 1.004884] | 해당 없음 |

Native와 같은 Arrival Trace의 실제 유한 표본 QPS는 7.061314/5.846250/6.522305다.
Offered Rate와 유한 표본 Rate를 구분한다.

### Input8000/Output64, Offered QPS4.8

| 방법 | 독립 Start | Mean TTFT, ms | TTFT p95, ms | Mean TBT, ms | ITL p99, ms | TPOT p95, ms | Raw Output tokens/s | SLO Output tokens/s | Joint, % | 최대 ITL, ms |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Native | 3 | 6872.459 | 11105.885 | 23.646 | 25.447 | 24.095 | 210.527 | 177.667 | 84.3750 | 69.285 |
| Colocated | 3 | 489.616 | 958.171 | 65.823 | 294.385 | 108.592 | 289.349 | 4.521 | 1.5625 | 300.033 |
| Token | 3 | 4999.012 | 7730.321 | 24.053 | 26.131 | 24.122 | 233.699 | 227.635 | 97.3958 | 47.726 |
| Static | 3 | 6366.797 | 9352.247 | 24.070 | 26.272 | 24.090 | 223.682 | 211.065 | 94.2708 | 27.479 |
| Dynamic | 3 | 6005.238 | 8616.312 | 24.068 | 25.809 | 24.085 | 230.340 | 218.606 | 94.7917 | 42.653 |

비율의 방향은 Candidate/Reference다. 대괄호는 95% CI다.

| Reference → Candidate | n | TTFT p95 비율 | ITL p99 비율 | TPOT p95 비율 | SLO Output 비율 | Native 대비 Decode5% Gate |
|---|---:|---|---|---|---|---|
| Native → Token | 3 | 0.696056 [0.668338, 0.718497] | 1.026872 [1.018527, 1.035927] | 1.001147 [1.000786, 1.001489] | 1.281242 [1.147091, 1.427459] | PASS |
| Native → Static | 3 | 0.842098 [0.770538, 0.896699] | 1.032424 [1.023527, 1.039948] | 0.999799 [0.998755, 1.000874] | 1.187977 [1.077616, 1.304143] | PASS |
| Native → Dynamic | 3 | 0.775833 [0.697152, 0.850962] | 1.014230 [1.006330, 1.020104] | 0.999580 [0.998817, 1.000306] | 1.230421 [1.062536, 1.385024] | PASS |
| Token → Static | 3 | 1.209814 [1.152767, 1.248021] | 1.005406 [0.996016, 1.017852] | 0.998654 [0.997477, 1.000009] | 0.927207 [0.873264, 0.967537] | 해당 없음 |
| Static → Dynamic | 3 | 0.921309 [0.890763, 0.961013] | 0.982378 [0.969530, 0.995998] | 0.999781 [0.999432, 1.000063] | 1.035728 [0.972472, 1.105970] | 해당 없음 |

Native와 같은 Arrival Trace의 실제 유한 표본 QPS는 5.295985/4.384687/4.891729다.
Offered Rate와 유한 표본 Rate를 구분한다.

### Input8000/Output64, Offered QPS6.4

| 방법 | 독립 Start | Mean TTFT, ms | TTFT p95, ms | Mean TBT, ms | ITL p99, ms | TPOT p95, ms | Raw Output tokens/s | SLO Output tokens/s | Joint, % | 최대 ITL, ms |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Native | 3 | 9817.651 | 16843.967 | 23.640 | 25.469 | 24.096 | 211.478 | 101.343 | 47.9167 | 27.337 |
| Colocated | 3 | 930.199 | 1956.026 | 122.922 | 303.156 | 174.854 | 358.041 | 2.728 | 0.7812 | 425.651 |
| Token | 3 | 8040.863 | 13171.329 | 24.043 | 25.838 | 24.100 | 235.219 | 167.832 | 71.3542 | 27.495 |
| Static | 3 | 8884.558 | 14462.466 | 24.074 | 25.299 | 24.088 | 229.939 | 138.311 | 60.1562 | 26.076 |
| Dynamic | 3 | 8808.851 | 14578.440 | 24.079 | 26.054 | 24.102 | 229.965 | 143.704 | 62.5000 | 28.090 |

비율의 방향은 Candidate/Reference다. 대괄호는 95% CI다.

| Reference → Candidate | n | TTFT p95 비율 | ITL p99 비율 | TPOT p95 비율 | SLO Output 비율 | Native 대비 Decode5% Gate |
|---|---:|---|---|---|---|---|
| Native → Token | 3 | 0.781961 [0.758234, 0.812865] | 1.014494 [1.002809, 1.027265] | 1.000164 [0.999459, 1.000942] | 1.656085 [1.456341, 1.905805] | PASS |
| Native → Static | 3 | 0.858614 [0.833781, 0.879535] | 0.993329 [0.978723, 1.008931] | 0.999636 [0.998917, 1.000499] | 1.364786 [1.266781, 1.522327] | PASS |
| Native → Dynamic | 3 | 0.865499 [0.845964, 0.895656] | 1.022977 [1.008535, 1.041305] | 1.000239 [0.999778, 1.000703] | 1.418003 [1.265761, 1.595955] | PASS |
| Token → Static | 3 | 1.098026 [1.058497, 1.133938] | 0.979137 [0.954854, 1.000341] | 0.999472 [0.999036, 0.999852] | 0.824104 [0.772725, 0.882125] | 해당 없음 |
| Static → Dynamic | 3 | 1.008019 [0.972408, 1.045378] | 1.029848 [1.005984, 1.052665] | 1.000603 [1.000164, 1.000966] | 1.038993 [0.973296, 1.119091] | 해당 없음 |

Native와 같은 Arrival Trace의 실제 유한 표본 QPS는 7.061314/5.846250/6.522305다.
Offered Rate와 유한 표본 Rate를 구분한다.

### 선정된 실행 설정

Output16의 설정은 다음과 같다.

| 방법 | Discovery에서 동결한 설정 |
|---|---|
| Token | fine_Token2400_standard_D4_N128_O16_q6.4_v1 / P slots4 / D admission4 / Suffix2400 |
| Static | fair_Layer_Last12_P4_D8_N128_q6.4_v1 / P slots4 / D admission8 / Last12 / Fraction1.0 |
| Dynamic | adaptive_source_flow_H2_N128_O16_q6.4_v1 / P slots4 / D admission4 / Horizon2 |

Output64의 설정은 다음과 같다.

| 방법 | Discovery에서 동결한 설정 |
|---|---|
| Token | O64_Token_S1024_P4_D8_N64_q4.8_v1 / P slots4 / D admission8 / Suffix1024 |
| Static | macro8_cal_Last4_N64_O64_q4.8_v1 / P slots4 / D admission8 / Last4 / Fraction1.0 |
| Dynamic | flow8_H2_N64_O64_q4.8_v1 / P slots4 / D admission8 / Horizon2 |

Source Group은 Coordinator Macro다.
P slots4/Token Budget32000에서 Source Macro8을 한 GPU Batch8로 표시하지 않는다.
실제 P/D Shape는 각 Run의 `measured_geometry_*.json`에 기록했다.
Colocated는 사용자 지정 B8192/S1024의 Standard Continuous DP2다.
이를 모든 SLO에 대한 전역 최적 Colocated라고 주장하지 않는다.
이번 캠페인은 중단한 Aggregated B512/S16 탐색을 재개하지 않았다.
Token Arm의 D-side B512 검사는 다른 실험이다.

## 같은 작업량 Control — 독립3회

N144, Output16, QPS6.4에서 같은64개 Discovery 원문을 선언된 주기로 반복했다.
새 원문 Holdout은 아니다.
Seeds는 2026111760/1761/1762다.
LayerLast12와 Token2656/2672는 실제 P/D GEMM Rows와 유효 KV Bytes를 맞췄다.
실제 Source/Destination Group도 일치했다.
Layer의 Hidden/Residual과 Source Credit 점유는 같지 않다.

| Metric | Layer/Token | 독립 Start 기반95% CI |
|---|---:|---|
| TTFT p95 | 0.860530 | [0.791981, 0.927848] |
| ITL p99 | 1.009008 | [0.999586, 1.019340] |
| TPOT p95 | 1.001546 | [1.000400, 1.002979] |
| Raw Output tokens/s | 1.044749 | [1.023927, 1.064201] |
| SLO Output tokens/s | 1.080545 | [1.026571, 1.144348] |

TTFT p95는13.947% 감소했다.
SLO Output은8.054% 증가했다.
Raw Output은4.475% 증가했다.
Raw 개선이5% 이상이라고 입증한 것은 아니다.
Layer의 Native-relative Decode5% Gate도 통과했다.
작은 Start 수와 반복 Prompt Pool의 한계를 유지한다.
Last8의 초기 n=1 Control에서는 반대로 Token이 더 빨랐다.
따라서 축의 보편적 우위가 아니라 조건부 이점이다.

## Layer/Token의 원인·정합성

Token Arm은 P의 완성된 Prefix KV를 D로 전송한다.
D는 원래 Position의 나머지 Token을 Full36으로 계산한다.
이전 Prefix를 다시 계산하거나 두 Chunk의 독립성을 가정하지 않는다.
Layer별 KV Streaming을 사용한 Token Pipeline은 `NOT_RUN`이다.
P-side Chunk512 PD는 N32 State Gate와 N64 성능 Pilot을 수행했다.
다른 P-side Chunk의 광범위한 튜닝과 새 N128 독립3회는 `NOT_RUN`이다.
Suffix Layerℓ는 Prefix KVℓ와 Suffix Hiddenℓ−1에 의존한다.
따라서 Prefix의 다음 Layer와 겹치는 별도 구현은 가능할 수 있다.
현재 NIXL의 Layer-load Wait는 No-op이며 이 Pipeline을 제공하지 않는다.
이번 우위는 검증한 Completed-prefix Token 구현과 탐색 Grid에 한정한다.
Layer Arm은 Partial KV와 Hidden/Residual을 전달한다.
D는 잔여 Layer 뒤에 정상 Full36 Decode를 수행한다.

Suffix512/1024/2048/4096과 인접 경계 Plan은 N8→N32 State Gate를 통과했다.
새 Prompt128의 주요 Token Suffix와 모든 Layer Cut도 전체 Prompt State를 통과했다.
최종 후보의 추가 N128 Gate는 별도 파일에 기록한다.
Source Macro8/P slots4와 실제 D Prefill Batch8의 N32 State 검사도 별도로 수행했다.
Clean 성능 실행마다 Output Gate를 적용했다.
성능 Run마다 전체 Hidden을 복사했다는 뜻은 아니다.
Held-out PD의 KV·ACK·사용한 Reservation/Hidden Pool 복원도 검사했다.
정상 종료 검사이며 Cancellation이나 GPU Failure 복구의 보장은 아니다.

같은 GEMM Rows의 Last12 Control에서 Token의 D Attention Pair는 Layer의1.666580677배다.
이는 늦은 Query가 더 긴 Causal Prefix를 참조하기 때문이다.
전체 모델 계산을 추가하거나 생략한 것이 아니다.
이는 `STRUCTURAL_ACCOUNTING`이며 GPU 시간 비율이 아니다.
Token 경로는 P Prefix의 불필요한 Sample을 버리지만 현재 Engine은 이를 계산한다.
Layer 경로도 사용하지 않는 P Sample 경로를 유지한다.
이를 Token만의 고유한 비용으로 집계하지 않는다.
해당 공통 LM-head/Sampling을 제거한 최적 경로는 별도로 검증하지 않았다.
같은 Token-layer Rows는 모든 Operator의 동일 작업량을 뜻하지 않는다.
Token Arm의 Last0 Histogram은 Layer Cut을 하지 않는다는 뜻이다.
실제 Token Suffix 이전은 별도 D Rows와 Migrated Request Count로 확인한다.

고립된 Projection의 GPU1 합산 추정은 Layer206.02829ms, Token212.32665ms였다.
Token/Layer는1.030570배였다.
이는 실제 Layer0 Weight의 Shape별 Event 중앙값을 반복한 추정이다.
36개 Layer의 Exclusive Kernel Time이 아니다.
새 GEMM Kernel로 일부 차이를 줄일 수 있다.
Attention Work의 축별 분배는 GEMM 변경만으로 같아지지 않는다.
그러나 이 사실만으로 Serving 우위가 보장되지는 않는다.

주요 비용의 배타적 기여율은 `NOT_IDENTIFIED`다.
GEMM·Attention·Queue·Transfer의 중앙값을 더해 TTFT 개선을 분해하지 않았다.
Layer의 추가 Hidden/Residual은 Request별131,072,000Bytes다.
유효 KV Bytes가 같아도 현재 Layer Cache의 Source Block Credit은 더 크다.
이 불리한 차이도 그대로 유지했다.

## TP2 — 최초 차이 식별, Strict Baseline 차단

기존 N8 중 가장 이른 Output 분기는 Request4의 Position6이었다.
새 Eager 순차 Capture에서는 같은 Prefix의 Layer0 `self_attn.o_proj`가 최초 차이였다.
수집한 앞선 Q/K/V·RoPE·Attention Rows는 일치했다.
Prefill 마지막8 Rows의 Max Absolute Error는0.015625였다.
RMS Error는0.00020561653946060687였다.
Relative L2는0.003151108743622899였다.

실제 Operand와 Weight Shard를 합치면 TP1과 정확히 일치했다.
Full-K BF16 GEMM은 실제 TP1을 재현했다.
K/2의 BF16 부분합 두 개와 합산은 실제 TP2를 재현했다.
Collective 전의 Local BF16 Projection 반올림이 이 경계의 차이를 설명한다.
실제 Collective Backend는 PYNCCL이었다.
Custom All-reduce는 비활성화됐다.
PyTorch의 `BF16_reduced_precision_reduction` 옵션도 거짓이었다.
NCCL 내부 Precision이나 Ring/Tree Kernel 선택까지 계측한 것은 아니다.
잘못된 Weight Shard나 누락된 Rank 합산으로 설명되지 않는다.
이 결론은 모든 TP2 Runtime 오류를 배제하는 증명은 아니다.

Step6의 TP1 Logit은 Token1986=28.125, Token785=28.0이었다.
TP2는 둘 다27.875였고 실제 Argmax는785였다.
FP32 Replay는 진단용이다.
FP32나 Token 덮어쓰기를 BF16 성능 경로에 넣지 않았다.

판정은 `NUMERICAL_NON_EQUIVALENCE_IDENTIFIED`다.
TP1 Strict Parity는 `FAILED`다.
전체 TP2 KV/Hidden의 Within-TP2 검증은 `NOT_MEASURED`다.
통일된 Client Timestamp의 적격 TP2 성능 비교는 `NOT_RUN`이다.
TP2를 Main Table에 넣지 않는다.

## Dynamic의 독립 기여와 Matched Geometry

실제 Histogram/Geometry Control은 3개 독립 Pair가 Matching을 통과했다.

| Metric | Dynamic/Matched Control, 95% CI |
|---|---|
| TTFT_p95_ms | 0.951320 [0.921702, 1.001450] |
| ITL_p99_ms | 1.014996 [1.000390, 1.032196] |
| TPOT_p95_ms | 0.998726 [0.995691, 1.001250] |
| output_tokens_s | 1.016152 [1.007067, 1.025663] |
| Decode-priority SLO Output tokens/s | 1.016152 [1.007067, 1.025663] |

Output64/QPS4.8의 추가 Matched Control Admission은False였다.
Native-relative Gate와 SLO Rate의 반복 우위가 함께 성립하지 않으면 추가 GPU Control을 실행하지 않았다.

Output64/QPS6.4의 추가 Matched Control Admission은False였다.
Native-relative Gate와 SLO Rate의 반복 우위가 함께 성립하지 않으면 추가 GPU Control을 실행하지 않았다.

이 Control은 실제 Cut Histogram·완료 Work·Source/Destination 구성원을 맞춘다.
Role별 실제 GPU Shape Histogram도 별도 검사한다.
Cut의 시간 배치는 의도적으로 다르다.
Control은 사후 Aggregate Cut Count를 사용한다.
따라서 배포 가능한 Best Static을 대신하지 않는다.
Cache 상태와 Retention Duration까지 같다는 뜻도 아니다.

정책은 이미 도착한 FIFO 작업만 사용한다.
P Prefix 제출 전에 Last0/2/4/8/12를 평가한다.
현재 Group의 KV·Hidden Reservation과 유효 Deadline Conflict를 검사한다.
기존 Cohort가 끝나기 전에는 새 Suffix를 Admission하지 않는다.
미래 Arrival 또는 미래 실측 완료시각은 Online 입력이 아니다.
110ms 추가 Margin과 실측 Tail은 WCET가 아니다.

실제 Source 가변 Group Discovery에서 다섯 Cut을 모두 선택했다.
선택 횟수를 늘리기 위해 작은 Cut을 강제하지 않았다.
그러나 Cut 다양성만으로 Strong Static보다 좋은 정책이라고 판정하지 않는다.
새 문서의 Static/Dynamic 비율과 Native-relative CI를 함께 검사해야 한다.

### 실제 Online Predictor 검증 — TRACE_ANALYSIS

아래 표본은 같은 Run 안의 상관된 Source Group이다.
Group 수를 독립 Server Start 수로 사용하지 않는다.
First-output 예측은 GPU Kernel 완료가 아니라 Client 첫 출력 경계를 대상으로 한다.

| Run | Group 수 | P-ready MAE, ms | First-output MAE, ms | First-output 절대오차 p95, ms | 과소예측, % | Upper-bound Coverage, % | Controller p95, ms |
|---|---:|---:|---:|---:|---:|---:|---:|
| heldout_Dynamic_N128_O16_q6.4_s2026111740_v1 | 34 | 19.913 | 84.612 | 114.280 | 5.882 | 100.000 | 1.132 |
| heldout_Dynamic_N128_O16_q6.4_s2026111741_v1 | 36 | 12.386 | 35.862 | 81.728 | 22.857 | 100.000 | 1.953 |
| heldout_Dynamic_N128_O16_q6.4_s2026111742_v1 | 33 | 14.265 | 70.409 | 95.459 | 6.061 | 100.000 | 1.484 |
| heldout_Dynamic_N128_O64_q4.8_s2026111740_v1 | 16 | 29.394 | 74.482 | 147.983 | 12.500 | 100.000 | 1.808 |
| heldout_Dynamic_N128_O64_q4.8_s2026111741_v1 | 16 | 35.189 | 84.172 | 163.721 | 6.250 | 100.000 | 3.503 |
| heldout_Dynamic_N128_O64_q4.8_s2026111742_v1 | 16 | 13.502 | 208.693 | 313.277 | 0.000 | 100.000 | 3.530 |
| heldout_Dynamic_N128_O64_q6.4_s2026111740_v1 | 16 | 11.825 | 120.262 | 236.198 | 0.000 | 100.000 | 4.010 |
| heldout_Dynamic_N128_O64_q6.4_s2026111741_v1 | 16 | 22.725 | 95.358 | 216.695 | 37.500 | 100.000 | 3.507 |
| heldout_Dynamic_N128_O64_q6.4_s2026111742_v1 | 16 | 12.171 | 139.120 | 304.332 | 0.000 | 100.000 | 3.483 |

## Oracle와 변동성 — OFFLINE_ORACLE와 TRACE_ANALYSIS

상태 갱신 Replay는 P/D 완료시간과 Credit Lifetime을 갱신했다.
모델은 실제 Kernel Trace가 아니라 Host Service Enclosure를 사용했다.
유한 Beam은 전역 최적해나 성능 상한이 아니다.

같은 Mixed Output 모델에서 Joint95% Static은231.036054 SLO tokens/s였다.
Beam1024는235.761290이었다.
추가 Headroom은2.045%였다.
이는 유망 기준5%보다 작았다.
이 Grid의 대규모 Mixed Sweep은 진행하지 않았다.

실제 Mixed Output n=1 Pilot에서 Online H2는230.700 SLO tokens/s였다.
Fraction Static은230.853이었다.
두 방법의 Joint Attainment는 각각90.625%와88.28125%였다.
둘 다Joint95% Gate를 통과하지 못했다.
미래 정보로 선정한 Beam Plan의 GPU Diagnostic은 Online 성과가 아니다.
모델의 작은 Headroom을 실측 Dynamic 이득으로 바꾸어 보고하지 않는다.

Host 예산의 변동성은 기록했다.
이를 물리적인 Decode-safe GPU Idle로 해석하지 않는다.
실제 Mixed-input, Natural EOS와 Bursty Serving의 폭넓은 검증은 `NOT_RUN`이다.
Input8000 외에는 현재 정책이 보수적으로 Native를 선택한다.
모든 Heterogeneous Workload에서 Dynamic이 불가능하다고 주장하지 않는다.

## 계측·실패·Runtime 비용

CUDA Event A/B/A에서 Layer Capture의 네 Metric 변화는5% 이내였다.
Token Capture의 TTFT p95 비율은1.081923배로 간섭 기준에 실패했다.
Token Event Capture는 Diagnostic으로만 사용한다.
Event Enclosure를 Exclusive Kernel Time이나 Kernel Union으로 표시하지 않는다.
두 Run의 시간을 합쳐 하나의 실제 Timeline으로 만들지 않는다.

별도 CPU Credit Lane은 RPC 지연을 줄였다.
그러나 전체 처리량 개선은 재현하지 못했다.
이 옵션은 최종 Main 정책에서 비활성화했다.
이를 Dynamic의 기여로 포함하지 않는다.

Mixed Static의 진행 정지 한 건은 실패로 보존했다.
D Free16613 Blocks 중 미래 예약16152 Blocks가 남았다.
유효461 Blocks는 Native Prompt 한 개의501 Blocks보다 작았다.
전체 Cohort Import Barrier와 미래 Reservation이 순환 대기를 만들었다.
마지막 Callback 순서는 `NOT_IDENTIFIED`다.
예약 상한32→16의 별도 진단은 Output128개와 Credit 복원을 통과했다.
실제 Cut이 달라졌으므로 성능 비교로 사용하지 않았다.

Mixed H2의 Client Joint 실패12개 중8개는 ITL 실패였다.
그8개는 Worker 간격이 기준 이내였다.
Client 간격과 Worker 간격의 차이는 Delivery Lag 변화로 확인됐다.
Client SLO를 Worker 경계로 바꾸지 않았다.
특정 함수의 배타적 원인은 `NOT_IDENTIFIED`다.

새 Output64/QPS4.8의 Seed2026111741에서도 Client ITL 실패가8 Requests에서 발생했다.
TTFT와 TPOT 실패는 없었다.
Worker 간격은 모두31ms 이내였다.
q56의 Client42.641804ms는 Worker24.019083ms와 Delivery Lag 변화18.622721ms로 분리됐다.
이8개 Request의512 Output Tokens는 원래 SLO 집계에서 제외했다.
이 지연을 제거한 Counterfactual Goodput을 Main 결과로 사용하지 않았다.

일부 UCX Context Query Code4는 Coordinator COMPLETE 뒤에 발생했다.
이는 Serving 중의 CUDA Launch Failure와 별도로 분류했다.
프로세스 전체가 Error-free였다고 주장하지 않는다.
Warmup 없는 State Capture와 측정 전 JIT 경고도 Clean 성능과 분리했다.
초기 Hook·검증기·Warmup·분석 Schema 실패를 모두 보존했다.

## 최종 질문에 대한 답

1. TP2의 첫 유효 Tensor 차이는 수집한 Layer0 Output Projection이다.
2. 같은 Operand Replay가 BF16 Local Projection 반올림의 수치 차이를 재현했다.
3. Strict TP1 Parity를 가진 TP2 성능 Baseline은 확보하지 못했다.
4. 실제 Dense AR Token-axis P→D 계산 이전은 구현했다.
5. Causal Prefix KV·전체 Prompt KV·Hidden·Work·ACK Gate를 별도 State 실행에서 검사했다.
6. 같은 Last12 작업량에서는 Layer가 유리했다. Last8 초기 Control은 반대였다.
7. 같은 SLO의 우위 범위는 위 독립 Start Table에 한정한다.
8. Attention 분배와 일부 GEMM Shape 차이는 확인했다. 배타적 기여율은 미식별이다.
9. Host-state 예산과 실제 Cut의 변화는 관측했다. 물리적 Safe Idle은 미측정이다.
10. 강한 Joint95% Static 대비 근사 Beam의 모델 Headroom은2.045%였다.
11. Mixed Pilot의 Dynamic SLO Rate는 Static보다 높지 않았다. Primary 세 조건에서도 SLO Rate의 반복 우위를 입증하지 못했다.
12. Token Suffix·Chunk·Group과 Static Cut·Fraction·Group을 Discovery에서 탐색했다.
13. 실제 Histogram·Work·Group·GPU Shape Matching은 별도 파일에 기록한다.
14. Mean TTFT·Mean TBT·ITL·Raw/SLO Token Rate는 위 Table과 Per-start CSV에 보존했다.
15. 구현·State Gate의 통과와 Performance·Dynamic 독립 기여의 통과를 구분한다.

같은 작업량의 조건부 Layer 이점은 조건과 제한을 명시한 실험 결과로 기록할 수 있다.
전역 최적 Token Baseline에 대한 우위나 Dynamic의 독립 Systems Contribution을 확정하지 않는다.
전역 우위·Sustainable Capacity·미실행 Workload의 성공을 주장하지 않는다.

## 최종 검증과 Resource 복원

새 문서의 Clean Serving 평가74개 Start를 완료했다.
이 중 세 Primary 조건은 각각 Method5개×독립 Start3개다.
추가 QPS의 Candidate1회 결과는 탐색으로만 표시한다.
실험 Inventory의 상태 수는{'CORRECTNESS_ONLY': 53, 'MEASURED': 195, 'FAILED': 7, 'NOT_RUN': 18}다.
이 수는 정합성 Capture와 진단도 포함한다.
모든 MEASURED가 Primary 적격 성능을 뜻하지 않는다.
Raw Directory가 있지만 완료 Receipt가 없는 초기 시도도 실패로 보존했다.
CPU 계약·분석 검사는24개를 통과했다. Return Code는0다.
GPU0–7의 승인된 Burner 복원 검사는True다.
실제 양의 NVML 사용률과 소유한 Worker를3회 연속 확인했다.
실험의 Quiet Lane에서는 GPU0–7을 예약해 승인 Burner를 중단했다.
CPU-only 구간과 최종 종료 뒤에는 승인된 Burner를 복원했다.
다른 사용자의 GPU Process는 종료하지 않았다.

Archive는 원본33,886,572,326Bytes를 보존한다.
압축 크기는2,979,414,163Bytes이며48MiB Part는60개다.
전체 Member의 압축 해제 후 SHA256 검사는True다.
로컬 원본은 그대로 남겼다.

## Figure·근거·재현

- [정합성과 실행 계약](dense_ar_layer_vs_token_correctness_and_execution.md).
- [Layer/Token 성능과 원인](dense_ar_layer_vs_token_online_results.md).
- [TP2 수치 원인](dense_ar_tp2_numerical_correctness_root_cause.md).
- [Dynamic/Static와 Oracle](dense_ar_dynamic_vs_static_oracle_and_online_results.md).
- [Host 예산의 변동성](dense_ar_safe_budget_variance_analysis.md).
- [Figure13개와 Host/Stream Timeline](dense_ar_layer_token_dynamic_main_figures.md).
- [전체 독립 Start와 CI](dense_ar_layer_token_dynamic_headroom_20261117/analysis/heldout_start_statistics.json).
- [Per-start CSV](dense_ar_layer_token_dynamic_headroom_20261117/analysis/heldout_per_start_metrics.csv).
- [실제 Matched GPU Geometry 통계](dense_ar_layer_token_dynamic_headroom_20261117/analysis/heldout_policy_matched_actual_geometry_statistics.json).
- [정상 KV·ACK·Credit 복원](dense_ar_layer_token_dynamic_headroom_20261117/analysis/heldout_normal_lifetime_credit_audit.json).
- [같은 작업량 독립3회](dense_ar_layer_token_dynamic_headroom_20261117/analysis/matched_Last12_independent_start_statistics.json).
- [실험 Inventory와 실패 상태](dense_ar_layer_token_dynamic_headroom_20261117/analysis/experiment_inventory.json).
- [CPU 검사와 이전 Source 보존](dense_ar_layer_token_dynamic_headroom_20261117/analysis/final_CPU_verification.json).
- [Weight SHA256 검사](dense_ar_layer_token_dynamic_headroom_20261117/state/model_weights_SHA256_verification.json).
- [GPU0–7 Burner 복원](dense_ar_layer_token_dynamic_headroom_20261117/state/final_GPU_resource_restoration.json).
- [손실 없는 Archive Manifest](dense_ar_layer_token_dynamic_headroom_20261117/archives/large_evidence_manifest.json).
- [공개 파일 SHA256 Manifest](dense_ar_layer_token_dynamic_headroom_20261117/state/publication_SHA256_manifest.json).
- [Evidence 복원·분석 안내](dense_ar_layer_token_dynamic_headroom_20261117/README.md).

큰 Evidence는 손실 없는 분할 Archive로 공개한다.
원본별 SHA256과 압축 해제 후의 SHA256을 비교한다.
로컬 원본은 삭제하지 않는다.
복원 명령과 Publication 검증은 Evidence Manifest와 Local Receipt에 기록한다.

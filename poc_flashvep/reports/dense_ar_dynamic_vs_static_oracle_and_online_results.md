# Dense AR Dynamic-vs-Static: Oracle와 Online 검증

## 검증할 기여

Dynamic의 기여는 더 많은 Request를 이전한 효과와 구분해야 한다.
실제 Cut의 Histogram, Token-layer Rows와 Source/Destination Group을 함께 검사한다.
같은 Layer 수라도 Group Fragmentation이 다르면 순수한 정책 비교가 아니다.
계획한 Group이 아니라 실제 완료한 Group을 검사한다.

모델은 Qwen/Qwen3-8B다.
Revision은 `b968826d9c46dd6066d109eabc6255188de91218`다.
Original BF16과 Exact Attention을 유지했다.
모든 Decode는 36개 Layer를 방문한다.
실험은 GPU0/1의 동일한 H100 SXM 80GB 두 대를 사용한다.
GPU별 KV Budget은 42GiB다.
SLO는 기존 Frozen 값을 유지했다.

## 구현

새 정책은 기존 Exact Partial Forward와 NIXL/Hidden Transport를 재사용한다.
Candidate는 Last0/2/4/8/12다.
이미 도착한 FIFO Group만 2~3개까지 Lookahead에 포함한다.
정책은 후보별 예상 SLO Token Rate와 Queue Waiting을 계산한다.
P Compute Saving만 최대화하지 않는다.
같은 점수에서는 더 작은 Cut을 우선한다.

P Prefix 제출 전에 Cut을 선택한다.
D의 KV·Hidden Credit을 예약한다.
현재 Decode Cohort의 Retirement 이전에는 Suffix를 실행하지 않는다.
기존 Decode의 유효 Deadline Conflict도 직접 거절 조건으로 사용한다.
새 Cohort의 Decode Step Bound와 해당 Request의 유효 TTFT Deadline도 검사한다.
Stale State나 Calibration하지 않은 Input에서는 Native로 돌아간다.
Commit 후에는 기존 Prefix를 버리지 않는다.

Lookahead의 뒤쪽 Group에는 예측 Credit만 적용된다.
실제 Reservation은 현재 Group에만 수행한다.
미래 Arrival을 정확히 안다고 가정하지 않는다.
정책이 모든 미래 SLO를 보장한다고 주장하지 않는다.

## 기존 Static이 강했던 이유

Input8000/Output16은 Request별 연산량이 거의 같다.
같은 Cut의 P Service와 D Residency도 반복된다.
강한 Static Cut이 P/D Pipeline을 잘 맞추면 상태별 조정의 추가 여지가 작다.
이는 실험으로 검사할 설명이다.
모든 Heterogeneous Workload에서 Dynamic이 불필요하다는 뜻은 아니다.

새로운 Fixed Group4 Discovery에서 Dynamic H2와 H3는 모두 같은 Histogram을 선택했다.
Last8은 12 Requests였다.
Last12는 116 Requests였다.
이 Histogram은 실제 Online 선택이다.
Last2/4를 강제로 선택한 정합성 시험과 구분한다.

Output64의 Source Group8/D Cohort8 Discovery에서는 다른 Cut이 실제로 선택됐다.
H2는 Last4 48개, Last8 8개와 Native 8개를 선택했다.
H3는 Last4 40개, Last12 8개와 Native 16개를 선택했다.
작은 Cut의 실제 사용은 확인했다.
그러나 더 강한 Static보다 좋은 결과인지 별도로 비교해야 한다.

P Source의 가변 Group을 허용한 Output16 실행도 수행했다.
H2는 Last0/2/4/8/12를 각각 2/6/4/4/112 Requests에서 선택했다.
H3는 각각 1/3/4/8/112 Requests에서 선택했다.
이는 강제 Cut 시험이 아닌 실제 Online 선택이다.
H2의 TTFT p95는 5913.454ms였다.
ITL p99는 24.488ms였고 Raw Output은 78.417 tokens/s였다.
H3의 대응 값은 6273.560ms, 24.300ms와 76.696 tokens/s였다.
이 두 Discovery 결과만으로 Strong Static 대비 추가 이득을 확인하지 못했다.

## Discovery 결과 — MEASURED, 각 n=1

| 조건 | 정책 | TTFT p95, ms | ITL p99, ms | Raw Output tokens/s |
|---|---|---:|---:|---:|
| N128, Output16, QPS6.4 | Flow H2, Group4 | 6728.313 | 25.164 | 76.530 |
| 같은 조건 | Flow H3, Group4 | 6301.199 | 24.328 | 77.513 |
| 같은 조건 | Static Last12, P4/D8 | 5859.287 | 24.265 | 78.866 |
| N64, Output64, QPS4.8 | Flow H2, Group8 | 6057.684 | 25.331 | 207.712 |
| 같은 조건 | Flow H3, Group8 | 9639.395 | 26.904 | 174.833 |
| 같은 조건 | Static Last4, Group8 | 5987.566 | 25.228 | 207.213 |

Source Group이나 D Geometry가 다른 행은 Best-vs-Best 탐색이다.
이를 같은 Geometry의 순수한 정책 비교로 표시하지 않는다.
Horizon을 늘린 결과가 항상 좋아지지는 않았다.
Discovery 한 번의 우열은 최종 통계적 판정이 아니다.

## 긴 Output의 Cohort Geometry

Output64에서는 가변 Source Group4가 오히려 나빴다.
같은 N64/QPS4.8의 Prompt·Arrival·GPU를 사용했다.
Adaptive Source H2의 TTFT p95는14456.718ms였다.
Raw Output은140.140 tokens/s였다.
실제 D Cohort는19개였다.
크기1/2/3/4의 Cohort가 각각1/4/1/13개였다.
첫 출력부터 마지막 출력까지의 Cohort Span 평균은1430.175ms였다.

Source Macro8/D Cohort8의 H2는8개 Cohort를 사용했다.
각 Cohort는8 Requests였다.
같은 Span 평균은1516.079ms였다.
Raw Output은207.712 tokens/s였다.
Static Last4의 대응 값은1516.551ms와207.213 tokens/s였다.

현재 Guard는 한 Cohort가 끝나기 전 다음 Cohort를 받지 않는다.
작은 Cohort는 비슷한 Decode Residency 동안 더 적은 Request를 처리한다.
이는 실제로 관측한 구현의 동시성 제한이다.
GPU에 물리적인 여유가 없다는 증거는 아니다.
Cost Model과 Cut도 달라 Geometry만의 배타적인 기여율은 식별하지 못했다.
Output64의 최종 Dynamic은 이 Discovery를 바탕으로 Macro8/D8을 동결했다.

## Strong Static과 Oracle

Static을 Last8 하나로 제한하지 않았다.
Last0/2/4/8/12, Fraction과 Group Geometry를 탐색했다.
Mixed Output에는 Cap별 Cut과 Cap별 Fraction도 포함했다.
Cap별 Fraction의 한 탐색은 9009개 조합을 평가했다.

실제 Macro8 Calibration을 사용한 Mixed Output 근사 Oracle는 약 4.2% Headroom을 보였다.
강한 Cap/Fraction Static을 포함하면 Beam256의 추가 이득은 사라졌다.
Joint95% 제한을 따로 적용한 확장 Beam1024의 이득은 약 2.045%였다.
이 결과는 해당 모델과 탐색 공간에 한정한다.
유한 Beam은 최적 Oracle의 상한이 아니다.
자세한 값과 Joint95% 제한은 [예산 분석](dense_ar_safe_budget_variance_analysis.md)에 기록한다.

모델에서 5% 이상의 유망한 Headroom을 확보하지 못했다.
따라서 같은 Grid의 대규모 Mixed Sweep은 진행하지 않았다.
작은 실제 Model-validation Pilot만 유지했다.
Mixed Output Online H2의 Raw Output은 236.108 tokens/s였다.
Joint Attainment는 90.625%였고 SLO Output은 230.700 tokens/s였다.
같은 Group8의 Fraction Static은 각각 239.065, 88.28125%와 230.853이었다.
두 SLO Token Rate의 차이는 매우 작았다.
두 방법 모두 이 Pilot의 Joint95% Gate를 통과하지 못했다.

Controller Zero도 같은 Mixed Output Pilot에서 실행했다.
Raw Output은213.464 tokens/s였다.
Native는213.887 tokens/s였다.
그러나 Controller Zero는 Source Macro8/D8이었다.
Native의 Source/Destination Geometry는 달랐다.
이 두 값을 순수 Controller Overhead 차이로 해석하지 않는다.
같은 Geometry의 Held-out Controller Zero 독립3회는 `NOT_RUN`이다.

추가한 Joint95% Static Plan의 실제 Attainment는 92.96875%였다.
SLO Output은 228.365 tokens/s였다.
미래 정보로 고른 Beam1024 Plan의 실제 실행은 95.3125%와 235.335 tokens/s였다.
이 실행은 사후 Oracle Plan의 GPU Diagnostic이다.
실시간 Online 정책의 성능으로 표시하지 않는다.
실측 Offline Plan이 큰 Online Headroom을 입증한 것도 아니다.

Online H2의 Request12개가 Client Joint SLO에 실패했다.
그중 4개는 TTFT에 실패했다.
다른 8개는 Client ITL에 실패했다.
이 8개 Request의 Worker Token 간격은 기준 이내였다.
Client Interval의 증가는 Delivery Lag 변화로 확인됐다.
이를 특정 Policy 함수의 배타적인 원인이라고 확정하지 않는다.
Client SLO 실패를 Worker 경계로 바꾸어 통과 처리하지 않는다.

이 중 Group을 결합한 Static 진단은 진행 정지로 중단했다.
Source/Destination Queue가 남은 상태에서 양쪽 GPU의 NVML 사용률이 약0%로 지속됐다.
물리적인 Kernel Idle의 직접 측정은 아니다.
실패 로그와 마지막 Queue 상태를 보존했다.
이 Run은 Qualified 성능이 아니다.
Trace에서는 미래 Suffix 예약과 Native Cohort의 Credit 순환 대기를 확인했다.
D의 Free Blocks는 16613개였다.
이 중 미래 예약은 16152개였다.
남은 461개는 Native Prompt 한 개의 501 Blocks보다 작았다.
마지막 Cohort의 7개 중 q30/q31의 Prefix Read는 제출되지 않았다.
전체 Cohort의 Import를 기다리는 Barrier도 완료되지 않았다.
이는 GPU Compute 부족이 아닌 구현의 Credit Ordering 문제와 일치한다.
최종 Per-request Allocation Log는 강제 종료 전에 Flush되지 않았다.
정확한 Callback 순서는 `NOT_IDENTIFIED`다.
예약 상한만 줄인 별도 진단은 성능 기준선으로 사용하지 않는다.
그 진단은 상한을 32에서 16 Requests로 줄였다.
전체128개 Output이 Reference와 일치했다.
P/D의 최종 Free KV Blocks는 모두 초기값19113개로 복원됐다.
남은 Reservation Generation도 없었다.
실제 Cut Fallback이 바뀌므로 기존 실패와 Latency를 직접 비교하지 않는다.
이 실행의 전체 Prompt KV·Hidden Capture는 별도로 수행하지 않았다.

## Matched Control 계약

Held-out Dynamic의 실제 Cut Histogram을 기록한다.
같은 Source Group 크기 안에서 Cut Label을 섞는다.
기록한 FIFO Source Group 크기를 Control에서 정확히 재현한다.
이 Control은 Request 수, 이전 Layer Work와 Cut Histogram을 맞춘다.
실제 Destination Group도 같아야 Matching을 통과한다.

이 Control은 사후의 Aggregate 정보를 사용한다.
따라서 미리 선정한 Deployable Static 정책이 아니다.
같은 양의 작업을 다른 위치에 배치했을 때의 진단 Control이다.
Matching 실패를 Dynamic의 독립적인 정책 이득으로 표시하지 않는다.
Strong Static의 Held-out 결과도 별도로 비교한다.

Output16/QPS6.4의 세 독립 Pair는 실제 Matching을 통과했다.
Source/Destination 구성원과 Work뿐 아니라 Role별 GPU Shape Histogram도 일치했다.
Dynamic/Control의 Raw Output 비율은1.016152였다.
95% CI는[1.007067, 1.025663]이었다.
같은 이전량과 Geometry에서도 약1.615%의 Rate 이점은 관측됐다.
이는 강한 Static을 이겼다는 뜻이 아니다.

TTFT p95 비율은0.951320이었다.
95% CI는[0.921702, 1.001450]이었다.
따라서 이 Control 대비 TTFT 개선은 통계적으로 확정하지 못했다.
ITL p99 비율은1.014996이었다.
95% CI는[1.000390, 1.032196]이었다.
Rate의 작은 이점과 Decode Tail 증가를 함께 보고한다.
사후 Aggregate 정보를 쓰는 Control의 한계도 유지한다.

## 통계와 증거

새 원문 Output16/QPS6.4의 독립3회에서 Dynamic의 Strong Static 우위는 입증되지 않았다.
Dynamic/Static TTFT p95 비율은1.077452였다.
95% CI는[0.995131, 1.200123]이었다.
SLO Output 비율은0.988360이었다.
95% CI는[0.970581, 1.004884]이었다.
Absolute Joint SLO는 세 Start 모두 통과했다.

그러나 Dynamic의 Native-relative Decode5% Gate는 통과하지 못했다.
ITL p99 비율은1.034447이었다.
95% CI는[1.000057, 1.065044]이었다.
점 추정치가5% 이내여도 CI 상한이 기준을 넘는다.
TPOT p95는 통과했지만 ITL 실패를 상쇄하지 않는다.
같은 Histogram Control의 작은 Rate 이점을 종합 GO로 해석하지 않는다.

Output64/QPS4.8의 독립3회는 다른 양상을 보였다.
Dynamic/Static TTFT p95는0.921309배였다.
95% CI는[0.890763, 0.961013]이었다.
Raw Output은1.029768배였다.
95% CI는[1.018701, 1.040291]이었다.
Native-relative Decode5% Gate도 통과했다.
ITL과 TPOT 비율의 CI 상한은 각각1.020104와1.000306이었다.

그러나 Primary SLO Output 비율은1.035728이었다.
95% CI는[0.972472, 1.105970]이었다.
따라서 같은 SLO의 유효 Token Rate 우위는 확정하지 못했다.
Raw Rate와 TTFT의 이점을 모두 없다고 해석하지 않는다.
이전 Work가 더 많아진 효과와 독립 정책 효과도 아직 분리하지 못했다.
Primary SLO 우위 CI가 기준을 통과하지 못해 추가 Output64/QPS4.8 Matched GPU Control은 실행하지 않았다.

Seed2026111741에서 Client ITL 때문에8 Requests가 실패했다.
이 Run의 TTFT와 TPOT 실패는 없었다.
Worker의 모든 Token 간격은31ms 이내였다.
예를 들어 q56의 Client 간격은42.641804ms였다.
Worker 간격은24.019083ms였고 Delivery Lag 변화는18.622721ms였다.
이8개 Request의512 Output Tokens는 원래 Client SLO 집계에서 제외했다.
어떤 Coordinator 함수가 이 지연을 만들었는지는 `NOT_IDENTIFIED`다.
Client 경계를 Worker 경계로 바꾸어 성능을 재평가하지 않았다.

최종 비율은 같은 Arrival Seed의 독립 Server Start를 짝지어 계산한다.
Request나 Token을 독립 Start로 취급하지 않는다.
Native-relative ITL p99와 TPOT p95의 CI 상한이 모두 1.05 이하여야 보호를 통과한다.
Absolute SLO와 Relative 비열등성은 별도 결과다.
N128의 완료 Token Rate는 Sustainable Capacity가 아니다.

- [Held-out 선정 계약과 통계](dense_ar_layer_token_dynamic_headroom_20261117/analysis/heldout_start_statistics.json).
- [실제 Histogram/Geometry Control](dense_ar_layer_token_dynamic_headroom_20261117/analysis/heldout_exact_histogram_geometry_controls.json).
- [실제 GPU Shape까지 확인한 독립3회 통계](dense_ar_layer_token_dynamic_headroom_20261117/analysis/heldout_policy_matched_actual_geometry_statistics.json).
- [예측 오차](dense_ar_layer_token_dynamic_headroom_20261117/analysis/online_prediction_and_deadline_validation.json).
- [후보별 실제 허용·거절](dense_ar_layer_token_dynamic_headroom_20261117/analysis/online_safe_budget_variance.json).
- [긴 Output의 실제 Cohort Geometry](dense_ar_layer_token_dynamic_headroom_20261117/analysis/longer_output_cohort_geometry_diagnosis.json).
- [정책 구현](../dense_ar_layer_token_pd/flow_policy.py).
- [실제 Matching 검사](../dense_ar_layer_token_pd/matched_policy_controls.py).
- [Mixed Cohort의 Credit 정지 분석](dense_ar_layer_token_dynamic_headroom_20261117/analysis/mixed_cohort_credit_stall_trace.json).
- [예약 상한16의 정상 완료 진단](dense_ar_layer_token_dynamic_headroom_20261117/analysis/mixed_cohort_credit16_diagnostic.json).
- [Client·Worker 전달 간격 분리](dense_ar_layer_token_dynamic_headroom_20261117/analysis/worker_delivery_SLO_failure_diagnosis.json).
- [Constrained Static·사후 Oracle Plan의 GPU 진단](dense_ar_layer_token_dynamic_headroom_20261117/analysis/mixed_constrained_static_oracle_GPU_validation.json).

최종 Held-out 수치와 판정은 최종 요약에 기록한다.
Physical Kernel Idle, SM/HBM Counter와 미래 Arrival 보장은 `NOT_MEASURED`다.

## Runtime 로그의 시간 경계

일부 Worker는 종료 중 UCX Context Query Error를 기록했다.
CUDA Driver의 Code4는 Driver 종료 상태를 의미한다.
[NVIDIA의 CUDA12.9.1 정의](https://docs.nvidia.com/cuda/archive/12.9.1/cuda-driver-api/group__CUDA__TYPES.html)를 확인했다.
해당 Query의 Epoch Timestamp와 Coordinator COMPLETE를 직접 비교했다.
이 경고가 측정 Window 뒤에 발생했는지도 별도로 기록한다.
프로세스 전체가 Error-free였다고 주장하지 않는다.
Serving 중의 Launch Failure와 종료 중의 Context Query를 혼동하지 않는다.

JIT 경고도 측정 시작 전인지 검사했다.
예를 들어 Flow8 H3의 JIT 경고는 측정 시작보다 약 7.46초 앞섰다.
따라서 그 경고만으로 H3의 낮은 처리량을 설명하지 않는다.
측정 경계가 불확실한 JIT는 별도 표시한다.
Warmup 없는 State Capture의 시간을 성능으로 사용하지 않는다.
[CUDA/JIT 감사](dense_ar_layer_token_dynamic_headroom_20261117/analysis/CUDA_and_JIT_log_audit.json)에 원자료와 분류를 보존한다.

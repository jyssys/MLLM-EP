# Dense AR Layer PD: 독립 반복의 Decode 보호 결과

## 핵심 판정

**선정된 Static은 Native 대비 TTFT 개선과 Decode 5% 비열등성을 통과했다.**
**COST Dynamic은 TTFT를 더 줄였지만 Decode 비열등성을 입증하지 못했다.**

이 문서는 사전에 고정한 Output16의 8개 Arm을 다룬다.
모든 Arm은 N128 × 5개 독립 Server Start를 완료했다.
고정 50%의 추가 대조군은 탐색적 검증이다.
그 결과와 Output32/64는 [최종 보고서](dense_ar_dynamic_layer_pd_crossover_final_summary.md)에 분리한다.

## 평가 조건

Qwen3-8B의 원본 BF16과 36개 Layer를 유지했다.
Revision은 `b968826d9c46dd6066d109eabc6255188de91218`이다.
GPU0/1, Input8000, Output16, Offered QPS4.777100014491616을 사용했다.
GPU당 KV Budget은 42GiB이다.
Prefix Caching은 껐다.
Greedy Decoding과 Ignore EOS를 사용했다.

Discovery와 겹치지 않는 128개 Prompt ID를 사용했다.
독립 Arrival Seed는 `2026111341`부터 `2026111345`까지이다.
같은 Seed의 모든 Arm은 같은 Poisson Arrival과 Prompt를 사용했다.
5회는 독립 Start이다. 640개의 독립 원문을 뜻하지는 않는다.
Runtime, SLO, 비용 표, Baseline 선정은 Held-out 평가 전에 고정했다.

Primary Latency에는 CUDA Marker와 CUPTI를 사용하지 않았다.
공통 Host Audit과 NVML 관측은 유지했다.
TTFT와 ITL은 Coordinator의 Token Delivery 시각으로 계산했다.
첫 ITL도 포함했다.
Service TTFT는 `NOT_MEASURED`이다.

## 설정 선정

Native는 `lean_native_O16_N64_v1`이다.
Colocated는 검증된 HND Cohort4, Round-robin DP2이다.
Static은 Last12의 Source-group Counter Fraction31.25%이다.
COST는 Frozen Lookup Table과 현재 Host 상태를 사용한다.

IDLE은 `t2−t1`에 해당하는 추정 Budget만 검사한다.
DEADLINE은 예상 Decode Drain과 Deadline Guard를 추가한다.
COST는 Handoff, Suffix, Commit, 예상 Native 완료와의 차이를 추가로 검사한다.
ControllerZero는 같은 Controller를 사용하되 이전하지 않는다.
MatchedFraction은 Discovery COST의 31.25%를 고정한 Request-ID Block 대조군이다.

Static의 Counter Fraction은 요청 비율과 정확히 같지 않을 수 있다.
Request-ID Block 대조군은 요청 비율을 정확히 고정한다.
두 방식 모두 미래 Arrival이나 Held-out 완료시각을 사용하지 않는다.

## 실제 성능

아래 값은 5개 Start의 해당 Percentile 또는 Rate를 평균한 값이다.
Token을 합쳐 계산한 Pooled Percentile은 아니다.
Latency 단위는 ms이다.
Completed QPS와 Token/s는 Finite-cohort 값이다.

| Arm | TTFT p95 | ITL p99 | TPOT p95 | E2E p95 | Completed QPS | Output Token/s |
|---|---:|---:|---:|---:|---:|---:|
| Native | 10011.60 | 31.051 | 26.327 | 10389.46 | 3.382 | 54.118 |
| Colocated DP2 | 2895.45 | 32.194 | 27.615 | 3291.51 | 4.385 | 70.158 |
| Best Static | 7012.23 | 29.888 | 26.007 | 7383.44 | 3.715 | 59.432 |
| COST | 5663.08 | 31.675 | 26.614 | 6053.68 | 3.907 | 62.506 |
| DEADLINE | 5700.98 | 32.326 | 27.154 | 6096.41 | 3.896 | 62.328 |
| IDLE | 5874.46 | 31.213 | 27.092 | 6265.04 | 3.871 | 61.938 |
| ControllerZero | 10163.12 | 33.132 | 27.211 | 10557.53 | 3.361 | 53.774 |
| MatchedFraction | 7082.10 | 31.759 | 26.862 | 7478.51 | 3.698 | 59.170 |

Start별 값과 p50/p99, First/Steady ITL, Maximum ITL은 [기계 판독 결과](dense_ar_dynamic_layer_pd_crossover_20261113/analysis/heldout_paired_statistics.json)에 보존했다.
[전체 Cell CSV](dense_ar_dynamic_layer_pd_crossover_20261113/analysis/all_serving_cells.csv)도 제공한다.
큰 Cohort의 Drain Rate를 지속 가능한 Serving Capacity로 해석하지 않는다.

## Native 대비 비율과 신뢰구간

각 비율은 Start 평균의 비율이다.
독립 Start를 Paired Bootstrap 단위로 사용했다.
5개 Start의 모든 3125개 경험적 재표본을 열거했다.
표는 Percentile 95% 신뢰구간을 제시한다.
요청이나 Token을 독립 Start 대신 재표본하지 않았다.

| Arm / Metric | 비율 | 95% CI | 판정 |
|---|---:|---|---|
| Static TTFT p95 | 0.70041 | [0.66205, 0.73156] | 개선 |
| Static E2E p95 | 0.71067 | [0.67286, 0.74166] | 개선 |
| Static ITL p99 | 0.96256 | [0.91748, 1.00809] | 1.05 비열등성 통과 |
| Static TPOT p95 | 0.98784 | [0.95574, 1.02006] | 1.05 비열등성 통과 |
| COST TTFT p95 | 0.56565 | [0.52067, 0.62496] | 개선 |
| COST E2E p95 | 0.58268 | [0.53838, 0.64215] | 개선 |
| COST ITL p99 | 1.02009 | [0.94364, 1.10087] | 비열등성 미입증 |
| COST TPOT p95 | 1.01090 | [0.97129, 1.05079] | 비열등성 미입증 |

COST의 점 추정치는 두 Decode 지표 모두 1.05보다 낮다.
그러나 신뢰구간 상한은 기준을 넘는다.
이를 Decode 보호 성공으로 보고하지 않는다.
반대로 이 신뢰구간만으로 실제 악화가 반드시 5%를 넘는다고 주장하지도 않는다.
5개 Start는 Production Latency Guarantee를 확정하기에 부족하다.

COST/Static의 TTFT p95 비율은 0.80760이다.
95% CI는 [0.75435, 0.86214]이다.
그러나 ITL p99 비율은 1.05978이다.
그 95% CI는 [1.01806, 1.09903]이다.
추가 TTFT 이득과 Decode Tail 차이를 함께 해석해야 한다.

## SLO Regime별 결과

모든 Threshold는 Dynamic 실행 전에 고정했다.
Output16의 TTFT/ITL/TPOT Threshold는 다음과 같다.

- TTFT-priority: 2000/54/52ms.
- Balanced: 5000/34/33ms.
- Decode-priority: 10000/30/29ms.

Request Decode Pass는 Request-average TPOT와 Request-max ITL을 모두 요구한다.
Joint Pass는 TTFT도 요구한다.
Goodput은 Joint Pass 요청 수를 전체 관측 구간으로 나눈 값이다.

| Regime | Native Joint / Goodput | Colocated Joint / Goodput | Static Joint / Goodput | COST Joint / Goodput |
|---|---|---|---|---|
| TTFT-priority | 7.969% / 0.269 | 71.719% / 3.140 | 10.000% / 0.370 | 10.000% / 0.389 |
| Balanced | 32.656% / 1.107 | 94.844% / 4.158 | 49.688% / 1.840 | 70.781% / 2.761 |
| Decode-priority | 67.031% / 2.273 | 61.875% / 2.706 | 84.688% / 3.145 | 63.281% / 2.474 |

Goodput 단위는 Requests/s이다.
Joint 비율의 분모는 Arm마다 640 Requests이다.
Goodput은 5개 Start의 Rate 평균이다.

Balanced에서는 Colocated가 COST보다 우세하다.
COST/Colocated Goodput 비율은 0.66388이다.
95% CI는 [0.48673, 0.81344]이다.
Static의 Decode-priority Goodput은 Colocated보다 점 추정치가 높다.
그러나 비율 1.16241의 95% CI는 [0.99142, 1.34196]이다.
따라서 Colocated 대비 Goodput 우위는 확정하지 않는다.

## 실제 완료한 이전 계산

| Arm | Last0 | Last2 | Last4 | Last8 | Last12 | 이전 Requests | 완료한 D Prefill Layers |
|---|---:|---:|---:|---:|---:|---:|---:|
| Static | 442 | 0 | 0 | 0 | 198 | 198/640 | 2376/23040 |
| COST | 344 | 0 | 0 | 0 | 296 | 296/640 | 3552/23040 |
| DEADLINE | 60 | 24 | 260 | 272 | 24 | 580/640 | 3552/23040 |
| IDLE | 60 | 20 | 264 | 276 | 20 | 580/640 | 3544/23040 |
| MatchedFraction | 440 | 0 | 0 | 0 | 200 | 200/640 | 2400/23040 |

COST는 실제 현재 상태에 따라 Last0 또는 Last12를 선택했다.
여러 양수 Cut을 선택한 COST의 증거는 이 조건에서 없다.
DEADLINE과 IDLE에서는 실제 Online 실행 중 Last0/2/4/8/12가 모두 선택됐다.
강제 Mixed-cut 정합성 시험과 이 결과를 구분한다.

COST의 요청 이전 비율은 46.25%이다.
이전한 Layer Work 비율은 15.4167%이다.
MatchedFraction의 31.25%는 Discovery에서 고정한 값이다.
Held-out COST의 실제 비율과 완전히 일치하는 대조군은 아니다.
따라서 그 TTFT 차이를 순수한 State-aware 선택의 기여율로 분해하지 않는다.

정합성 시험에서는 실제 Layer 방문 횟수와 KV/Hidden을 직접 검사했다.
Serving의 완료 Layer 수는 완료 요청의 확정 Cut과 검증된 Executor 계약으로 집계했다.
Marker-free Serving의 모든 요청에서 전체 KV/Hidden Tensor를 다시 Hash한 것은 아니다.
단순 Admission 시도를 완료한 이전 계산으로 세지 않았다.

## 실행과 Ownership의 범위

P의 해당 Prefix를 제출하기 전에 Cut을 결정한다.
각 Positive Cut은 D Reservation 이후 Commit한다.
이미 전체 Forward를 제출한 뒤 과거 Layer에서 멈추지 않는다.
P는 Prefix를 한 번 계산한다.
D는 Hidden/Residual과 Partial KV를 받아 Suffix를 한 번 계산한다.
이후 Decode는 Full36으로 진행한다.

D는 기존 User Decode Cohort가 끝난 뒤 다음 Suffix Cohort를 실행한다.
즉시 Kernel Preemption이나 진행 중 Decode와의 Spatial Multiplexing은 구현하지 않았다.
Commit 전에는 Native Fallback이 가능하다.
Commit 후에는 Source를 ACK 전에 해제하지 않는다.
Post-commit D→P Recovery와 GPU Failure Replay는 `NOT_RUN`이다.
정상 종료의 Lifetime Pass를 장애 복구 검증으로 해석하지 않는다.

## P Critical Path와 계측 한계

Native의 Host Source-production Span 평균은 37.251s였다.
Static은 33.895s, COST는 32.027s였다.
COST/Native 비율은 0.85974이다.
95% CI는 [0.85210, 0.86895]이다.

이 Span은 첫 Source 제출부터 마지막 Source Ready까지의 Host 관측이다.
Queue, 준비, 계산, Backpressure를 포함한다.
Exclusive P GPU Compute Time이나 Kernel Busy가 아니다.
P에서 생략한 Layer Work와 Host Span 단축은 확인했다.
Clean Exclusive P Kernel Time 절감량은 `NOT_MEASURED`이다.

별도 CUDA Event A/Events/C 대조군도 실행했다.
Native, Colocated, Static, COST, Last8 모두 전체 5% 계측 Gate를 통과하지 못했다.
그 Event Enclosure는 Diagnostic으로만 사용한다.
Kernel Union, Exclusive CPU, Wire Time, Physical Safe Idle로 바꾸어 해석하지 않는다.

## 해석

이 Runtime에서는 안전한 제한적 Layer 이전의 실익이 존재한다.
그 증거는 Best Static의 독립 반복 결과이다.
더 복잡한 COST 정책의 추가 TTFT 이득도 관측했다.
그러나 그 정책의 Native-relative Decode 보호는 아직 입증하지 못했다.
Strong Colocated 대비 전체 Serving 승리도 확인하지 못했다.
이는 D의 물리적 계산 여유가 없다는 구조적 NO-GO를 뜻하지 않는다.

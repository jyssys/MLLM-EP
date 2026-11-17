# Dense AR Layer-vs-Token Online 비교

## 비교 계약

이 보고서는 같은 이전 작업량 Control과 Best-vs-Best를 구분한다.
한 가지 Token Suffix를 모든 부하에 강제하지 않는다.
검증한 Token Arm은 Completed-prefix KV를 넘기는 방식이다.
Layer별 KV Streaming과 Token Pipeline은 미실행이다.
Token Suffix, Source Group과 Destination Batch를 각각 탐색한다.
Layer의 Last2/4/8/12와 Group Geometry도 탐색한다.
선정은 기존 Discovery Prompt에서 수행한다.
새 Prompt의 Candidate Latency를 보기 전에 설정을 동결한다.

Qwen/Qwen3-8B Revision은 `b968826d9c46dd6066d109eabc6255188de91218`다.
Original BF16, Exact Causal Attention과 Full36 Decode를 유지한다.
GPU0/1의 동일한 H100 SXM 80GB 두 대를 사용한다.
GPU별 KV Budget은 42GiB다.
Prefix Caching은 비활성화한다.
Input은 8000 Tokens다.
Output16과 Output64는 `ignore_eos=True`로 고정한다.

Mean TBT는 전달된 연속 Token 간격의 평균이다.
TPOT는 Request별 `(마지막 Token−첫 Token)/(Output Tokens−1)`이다.
First ITL을 포함한다.
Worker Emission과 Coordinator Delivery도 별도로 기록한다.
TTFT·ITL·TPOT의 Request별 Frozen SLO를 모두 통과해야 SLO Token에 포함한다.
Finite-cohort Drain Rate를 Sustainable Capacity로 표시하지 않는다.

## P-side Chunked PD 별도 Pilot

별도 P-side Chunked PD도 구현했다.
P에서만 Chunk512로 전체 Prefill을 수행한다.
이 방법은 Token-axis P→D 이전과 다르다.
N32 전체 State Gate를 통과했다.
아래 Pilot은 N64/Output16/QPS4.8/Seed2026111710의 각1회 결과다.

| 방법 | TTFT p95, ms | ITL p99, ms | Raw Output tokens/s |
|---|---:|---:|---:|
| Native | 5120.909 | 24.563 | 55.092 |
| P-side Chunk512 PD | 6445.624 | 24.231 | 51.085 |

다른 P-side Chunk Size의 광범위한 튜닝과 새 N128 독립3회는 `NOT_RUN`이다.
이 한 Pilot을 가장 강한 모든 P-side Chunked PD의 성능으로 표시하지 않는다.

## 같은 작업량 Control의 경계

Layer Last12와 Token2656/2672의 N144 Plan은 GEMM Token-layer Rows를 정확히 맞춘다.
두 방식의 논리적인 Prefix KV Bytes도 같다.
Source와 Destination Group의 실제 구성원도 검사한다.
Layer에는 Request별 131,072,000 Bytes의 Hidden/Residual이 추가된다.
두 방식의 Attention Pair 수와 실제 P Compute Saving까지 같지는 않다.
Token 경로는 P Prefix의 Sample을 생성한 뒤 버린다.
이는 기존 Engine의 실제 실행 비용이다.
Layer 경로도 P의 사용하지 않는 Source Sample 경로를 유지한다.
이를 Token만의 고유한 추가 비용이라고 주장하지 않는다.
이 공통 LM-head/Sampling을 생략한 최적 구현까지 평가한 것은 아니다.
Token-layer Rows의 일치는 모든 Operator의 작업량 일치가 아니다.
Source의 KV Credit 점유도 같지 않다.
현재 Layer 경로는 Full36 Cache Group의 500 Blocks를 할당한다.
Last12와 같은 Token 작업량은 Prefix333/334 Blocks만 사용한다.
Layer의 미계산 상위 Layer는 유효 KV가 아니다.
그러나 같은 Cache Group의 Block Credit을 사용한다.
이 차이는 Layer에 불리할 수 있다.
같은 유효 KV Byte를 같은 Cache Capacity라고 해석하지 않는다.

Last12 Control에서 Token의 D Attention Pair 수는 Layer의 1.666580677배다.
이는 `STRUCTURAL_ACCOUNTING`이다.
GPU 시간 또는 Online 성능 비율이 아니다.
늦은 Token Query는 더 긴 Causal Prefix에 Attend한다.
전체 시스템의 Attention을 추가하거나 누락하는 것은 아니다.
P와 D 사이의 Attention Work 분배가 다르다.

## 초기 같은 작업량 결과 — MEASURED, 각 n=1

아래 값은 독립 반복을 완료하기 전의 Discovery Control이다.
새 Prompt Holdout이 아니다.
QPS6.4와 N144를 사용했다.

| Control | TTFT p95, ms | ITL p99, ms | Raw Output tokens/s |
|---|---:|---:|---:|
| Last8 Layer | 11308.824 | 24.706 | 68.160 |
| Last8 Token1776/1792 | 10096.451 | 24.563 | 70.793 |
| Last12 Layer | 6957.056 | 24.492 | 78.840 |
| Last12 Token2656/2672 | 8584.514 | 25.021 | 74.438 |

Last8에서는 Token이 더 빨랐다.
Last12에서는 Layer가 더 빨랐다.
따라서 이 결과는 Layer의 보편적인 우위를 지지하지 않는다.
별도의 독립 Arrival Seed 3개로 Last12 Control을 재검사한다.
같은64개 Discovery 원문을 선언된 주기로 사용한다.
이 반복을 새 원문 Holdout으로 표시하지 않는다.

## 독립 Arrival 3회: 같은 Last12 작업량 — MEASURED

Seeds는 2026111760/2026111761/2026111762다.
각 Start는 N144다.
같은64개 Discovery 원문을 선언된 주기로 사용했다.
실제 P/D GEMM Rows, KV Bytes와 Source/Destination Group 검사를 모두 통과했다.
이전 Hidden/Residual Byte와 Source KV Credit은 같지 않다.

| 방법 | TTFT p95 평균, ms | Raw Output tokens/s | SLO Output tokens/s | Start별 Joint Attainment |
|---|---:|---:|---:|---|
| Native | 18959.329 | 55.360 | 28.068 | 50.0000/51.3889/50.6944% |
| Layer Last12 | 8037.276 | 76.460 | 76.460 | 100/100/100% |
| Token2656/2672 | 9339.917 | 73.185 | 70.760 | 100/100/90.2778% |

아래 비율은 Layer/Token이다.
CI는 같은 Seed의 독립 Server Start를 짝지은 95% Bootstrap CI다.

| Metric | 비율 | 95% CI |
|---|---:|---:|
| TTFT p95 | 0.860530 | [0.791981, 0.927848] |
| ITL p99 | 1.009008 | [0.999586, 1.019340] |
| TPOT p95 | 1.001546 | [1.000400, 1.002979] |
| Raw Output tokens/s | 1.044749 | [1.023927, 1.064201] |
| SLO Output tokens/s | 1.080545 | [1.026571, 1.144348] |

TTFT p95는 13.947% 감소했다.
Raw Output은 4.475% 증가했다.
SLO Output은 8.054% 증가했다.
Native 대비 Layer의 Decode 5% CI Gate도 통과했다.
그 CI 상한은 ITL p99 1.021452, TPOT p95 1.026753이었다.
Layer는 세 Start 모두 Frozen Decode-priority Joint SLO를 통과했다.

이는 같은 작업량 Control에서 Layer의 조건부 Serving 이점을 지지한다.
Best-vs-Best Token 튜닝의 최종 결론은 아니다.
Raw Token Rate의 개선이 5% 이상이라고 CI로 입증한 것도 아니다.
Queue가 증가하는 Finite Cohort이므로 Sustainable Capacity 우위로 해석하지 않는다.

## 새 원문 Best-vs-Best, Output16 — MEASURED

기존 원문과 겹치지 않는128개 Prompt를 사용했다.
Seeds는2026111740/1741/1742다.
Output16과 Offered QPS6.4를 유지했다.
Discovery에서 Token2400/D4와 Static Last12/P4/D8을 각각 선정했다.
Candidate의 새 Prompt Latency를 보기 전에 설정을 동결했다.

| Metric | Static Layer/Token | 95% CI |
|---|---:|---|
| TTFT p95 | 0.906937 | [0.859896, 0.941932] |
| ITL p99 | 0.989010 | [0.969527, 1.009073] |
| Raw Output tokens/s | 1.024649 | [1.015425, 1.036093] |

TTFT p95는9.306% 감소했다.
Raw Output은2.465% 증가했다.
Layer의 Start별 Joint Attainment는100/100/100%였다.
Token은100/100/99.21875%였다.
Token의 마지막 Start에서는 Client ITL 때문에 한 Request가 실패했다.
Layer의 Native-relative Decode5% Gate도 통과했다.
ITL p99 비율 CI 상한은1.040000이었다.
TPOT p95의 해당 CI도 별도 통계 파일에 보존한다.

이는 새 원문의 동결 후보 비교에서도 조건부 Layer 이점을 지지한다.
같은 작업량 Control과 달리 두 방법의 이전 Rows와 Geometry는 독립 최적화했다.
한 번의 Discovery Safety Screen이 전역 최적 설정을 보장하지 않는다.
Streaming Token Pipeline과 공통 Source LM-head 제거 경로는 평가하지 않았다.
Output64와 다른 QPS의 결과는 별도 Table로 구분한다.

## 새 원문 Best-vs-Best, Output64 — MEASURED

Input8000/Output64/QPS4.8에서도 같은 새 Prompt와 세 Seed를 사용했다.
선정된 Token은 Suffix1024/P4/D8이었다.
Static Layer는 Last4/Source Macro8/D8이었다.
P slots는4다.

Static Layer/Token의 TTFT p95 비율은1.209814였다.
95% CI는[1.152767, 1.248021]이었다.
Raw Output 비율은0.957135였다.
95% CI는[0.947520, 0.967537]이었다.
이 조건에서는 Token이 더 빨랐다.
Output16의 Layer 이점을 Output64로 일반화하지 않는다.
같은 축이 모든 Decode Residency에서 최선이라는 결론도 내리지 않는다.

## GEMM과 Attention의 원인 분리

실제 Original BF16 Layer0 Projection Weight로 Primitive를 실행했다.
30회 Timing과 5회 Warmup을 Shape별로 수행했다.
실제 Full Forward나 Serving Latency를 측정한 것은 아니다.
CUDA Event는 Operator Enclosure다.
Norm, Attention, KV, Transfer와 Online 경쟁을 포함하지 않는다.

Last12의 같은 GEMM Rows에서 GPU1의 D Projection 합산 추정은 다음과 같다.
Layer는 206.02829ms다.
Token은 212.32665ms다.
Token/Layer는 1.030570배다.
GPU0에서는 1.047918배였다.
이 합계는 Layer0의 중앙값을 반복한 구조적 추정이다.
36개 실제 Layer의 Exclusive Kernel Time이 아니다.

이 Shape에서 GEMM 효율 차이는 존재했다.
그러나 GEMM만으로 전체 Serving 차이를 설명했다고 주장하지 않는다.
GEMM Kernel을 개선하면 이 차이 일부가 줄어들 수 있다.
Layer별 Token Pipeline도 Queue와 Handoff의 Critical Path를 바꿀 수 있다.
그 미실행 방법까지 이겼다고 주장하지 않는다.
늦은 Token Query의 Causal Attention Work 분배는 GEMM Kernel 변경만으로 사라지지 않는다.
그 사실만으로 Layer의 End-to-End 우위가 보장되지는 않는다.

## 타임라인과 계측 간섭

같은 Trace의 비계측 실행 2개와 Event Capture 1개를 비교했다.
이는 점 추정치의 간섭 Screen이며 통계적인5% 상한 보장은 아니다.
Layer Capture의 네 Metric 변화는 모두 5% 이내였다.
Token Capture의 TTFT p95 비율은 1.081923배였다.
따라서 Token Capture는 5% 간섭 기준을 통과하지 못했다.
Token Event 수치는 진단용이다.

Layer Capture의 P Prefix Enclosure p50은 733.414ms였다.
D Suffix Enclosure p50은 358.472ms였다.
D Decode Step Enclosure p50은 22.368ms였다.
이 값은 Stream Gap을 포함할 수 있다.
이를 Exclusive Kernel Time이나 물리적 GPU Busy로 해석하지 않는다.
Token의 대응 Enclosure는 각각 710.063/395.003/22.361ms였다.
Token 수치는 간섭 기준 실패의 제한을 갖는다.
두 실행의 시간축을 합쳐 하나의 실제 타임라인으로 만들지 않는다.

## 근거

- [정합성 계약](dense_ar_layer_vs_token_correctness_and_execution.md).
- [정확한 작업량 계산](dense_ar_layer_token_dynamic_headroom_20261117/analysis/matched_GEMM_rows_structural_accounting.json).
- [독립 Arrival 비교와 실제 Geometry 검사](dense_ar_layer_token_dynamic_headroom_20261117/analysis/matched_Last12_independent_start_statistics.json).
- [Held-out 독립 Start 통계](dense_ar_layer_token_dynamic_headroom_20261117/analysis/heldout_start_statistics.json).
- [고립 GEMM 비용](dense_ar_layer_token_dynamic_headroom_20261117/analysis/isolated_GEMM_matched_rows_Last12.json).
- [계측 간섭 검사](dense_ar_layer_token_dynamic_headroom_20261117/analysis/CUDA_event_interference_screen.json).
- [Host·Stream 타임라인 원자료](dense_ar_layer_token_dynamic_headroom_20261117/analysis/host_stream_timeline_data.json).

Best-vs-Best 최종 수치와 CI는 최종 요약에 따로 기록한다.
독립 Start가 하나인 점에는 CI를 만들지 않는다.
95% CI는 독립 Server Start를 Arrival Seed별로 짝지어 계산한다.
Request나 Token을 독립 Start처럼 재표본하지 않는다.

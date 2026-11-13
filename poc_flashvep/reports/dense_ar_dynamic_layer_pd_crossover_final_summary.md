# Dense AR Dynamic Layer PD: Crossover와 Decode 보호 최종 결과

## 1. 결론

**최종 판정은 `CONDITIONAL`이다. Dynamic의 GO 조건은 충족하지 못했다.**

실제 Request별 P→D Layer Continuation을 구현했다.
Output16의 독립 N128×5에서 COST Dynamic의 TTFT p95는 Native보다 43.43% 낮았다.
그러나 ITL p99와 TPOT p95의 95% CI 상한이 Native의 1.05배를 넘었다.
따라서 Decode 5% 비열등성을 입증하지 못했다.

반면 선정된 **Static Last12, Counter Fraction31.25%는 TTFT p95를 29.96% 줄이며 Decode 비열등성을 통과했다.**
이는 제한적 Layer 이전의 Serving 순이익이 존재한다는 증거이다.
Dynamic이 그 안전한 기준선을 더 잘 대체한다는 증거는 아직 부족하다.
Strong Colocated 대비 전체 Serving 승리도 확인하지 못했다.

Output64, QPS6.015607425656108에서는 PD/Colocated의 성능 Trade-off를 독립 3회 재현했다.
Colocated의 TTFT는 낮았지만 ITL과 TPOT는 높았다.
다만 적격 Colocated는 Cohort 방식이다.
그 차이를 Prefill–Decode Kernel Overlap의 인과적 증거로 보지 않는다.
Clean P Kernel Saturation과 물리적 Safe Slack도 인정 기준을 통과하지 못했다.

## 2. 단계별 판정

| 단계 | 판정 | 근거 |
|---|---|---|
| A: 성능 Trade-off 탐색 | `PARTIAL` | Output64에서 재현. Clean P Busy와 인과적 Prefill–Decode Overlap은 미확정 |
| A: 전체 Crossover Gate | `NO_CONFIRMED_PD_COLOCATED_CROSSOVER_ON_TESTED_GRID` | 물리 계측과 Continuous Baseline의 정합성 제약 |
| B: SLO 고정 | `PASS` | 독립 Low-load Calibration 15개 Start 후 Dynamic 평가 전에 고정 |
| C: Dynamic 실행·정합성 | `PASS` | 실제 Last0/2/4/8/12 실행, KV/Hidden, Layer 방문, ACK/Lifetime 검증 |
| D: COST의 Decode-safe 추가 이득 | `CONDITIONAL` | TTFT 개선. Native-relative Decode CI 미통과 |
| 제한적 Static 이전 | `PASS_ON_TESTED_ENVELOPE` | Output16의 TTFT/E2E 개선과 두 Decode CI 통과 |
| Strong Colocated 대비 시스템 우위 | `NOT_ESTABLISHED` | Balanced Goodput은 Colocated가 우세 |

이 결과를 구조적인 D 자원 부족이나 Layer Offloading의 보편적 불가능성으로 해석하지 않는다.
적격 Continuous Colocated와 더 안정적인 Tail 검증이 남아 있다.

성능표의 증거 수준은 `MEASURED`이다.
정합성 시험은 `CORRECTNESS_ONLY`, Host 재구성은 `TRACE_ANALYSIS`로 구분한다.
Roofline과 이론적 Payload는 `STRUCTURAL_ACCOUNTING`이다.
진단용 Kernel/Event 수치를 Clean Serving 성능에 합산하지 않았다.

## 3. 실험과 재현성

- Model: Qwen3-8B, Revision `b968826d9c46dd6066d109eabc6255188de91218`.
- 원본 BF16, Exact Attention, Full36 Layers, Greedy Decoding을 유지했다.
- Runtime: vLLM0.26.0+cu129, PyTorch2.11.0+cu129, NIXL1.4.1.
- Primary GPU는 H100 80GB의 GPU0/1이다. 각 Arm은 총 2GPU를 사용한다.
- KV Budget은 GPU당 42GiB이다. Prefix Caching은 사용하지 않았다.
- Input8000과 Output16/32/64/128/200을 탐색했다.
- 모델의 실제 Disabled-thinking Generation Suffix를 Tokenizer로 확인했다.
- ShareGPT/LongBench의 실제 문서 Prefix와 원문 ID를 보존했다. 반복 Filler Token은 사용하지 않았다.
- Discovery64와 Held-out128의 Prompt ID 및 Token Prefix는 분리했다.
- Poisson Arrival은 같은 Seed의 Arm 사이에서 동일하다.
- Output은 Ignore EOS로 고정했다. Natural EOS 일반화는 `NOT_RUN`이다.

Primary Output16은 QPS4.777100014491616, N128×5이다.
Arrival Seed는 `2026111341`부터 `2026111345`까지이다.
Output32/64는 QPS6.015607425656108, N128×3이다.
이때 Seed는 `2026111341/42/43`이다.
각 Start의 Arm 순서를 회전했다.
같은 128개 Held-out Prompt를 반복 사용했다. 640개의 독립 원문은 아니다.

총 336개 Runtime Job을 보존했다.
334개는 Runtime을 완료했고, 2개는 Runtime에서 실패했다.
완료 Job 중 25개는 별도의 Output 정합성 Gate에 실패했다.
Runtime 완료를 적격 Serving 성공으로 세지 않았다.
Primary Held-out은 58개 Start이다. 추가 탐색적 대조군은 5개 Start이다.

Primary는 CUDA Marker/CUPTI 없이 측정했다.
공통 Host Audit과 NVML 관측은 유지했다.
동시에 GPU Timing 실험을 실행하지 않았다.
공유 Quiet-lane Lease와 승인된 Resource Guard를 사용했다.
각 Job에 Startup/Serving/Process-group Watchdog을 적용했다.

TTFT/ITL은 Coordinator Token Delivery 기준이다.
TPOT는 첫 Token부터 마지막 Token까지를 `Output−1`로 나눈 값이다.
First ITL을 포함했다. First/Steady ITL도 별도로 저장했다.
Service TTFT는 `NOT_MEASURED`이다.
Offered QPS, 유한 Trace의 Arrival Density, 관측 Arrival QPS, Compute Submission QPS를 분리했다.
모든 Rate와 Goodput은 Finite-cohort 관측이다. Sustainable Capacity가 아니다.

## 4. Crossover 탐색과 Baseline 강도

독립 P-only Capacity는 Batch4/8/16에서 3.5978/3.5940/3.5857 Requests/s였다.
각 값은 같은 Start 안의 2개 Cohort 반복으로 측정했다.
QPS2.477/4.777/6.016 부근을 Output16/32/64/128/200에서 탐색했다.
큰 P Batch가 Capacity를 높인다는 증거는 없었다.
P8/P16의 실제 Serving도 검사했다. 이를 Native에 유리한 설정으로 선정하지 않았다.

Colocated는 Cohort4/8/16, Round-robin/Load-aware Routing, HND/NHD를 비교했다.
Chunk128/256/512/1024/2048과 Continuous 실행도 검사했다.
HND Chunk512/1024/2048에서는 각각 8/64, 6/64, 3/64 Requests의 Output이 달랐다.
Compile3의 Native N8과 Colocated N8도 Output Gate에 실패했다.
실패 원문의 로그와 설정을 보존했다.
정확한 Kernel-level 불일치 원인은 `NOT_IDENTIFIED`이다.

최종 DP2는 **가장 강한 적격 Cohort Baseline**이다.
최적 Continuous Colocated를 이겼다는 뜻은 아니다.
Colocated Tuning에 Native-relative Decode 5% 조건을 강요하지 않았다.
Frozen Balanced Joint Goodput과 TTFT로 독립 선정했다.
Native도 No-migration Controller와 D Cohort 크기를 포함해 선정했다.

### 독립 반복의 성능 Trade-off

아래 값은 Start별 Percentile의 평균이다. 단위는 ms이다.

| Workload | Arm | TTFT p95 | ITL p99 | TPOT p95 | Completed QPS | Output Token/s |
|---|---|---:|---:|---:|---:|---:|
| 8000/16, QPS4.777, 5 Starts | Native | 10011.60 | 31.051 | 26.327 | 3.382 | 54.118 |
| 동일 | Colocated | 2895.45 | 32.194 | 27.615 | 4.385 | 70.158 |
| 8000/32, QPS6.016, 3 Starts | Native | 14783.42 | 32.091 | 27.220 | 3.317 | 106.144 |
| 동일 | Colocated | 10441.76 | 31.572 | 27.309 | 3.847 | 123.119 |
| 8000/64, QPS6.016, 3 Starts | Native | 16457.58 | 33.122 | 27.517 | 3.195 | 204.453 |
| 동일 | Colocated | 10054.25 | 44.750 | 30.986 | 4.188 | 268.056 |

Output64의 Colocated/Native TTFT 비율은 0.61092, 95% CI는 [0.56468, 0.67096]이다.
ITL 비율은 1.35108, CI는 [1.18076, 1.58751]이다.
TPOT 비율은 1.12610, CI는 [1.08770, 1.18210]이다.
이는 측정한 Baseline 사이의 성능 Trade-off이다.

Output64 Seed41에서 Host-visible Decode 동시성의 최댓값은 Native8, DP2의 각 Replica15였다.
Decode가 존재할 때의 시간가중 평균은 Native6.829, DP2의 두 Replica9.087/11.189였다.
이는 User의 첫 Token부터 마지막 Token까지의 Host 관측이다.
실제 GEMM Batch나 Kernel Resource Utilization이 아니다.
같은 Replica의 Prompt Host Step과 기존 User Decode가 겹친 시간은 0s였다.
따라서 증가한 Batch/Workload Composition과 직접 Prefill Kernel 간섭을 구분해야 한다.
각 요인의 인과적 기여율은 `NOT_IDENTIFIED`이다.

[전체 Discovery 표와 제약](dense_ar_pd_colocated_qps_crossover_analysis.md)에 15개 Cell을 제시했다.
Output128/200은 Discovery만 수행했다. Held-out Dynamic 비교는 `NOT_RUN`이다.

## 5. P 병목과 D Headroom의 증거 수준

Output16 Native의 P Queue는 5개 Start 모두 증가했다.
Scheduler 관측 표본의 전반부/후반부 평균은 각각 8.80→27.50, 6.38→19.08, 15.92→26.25, 3.86→20.67, 4.20→22.75였다.
Remote KV 대기를 제외한 최대 Waiting은 32/22/34/27/32였다.
P의 KV Allocation Refusal은 모두 0이었다.
P의 최대 KV 사용률은 각 Start에서 31.39%였다.
Host Source-ready Rate 평균은 3.437 Requests/s였다.
독립 P Capacity3.598의 약 95.5%이다.

이는 실제 Queue와 Service Pressure의 증거이다.
Clean P Kernel Busy≥80%의 직접 증거는 아니다.

Native Output32의 4초/0.75초 CUPTI Capture는 Drop/Invalid0이었다.
그러나 둘 다 5% Profiler/A–A Variation Gate를 통과하지 못했다.
Diagnostic D Kernel Inactivity는 각각 46.03%/63.27%였다.
이 수치를 Clean Idle이나 Decode-safe Budget으로 인정하지 않았다.
별도 CUDA Event 대조군 5종도 전체 계측 Gate를 통과하지 못했다.

따라서 Clean Physical Idle, Spatial Compute Slack, Request-feasible Safe Capacity는 `NOT_MEASURED`이다.
그 값을 0으로 대체하지 않았다.
추가 계산의 실현 가능성은 실제 완료한 Layer Migration으로 별도 확인했다.
그 성공을 Clean Physical Idle 분포의 대체 증거로 사용하지 않았다.

## 6. SLO와 신뢰구간

Dynamic GPU 평가 전에 Low-load Native N16×3을 Output Cap마다 측정했다.
총 15개 독립 Start를 사용했다.
Research SLO는 TTFT-priority/Balanced/Decode-priority의 세 Regime이다.
Output16의 TTFT/ITL/TPOT는 2000/54/52, 5000/34/33, 10000/30/29ms이다.
Production 서비스 표준이라고 주장하지 않는다.

Frozen SLO SHA256은 다음과 같다.

`c8aa94213e4f4ce54821d01ba985a8e9e88a20566a3e8d1f3cc8060d1b3926d9`

Absolute SLO와 Native-relative 5% 기준을 분리했다.
독립 Start를 Paired Bootstrap 단위로 사용했다.
5회는 3125개, 3회는 27개의 경험적 재표본을 열거했다.
Token이나 Request를 독립 Start 대신 사용하지 않았다.
표의 CI는 양측 Percentile 95% CI이다.
작은 Start 수의 경험적 CI가 Production Guarantee를 뜻하지는 않는다.

[SLO 정의와 모든 Threshold](dense_ar_slo_calibration_and_goodput_analysis.md)를 별도로 제공한다.

## 7. 실제 Dynamic 구현

[새 구현](../dense_ar_dynamic_layer_pd/)은 기존 Exact Partial Forward와 NIXL 경로를 재사용한다.
기존 구현과 외부 Runtime 파일은 수정하지 않았다.

1. Source Group을 만들기 전에 현재 P Backlog와 D 상태를 읽는다.
2. Last0/2/4/8/12의 Prefix, Handoff, D Suffix, 완료비용을 비교한다.
3. D가 Request Generation, Cut, KV Credit을 예약한다.
4. Cut을 Commit한 뒤 P Prefix를 제출한다.
5. Partial KV와 Hidden/Residual을 전달한다.
6. D가 Suffix를 완료한 뒤 Full36 Decode를 수행한다.
7. 실제 ACK와 CUDA 소비 완료에 맞춰 Source와 Destination State를 해제한다.

이미 제출한 Full Forward를 과거 경계에서 중단하지 않는다.
같은 Source Group은 같은 Cut을 사용한다.
Group 사이에서는 Request별 Cut이 달라질 수 있다.
Commit 전 거절은 Native 경로를 사용한다.
Commit 후 Prefix를 버리거나 Layer를 재실행하지 않는다.

D는 이전 User Decode Cohort의 종료를 확인한 뒤 새 Suffix Cohort를 실행한다.
이는 실제 실행 Guard이다.
즉시 Kernel Preemption이나 Decode와의 Spatial Multiplexing은 아니다.
동일 Cut의 FIFO Cohort만 구성했다. Cut Bucket 재정렬은 구현하지 않았다.

Last2/4/8/12의 N8 State 검증과 N32 Mixed-cut 검증을 통과했다.
D8/D16의 더 큰 Batch도 별도 검증했다.
D16은 독립 Full Native 참조와 모든 36개 Valid KV Plane 및 최종 Hidden의 SHA256이 일치했다.
실제 Layer 방문, Full36 Decode Graph, Output, 정상 ACK/Lifetime을 각각 검사했다.
Serving 전체의 KV/Hidden을 다시 Hash한 것은 아니다.
완료 Layer 수는 완료 요청의 확정 Cut과 검증된 Executor로 집계했다.

## 8. Output16의 Primary 결과

모든 값은 N128×5의 Start 평균이다. Latency 단위는 ms이다.

| Arm | TTFT p95 | E2E p95 | ITL p99 | TPOT p95 | Balanced Goodput, req/s |
|---|---:|---:|---:|---:|---:|
| Native | 10011.60 | 10389.46 | 31.051 | 26.327 | 1.107 |
| Strong Colocated | 2895.45 | 3291.51 | 32.194 | 27.615 | 4.158 |
| Best Static | 7012.23 | 7383.44 | 29.888 | 26.007 | 1.840 |
| COST Dynamic | 5663.08 | 6053.68 | 31.675 | 26.614 | 2.761 |
| Deadline-aware | 5700.98 | 6096.41 | 32.326 | 27.154 | 2.700 |
| Idle-only | 5874.46 | 6265.04 | 31.213 | 27.092 | 2.552 |
| Controller Zero | 10163.12 | 10557.53 | 33.132 | 27.211 | 1.060 |
| Fixed ID Fraction31.25% | 7082.10 | 7478.51 | 31.759 | 26.862 | 1.774 |

Best Static은 Discovery에서 선정한 Last12, Counter Fraction31.25%이다.

| Native 대비 | 점 추정치 | 95% CI | 판정 |
|---|---:|---|---|
| Static TTFT p95 | 0.70041 | [0.66205, 0.73156] | 개선 |
| Static ITL p99 | 0.96256 | [0.91748, 1.00809] | 비열등성 통과 |
| Static TPOT p95 | 0.98784 | [0.95574, 1.02006] | 비열등성 통과 |
| COST TTFT p95 | 0.56565 | [0.52067, 0.62496] | 개선 |
| COST ITL p99 | 1.02009 | [0.94364, 1.10087] | 비열등성 미입증 |
| COST TPOT p95 | 1.01090 | [0.97129, 1.05079] | 비열등성 미입증 |

COST의 점 추정치가 1.05 미만이어도 GO로 선언하지 않았다.
추가 Start 없이 미입증을 통과로 바꾸지 않았다.
반대로 실제 악화가 반드시 5%를 넘는다는 통계적 결론도 아니다.

COST는 296/640 Requests와 3552/23040 Layers를 실제 완료했다.
요청 이전율은 46.25%, Layer Work 이전율은 15.4167%이다.
선택은 Last0/Last12였다. 여러 양수 Cut을 사용한 COST의 증거는 없다.
DEADLINE/IDLE은 실제 Online 상태로 Last0/2/4/8/12를 모두 선택했다.
이 결과를 강제 Mixed-cut 정합성 시험과 구분했다.

COST의 Host Source-production Span은 Native37.251s에서 32.027s로 줄었다.
비율은 0.85974, CI는 [0.85210, 0.86895]이다.
이는 Queue/준비/Backpressure를 포함한 Host Span이다.
Exclusive P GPU Compute Saving은 `NOT_MEASURED`이다.

Balanced에서 COST/Colocated Goodput은 0.66388, CI는 [0.48673, 0.81344]이다.
Static의 Decode-priority Goodput은 3.145req/s, Colocated는 2.706req/s였다.
그러나 그 비율의 CI는 [0.99142, 1.34196]이므로 우위를 확정하지 않는다.
[전체 SLO·Latency·이전 분포](dense_ar_dynamic_layer_decode_safe_serving_results.md)를 별도로 제공한다.

## 9. 단순 이전 비율 효과와 Dynamic 효과

첫 Held-out 결과를 확인한 뒤 고정 50% ID Mask를 추가했다.
이는 명시적인 탐색적 대조군이다. Primary 선정이나 SLO는 바꾸지 않았다.
같은 5개 Arrival Seed로 N128을 다시 실행했다.

고정50%는 320/640 Requests를 Last12로 이전했다.
TTFT/ITL/TPOT는 5532.17/33.140/27.507ms였다.
COST/고정50%의 TTFT 비율은 1.02366, CI는 [0.98428, 1.06238]이다.
TTFT 우위는 확인하지 못했다.
ITL 비율은 0.95578, CI는 [0.91371, 0.99686]이다.
고정50%보다 COST의 ITL이 낮은 탐색적 증거는 있다.
그러나 두 방법 모두 Native-relative Decode CI를 통과하지 못했다.

따라서 큰 TTFT 이득 중 상당 부분은 더 많은 Layer 이전으로 설명될 수 있다.
State-aware 선택의 독립적 인과 기여율은 `NOT_IDENTIFIED`이다.
비슷한 TTFT에서 Tail을 줄일 가능성은 남지만 확정 GO가 아니다.

COST/IDLE의 TTFT 비율은 0.96402, CI는 [0.92598, 1.00464]이다.
COST/DEADLINE은 0.99335, CI는 [0.93168, 1.05252]이다.
따라서 추가 Cost Model의 TTFT 우위도 확정하지 않는다.
[Paired Ablation 원문](dense_ar_dynamic_layer_pd_crossover_20261113/analysis/primary_policy_ablation_pairs.json)을 제공한다.

## 10. Output32의 Fallback과 일반화 한계

Output32에서 COST는 384/384 Requests를 Native로 실행했다.
실제 Layer 이전은 0이었다.
TTFT p95는 Native14783.42ms, COST15168.10ms였다.
비율은 1.02602, CI는 [1.00196, 1.05530]이다.
이 조건에서는 개선이 없었다.

COST의 Decode가 더 낮게 측정됐어도 이를 Layer 이전의 이득으로 볼 수 없다.
모두 Last0이었다.
Validation에서 고른 Static Last2도 TTFT 개선을 반복 확인하지 못했다.
Output64의 Held-out Dynamic/Static은 `NOT_RUN`이다.
Output128/200, Bursty Arrival, Natural EOS, 다른 Input 길이의 Held-out 검증도 `NOT_RUN`이다.
검사하지 않은 Workload 전체로 결론을 확장하지 않는다.

## 11. 비용 모델과 실제 대기

Roofline은 구조적 FLOPs/Traffic 분석에 사용했다.
Online 판단은 Frozen Lookup Table을 사용했다.
측정 HBM Traffic이나 Tensor Core Utilization은 `NOT_MEASURED`이다.

Held-out COST의 양수 Cut 결정은 74개였다.
Source-ready 예측 MAE는 37.439ms, 절대오차 p95는 50.742ms였다.
First-token 완료 예측 MAE는 49.859ms, 절대오차 p95는 101.972ms였다.
완료시각 과소예측 비율은 86.486%였다.
5ms Guard는 이 오차의 검증된 상한이 아니다.

Known active/pending Drain의 167개 관측에서 가중 MAE는 61.209ms였다.
물리적 Idle 예측 정확도는 `NOT_IDENTIFIED`이다.
현재 알려진 Decode 종료 예측과 미래 Safe Slack은 같은 값이 아니다.

5개 COST Start의 Commit→첫 Token 최대는 각각 1346.8/1373.2/1388.2/1384.3/1371.8ms였다.
Commit 이후 P Prefix 계산도 포함한 시간이다.
Source-ready→D Host Admission 최대는 7.53/6.16/6.37/6.55/33.31ms였다.
이 실행에서 수십 초의 Continuation Starvation은 관측하지 않았다.
그렇다고 장애 상황의 Bounded Completion을 증명한 것은 아니다.

Native KV Payload는 요청당 1125MiB이다.
Hidden과 Residual은 합계 125MiB이다.
Last12의 총 Payload는 875MiB이다. Native보다 250MiB 적다.
Native 전체 KV를 Layer Split의 추가 Overhead로 세지 않았다.
독립적인 Wire-only Time과 Exclusive CPU 기여율은 `NOT_IDENTIFIED`이다.
[비용과 예측 오차 보고서](dense_ar_dynamic_layer_cost_model_validation.md)에 세부 값을 보존했다.

## 12. 실패와 미해결 항목

| 항목 | 증거 수준 | 해석 |
|---|---|---|
| Continuous Chunk의 Output 불일치 | `FAILED` | 강한 Continuous Baseline 미확보. 일반 Chunk 방식의 불가능성은 아님 |
| Compile3 Output 불일치 | `FAILED` | 적격 Compile0 계열로 제한 |
| 느린 State 진단에서 Native 30s Lease 만료 | `FAILED` | 실제 ACK까지 보유하는 선택적 Guard 추가. 이후 재검증 통과 |
| D16 진단의 두 번째 Full Reference OOM | `FAILED` | 실제 Suffix 뒤 참조 계산에서 발생. 독립 Native State Hash로 재설계 후 통과 |
| Clean Kernel Capture | `NOT_MEASURED` | Drop0이어도 5% 간섭 기준 실패 |
| Spatial Compute Slack | `NOT_MEASURED` | NVML 또는 Kernel Inactivity로 대체하지 않음 |
| Post-commit D→P Recovery | `NOT_RUN` | 정상 ACK/Lifetime을 장애 복구로 과장하지 않음 |
| Dynamic 전체 GO | `NOT_ESTABLISHED` | Native-relative Decode CI와 Strong Baseline 우위 부족 |

CPU 검증은 41개 Test를 통과했다.
GPU 정합성, Marker-free 성능, Diagnostic Trace를 별도 Gate로 유지했다.
정상 종료에서는 KV/Hidden Credit 누수를 확인하지 않았다.
실패한 로그나 기존 결과는 삭제하지 않았다.

## 13. 필수 Figure와 근거

1. [QPS·Output별 Discovery](dense_ar_dynamic_layer_pd_crossover_20261113/figures/figure_1_pd_colocated_crossover.png).
2. [GPU Activity의 증거 수준별 Timeline](dense_ar_dynamic_layer_pd_crossover_20261113/figures/figure_2_gpu_activity_evidence_levels.png).
3. [실제 Dynamic Cut·Budget·Host Queue·예측 오차](dense_ar_dynamic_layer_pd_crossover_20261113/figures/figure_3_dynamic_cut_decisions.png).
4. [Held-out TTFT–Decode Pareto](dense_ar_dynamic_layer_pd_crossover_20261113/figures/figure_4_heldout_pareto.png).
5. [사전 고정 SLO의 Attainment와 Goodput](dense_ar_dynamic_layer_pd_crossover_20261113/figures/figure_5_slo_attainment_goodput.png).
6. [Dynamic Ablation과 95% CI](dense_ar_dynamic_layer_pd_crossover_20261113/figures/figure_6_dynamic_ablation.png).

SVG 원본도 같은 Directory에 저장했다.
Figure2는 Diagnostic Kernel Union과 CUDA Event Enclosure를 다른 Panel로 표시한다.
각 Panel은 자신의 실제 Server Start를 사용한다.
서로 다른 Run의 좋은 구간을 합쳐 Same-window 성공으로 표현하지 않았다.
개념적 Idle이나 가상의 Speedup을 그리지 않았다.

## 14. 최종 질문에 대한 직접 답변

1. **Trade-off QPS:** Output64의 QPS6.0156에서 독립 반복의 성능 Trade-off를 확인했다. 전체 물리 Crossover Gate는 미통과이다.
2. **Output 영향:** Output16에서는 Colocated가 TTFT와 Balanced Goodput에서 강했다. Output32는 Decode 차이가 작았다. Output64는 Colocated Tail이 커졌다.
3. **P 병목:** Queue 증가, KV Refusal0, Capacity 근접 Rate를 확인했다. Clean P Busy≥80%는 미확정이다.
4. **Colocated 간섭:** 성능 악화는 측정했다. 적격 Cohort의 직접 Prefill–Decode Kernel 간섭은 입증하지 못했다.
5. **D 계산 여유:** 실제 Layer 이전 완료는 확인했다. Clean 물리적 여유의 시간량은 `NOT_MEASURED`이다.
6. **Joint Crossover:** 성능 Trade-off는 있다. 강한 Continuous Baseline과 물리 계측까지 포함한 확정은 아니다.
7. **Static Last8의 원인:** 이번 Diagnostic에서는 Suffix Cost와 Decode Enclosure 차이를 관측했다. 과거 5.88%/6.75%의 인과 기여율은 분리하지 못했다.
8. **Idle Predictor:** 현재 알려진 Drain 오차는 정량화했다. 미래 유효 GPU Idle의 정확도는 `NOT_IDENTIFIED`이다.
9. **Deadline/Cost 효과:** 점 추정치 차이는 있다. Native-relative 안전성과 보편적 추가 우위는 미입증이다.
10. **Cut 선택:** COST는 이 Primary에서 0/12를 선택했다. IDLE/DEADLINE은 실제 0/2/4/8/12를 사용했다.
11. **5% 비열등성:** Best Static은 통과했다. COST는 두 CI 상한 때문에 미통과이다.
12. **Best Static 대비:** COST의 TTFT 비율은 0.80760, CI는 [0.75435, 0.86214]이다. 더 많은 이전과 Tail 차이도 존재한다.
13. **Colocated 대비 Goodput:** COST의 확정 우위는 없다. Static의 Decode-priority 점 추정치 우위도 CI가 1을 포함한다.
14. **Throughput:** Output16의 Native/Colocated/Static/COST는 54.118/70.158/59.432/62.506 Output Tokens/s이다. Finite-cohort 값이다.
15. **논문 기여 가능성:** 조건부이다. 안전한 Static 순이익은 있다. Dynamic의 추가 안전 이득과 적격 Continuous Baseline 비교가 더 필요하다.

## 15. 보존·복원·후속 우선순위

새 Branch는 `flashvep/dense-ar-dynamic-layer-pd-crossover-20261113`이다.
Publication Parent는 `367d50336791b4c540404053eadf924544a2ca0f`이다.
[Spec](dense_ar_dynamic_layer_pd_crossover_spec.md), [Source](../dense_ar_dynamic_layer_pd/), [Evidence](dense_ar_dynamic_layer_pd_crossover_20261113/)를 함께 제공한다.
Source Hash, Model Revision, GPU UUID, Dataset ID, Arrival Seed, Command와 Runtime Configuration을 보존했다.
[63개 Held-out 실행의 Audit](dense_ar_dynamic_layer_pd_crossover_20261113/analysis/heldout_evidence_audit.json)은 모두 통과했다.
모든 실행에서 고정한 236개 실행 Source Hash와 정상 Lifetime을 검사했다.

반복 Raw Evidence는 검증된 Lossless Archive에도 저장했다.
[Archive Manifest](dense_ar_dynamic_layer_pd_crossover_20261113/archives/large_evidence_manifest.json)에 파일별 SHA256과 복원 명령을 제시했다.
로컬 원본은 삭제하지 않았다.
사용자의 Checkout, Staged Index, 외부 Runtime Diff와 동결한 기존 파일은 보존했다.
[Resource 복원 증거](dense_ar_dynamic_layer_pd_crossover_20261113/state/resource_restoration_verified.json)에 GPU0~7의 Process Ownership과 새 생성 진행을 기록했다.

다음 연구에서는 먼저 Continuous Colocated의 Exactness를 해결해야 한다.
그다음 Native-relative Tail 변동과 Controller Zero의 비용을 분리해야 한다.
Batch-aware 완료 모델과 비용 상한의 재검증도 필요하다.
현재 결과만으로 더 복잡한 Predictor나 Kernel을 바로 확장할 근거는 부족하다.

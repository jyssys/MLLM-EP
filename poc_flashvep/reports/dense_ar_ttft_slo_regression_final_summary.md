# Native-relative TTFT와 Joint SLO 원인 조사

## 최종 과학적 판정

`SLO_FAILURE_CONFIRMED`와 검사한 `POLICY_TOTAL_EFFECT_CONFIRMED`다.
전체 GPU·DMA·Compiler 인과 분해는 `PARTIAL_ROOT_CAUSE`다.
새로운 Layer/Hybrid Serving Speedup은 확인하지 못했다.
기존209개 실행과 새 Main39개·Heldout30개 실행은 별도 Family로 재계산했다.
기존 결과와 새 결과를 합쳐 평균을 계산하지 않았다.
총278개 Timing의67104개 요청 관측값과3710695개 ITL을 검사했다.
이는67104개 고유 요청이나 독립 Server Start를 뜻하지 않는다.

## 두 연구 질문의 핵심

기존 Main400 Natural의 `SLO_FAILURE_CONFIRMED` 판정은 원시 요청으로 재현했다.
더 낮은 ITL p99가 더 높은 Joint SLO를 보장하지 않았다.
5초 부근의 TTFT 분포와 드문 긴 출력의 모든-ITL 실패가 중요했다.
TPOT는 해당 네 PD Method에서 실패하지 않았다.

Native-relative TTFT 회귀는 비교 기준을 구분해야 한다.
Stock은 Native보다 P Ready 이전 경로가 짧았다. 이후의 추가 대기가 더 컸다.
Layer-vs-Stock에서는 반대로 P 이전 경로의 손해가 D 이후의 절약보다 컸다.
모든 회귀를 P Attention 하나로 설명할 수 없다.
연산 이전, D의 Bulk Prefill 실행 조건과 Credit Feedback을 함께 봐야 한다.

새 결과는 기존 전제도 일부 반증했다.
새 Uncapped Stock의 Native-relative TTFT 회귀는 통계적으로 확인되지 않았다.
반면 Layer·Hybrid의 회귀는 Main Natural과 Heldout Replay에서 모두 확인됐다.
같은 Token의 D Cap 제한만으로도 Decode 개선과 Joint 악화가 동시에 발생했다.
따라서 Axis와 Admission 정책을 분리해야 한다.

## 실행 계약

| 항목 | 유지한 값 |
|---|---|
| Source Base | `0e2e062260103feb53bd0afbb0ac964ebb027cd9` |
| 모델 | Qwen3-8B, Original BF16, BI0, Greedy |
| Model Revision | `b968826d9c46dd6066d109eabc6255188de91218` |
| Attention·Decode | Exact Causal Attention, Full36 Decode |
| Runtime | 기존 vLLM0.26.0+cu129, PyTorch2.11.0+cu129, NIXL1.4.1 |
| Prefix Cache·Thinking | 비활성화 |
| Primary Scheduler | B8192/S1024, KV42GiB 예산/GPU, Block128 |
| Primary Workload | 동결 LongBench400-v2 Main400, Natural EOS, Cap2048, QPS2.0 |
| Main Seed | `2026112271/2026112272/2026112273` |
| Confirmation | Heldout192 Equal-output, 실행당11208개 출력 Token |
| Heldout Seed | `2026112281/2026112282/2026112283` |
| SLO | TTFT≤5000ms, 모든 Client ITL≤34ms, TPOT≤33ms/token |
| Decode Gate | 두 Native 기준의 paired CI 상한≤1.05, Absolute Decode Attainment≥95% |

Main400 SHA256은 `0845a0e44f001acf55dfd16407dc05e80dc56103590a8be3ac772473c1eb6cd9`다.
Heldout192 SHA256은 `c00eb4dd16deaa319100b47f9b22cc658e8003bdda08b56611ce544dac5e35f7`다.
Heldout192는 Discovery208에서 분리했다. Main400의 일부이기도 하다.
Calibration32와 Prime8은 Main400과 Request·Context가 모두 겹치지 않는다.
Warm은 Compiler Cache 재사용을 뜻한다. Prompt Prefix Cache 재사용이 아니다.

실제 H100 SXM80GB8개와 NVLink Topology를 실행 전에 검사했다.
Primary Timing과 Serial Capture는 모두 같은 GPU0/1 Pair에서 순차 실행했다.
P UUID는 `GPU-f217c8a0-1142-20f4-d84b-af29f3a47a0d`다.
D UUID는 `GPU-a77f3471-67d4-20b0-9fab-e502d4de5adb`다.
초기 Correctness만 네 Pair에 병렬 배치했다.
나머지 GPU는 승인된 Burner를 유지한다.

Strong Stock Token의 Full-model Compiled 경로를 유지했다.
NativeStrong과 정확한 Common-D Native도 모두 유지했다.
Common-D는 유한 P Source Window까지 같다는 뜻이 아니다.
Actual Batch도 별도 기록한다. D Cap이나 S1024를 Actual Batch로 바꾸지 않는다.

## 기존 표의 독립 재현

`HISTORICAL_RAW_RECALCULATION`이다.
주어진15개 Main400 실행의 원시 요청과 일곱 Metric이 반올림 허용오차 내에서 일치했다.
추가로 기존209개 실행의25개 Client Metric과 네 요청 Gate를 독립 재계산했다.
원래 Raw와 Cached Metric은 수정하지 않았다.

| Method | TTFT p95 ms | Mean TBT ms | ITL p99 ms | TPOT p95 ms/token | Raw tok/s | SLO tok/s | Joint % |
|---|---:|---:|---:|---:|---:|---:|---:|
| Agg DP2 | 1477.006 | 16.387 | 277.839 | 141.579 | 118.693 | 37.482 | 52.00 |
| NativeStrong | 8161.605 | 7.872 | 18.723 | 8.968 | 109.402 | 70.139 | 73.50 |
| Strong Stock Token | 11483.980 | 7.525 | 9.478 | 8.438 | 114.035 | 69.358 | 60.75 |
| Layer | 11679.085 | 7.219 | 8.538 | 7.955 | 112.677 | 49.029 | 46.42 |
| Hybrid | 11767.609 | 7.272 | 8.572 | 8.050 | 105.978 | 49.802 | 46.50 |

이 값은 독립 Start3회의 평균이다.
전체1200개 요청 관측값을 Pooling한 p95와 다르다.
Request나 Token을 독립 Server Start로 취급하지 않았다.
Raw Rate는 Drain을 포함한 유한 Cohort의 Output-only Rate다. 지속 가능한 Capacity가 아니다.

## Q1. Native보다 Split의 TTFT가 느려진 이유

기존 Main400의 요청별 Host 경로는 다음과 같다.
각 값은 Start별 요청 평균을 다시 Start 평균한 값이다. 단위는 ms다.

| Method | Arrival→P Ready | P Ready→첫 Client | Mean TTFT |
|---|---:|---:|---:|
| NativeStrong | 3443.691 | 70.883 | 3514.573 |
| Stock Token | 864.162 | 4141.997 | 5006.159 |
| Layer | 3525.294 | 2578.503 | 6103.797 |
| Hybrid | 4423.103 | 2199.199 | 6622.302 |

Stock−Native는 P 이전−2579.529ms, 이후+4071.115ms였다.
Layer−Native는 P 이전+81.603ms, 이후+2507.621ms였다.
반면 Layer−Stock은 P 이전+2661.132ms, 이후−1563.494ms였다.
같은 Layer라도 비교 기준에 따라 회귀의 위치가 다르다.
이것은 Host 평균의 위치 분해다. GPU-exclusive 비용이나 p95의 가법 분해가 아니다.

Native만 P 첫 Sample을 재사용했다는 설명은 검사한 Family에서 반증됐다.
검사한 Native와 Split은 모두 D 첫 출력 경로였다.
Native는 D의 Last-prompt Query 재계산도 수행했다.

또한 현재 Scheduler는 Bulk Prefill과 한 Token의 잔여 Query를 다르게 취급한다.
논리적 Decode가 Running이면 Bulk Prefill을 미룰 수 있다.
Native의 한 Token 잔여 Query는 해당 Bulk 조건을 피할 수 있다.
이는 Source로 확인한 실행 가능성 차이다.
요청별 Branch 원인과 그 전용 지연은 직접 계측하지 않았다.
논리적 Decode 상태와 대기의 겹침을 전부 GPU Compute 비용으로 확정하지 않는다.

## Q2. P 계산 절약과 추가 D 비용은 각각 얼마인가

적격 Serial32 Actual Batch1/1/1 Output16 Control에서만 GPU 비용을 직접 분리했다.
Main400 Natural의 Operator 비중으로 바꾸지 않는다.
Native·Layer·Hybrid는 각각 prospective Clean/Profile/Clean N3가 5% 간섭 Gate를 통과했다.

| Candidate−Native | P Kernel Union ms | D-first Kernel Union ms | P+D-first Kernel ms |
|---|---:|---:|---:|
| Layer Last2 | −8.963 [−9.175, −8.698] | +9.643 [9.195, 10.066] | +0.680 [0.061, 1.352] |
| Hybrid Last2+Suffix1/8 | −49.980 [−50.294, −49.591] | +49.744 [49.349, 50.150] | −0.237 [−0.517, 0.022] |

이 Control에서는 이전한 P 비용과 추가 D-first 비용이 거의 상쇄됐다.
D-first는 Continuation과 첫 Worker 출력 준비의 경계다. 순수 Suffix GEMM만의 시간이 아니다.
P+D-first Kernel 합계도 TTFT가 아니다.
Stock Capture는 ITL p99 간섭+20.1861%로 `DIAGNOSTIC`이다.
예정 후속 두 Capture는 `NOT_RUN`이다. 적격 Stock GPU 절약량은 확정하지 못했다.
Main400 Natural의 전용 GPU 비용과 DMA 전용 시간도 `NOT_MEASURED`다.

기존 Main400의 READ Host Call은 Method에 따라 평균 약14–16ms였다.
그 앞의 READ 준비 대기와 ACK 이후의 Scheduler 대기는 훨씬 길었다.
READ Host Enclosure를 DMA로 바꾸거나 전체 D 대기를 설명하는 값으로 사용하지 않는다.

이 Best-family 비교는 동일 Offloaded Rows가 아니다.
Stock의 명목 D Rows는43627968개, Layer는10906980개, Hybrid는34084644개다.
Layer의 짧은 D 경계를 전부 GEMM Efficiency로 설명하지 않는다.
Actual Work Audit는 상속한 `target_D_rows` 대신 실제 Partition·Model Visit를 검사한다.

## Q3. 인과 효과와 동시 관측 비용의 구분

동일 Source·GPU·Arrival에서 D Cap, Request Registration과 Credit 반환을 각각 변경했다.
각 Runtime Signature의 변경 Field가 하나인지 검사했다.
정책 효과에는 이후 Batch·Queue·Natural Output의 변화가 포함될 수 있다.
이를 순수 Host 시간의 독립적인 효과로 바꾸지 않는다.
Main39개와 Equal-output30개 Timing이 모두 완료됐다.
13개 정책의 새 GPU Composition Gate도 모두 통과했다.

같은 Stock의 D Cap0→2에서 Main TTFT p95 비율은1.41442 [1.10517, 1.67120]였다.
SLO-goodput 비율은0.71194 [0.33564, 0.95766]였다.
Equal Replay에서도 각각1.17105 [1.13126, 1.22165]와0.88355 [0.85273, 0.91061]였다.
Replay는 요청별 출력 길이까지 같았다.
따라서 이 정책 효과를 Natural EOS 작업량 차이만으로 설명할 수 없다.

Replay의 P Ready 이전 변화는−3.826ms, 이후는+944.377ms였다.
D Accept→Prepare가1107.700ms 증가했다.
D Prepare→First Worker는140.055ms 감소했다.
즉 짧은 D 실행 준비 경계가 더 큰 D 앞 Queue를 상쇄하지 못했다.
이는 정책 총 효과다. 각 Host Stage의 독립적인 인과 기여율은 아니다.

Request Registration 지연 해제는 앞 Queue를 다음 Queue로 옮겼다.
Replay Native는 D Submit→Accept−645.657ms, Accept→Prepare+639.380ms였다.
TTFT p95 비율은1.00001 [0.99912, 1.00053]였다.
Layer·Hybrid도 재현 가능한 TTFT 개선을 보이지 않았다.
안전한 Credit 반환도 Main Hybrid의 TTFT를1.15080 [1.01822, 1.23949]배로 늘렸다.
Credit 조기 반환이 항상 TTFT를 줄인다는 가설은 이 조건에서 반증됐다.

기존 Heldout192의15개 paired Control은 재사용했다.
Source·GPU·Arrival·출력 계약이 일치했고, Signature 변경이 한 Field였다.
Credit 반환만으로 TTFT 회귀가 일관되게 해소됐다는 증거는 없었다.
Warm Heldout Replay의 Coordinator GC 유예는 TTFT 비율0.98563 [0.98113, 0.99097]을 보였다.
이 결과로 과거 Main400의 모든 전달 Tail 원인을 GC로 확정하지 않는다.

## Q4. ITL·TPOT가 좋은데 Joint SLO가 낮은 이유

Joint는 TTFT, 모든 ITL과 TPOT의 AND다.
p95 한 점은 5초 근처의 TTFT 통과 수를 나타내지 않는다.
기존 Layer의 p50는6265.086ms, Stock은4064.664ms였다.
p50 차이는2200.422ms다. 반면 p95 차이는195.105ms였다.
두 p95가 비슷해 보여도 TTFT-only Attainment는46.83%와61.08%였다.

또한 낮은 Interval p99가 높은 요청별 모든-ITL 통과를 보장하지 않는다.
드문 큰 ITL 하나로 긴 요청의 모든 출력 Token이 Goodput에서 제외될 수 있다.
Layer의 ITL-only 실패5건은3498개 Token을 제외했다.
Stock의4건은462개 Token을 제외했다.
이 Layer5건은 Worker Emission Gap이 모두34ms 이하였다.
Client 전달 Lag의 변화가 해당 경계 초과에 필요했음을 확인했다.
이는 GPU Kernel 간섭이나 특정 GC의 원인을 확정한 것이 아니다.

## Q5. 개별 실패와 중복 Count

분모는 Method별400개 요청×Start3회, 총1200개 요청 관측값이다.
같은 요청이 여러 Start에 다시 등장한다.
아래 Count를1200개 고유 요청이나 독립 Start로 해석하지 않는다.

| 결과 | NativeStrong | Stock | Layer | Hybrid |
|---|---:|---:|---:|---:|
| Joint PASS | 882 | 729 | 557 | 558 |
| TTFT-only Fail | 266 | 467 | 634 | 625 |
| ITL-only Fail | 33 | 4 | 5 | 8 |
| TTFT+ITL Fail | 19 | 0 | 4 | 9 |
| TPOT-only / TTFT+TPOT / ITL+TPOT / All Fail | 각0 | 각0 | 각0 | 각0 |

Layer의 전체 TTFT 실패는638건, Stock은467건이다.
Layer의 전체 ITL 실패는9건, Stock은4건이다.
네 PD Method의 TPOT 실패는 모두0건이다.
Agg를 포함한 여덟 범주의 Count·Fraction·Token Weight는 별도 SLO 보고서에 전부 있다.

## Q6. 비슷한 Raw Throughput과 다른 Goodput

각 Start에서 `Goodput = Raw Rate − 실패 범주별 제외 Token Rate 합계`가 성립한다.
기존 Layer−Stock의 Goodput 차이는−20.328347 tok/s다.
Raw 차이는−1.357679 tok/s다.
제외 Rate 차이는 TTFT-only+13.127853, ITL-only+5.185026, TTFT+ITL+0.657789 tok/s다.
나머지 실패 범주의 차이는0이다.
이 항등식은 관측된 차이를 정확히 재현한다. Runtime 인과 기여율은 아니다.

Stock과 Layer의 전체 출력 합계는66347개와66361개였다.
합계가 비슷해도 SLO를 통과한 요청과 출력 Token의 구성이 달랐다.
Hybrid의 실제 출력은62459개였다. Stock보다 약5.86% 적었다.
Natural 작업량 차이를 통제하는 분석은 별도 Equal-output Replay로 수행했다.

새 Main에서도 같은 지표의 차이를 확인했다.
Stock·Layer·Hybrid의 Joint는73.83%·48.75%·44.67%였다.
TTFT-only 실패는314·613·661건이었다. 분모는 Method별1200개 관측값이다.
ITL-only 실패는0·0·1건이었다. TPOT 실패는 모두0건이었다.
이 새 Main의 Joint 격차는 주로 TTFT 통과 수의 차이였다.
과거 Layer의 긴 ITL-only5건 효과를 새 Main의 원인으로 복사하지 않았다.

Replay에서는 드문 긴 출력의 효과도 남았다.
Hybrid의 ITL-only2건은2648개 출력 Token을 제외했다.
전체 분모는33624개 Token과576개 요청 관측값이다.
낮은 Interval p99와 요청별 모든-ITL 통과는 계속 다른 Metric이었다.

## Q7. 5초 SLO를 유지할 때 개선 우선순위

Stock의 우선 조사 위치는 D Accept→Prepare다.
Layer는 P Admission·Source Feedback과 D 준비 대기를 함께 봐야 한다.
Hybrid는 Coordinator Credit 앞 대기도 중요했다.
Stage 전체를 없앤 Frozen-path 진단은 실제 구현 가능성을 보장하지 않는다.
이 Oracle은 이후 Queue·KV·ITL·출력·측정 기간을 갱신하지 않는다.
예를 들어 Stock의 D Accept→Prepare 전체 제거는 가상 Joint 통과를 실행당133.667건 늘렸다.
가상 Goodput은107.759 tok/s였다. 실제 Online Speedup이 아니다.
안전한 Credit 반환만으로 이 전체 대기를 제거할 수 있다는 뜻도 아니다.

## Q8. Native를 이길 실현 가능한 Headroom

Frozen-path에서 Native 목표에 도달했다는 사실만으로 실현 가능한 Headroom을 선언하지 않는다.
정합성·품질과 두 Native Decode Gate를 통과한 실측만 사용한다.
검증한 Decode-safe Window가 없으므로 위험한 Active-decode GPU 삽입은 수행하지 않는다.
전체 원인의 독립적인 GPU·DMA 기여율이 남아 `PARTIAL_ROOT_CAUSE`로 판정했다.
원인 조사 완료와 새로운 Serving Speedup 성공은 다른 판정이다.

새 Stock/NativeStrong의 Main TTFT p95 비율은0.94331 [0.75864, 1.35027]였다.
NativeCommon0 대비는0.90320 [0.75334, 1.17302]였다.
Stock의 유의한 회귀도 개선도 확정하지 못했다.
Main Stock의 Goodput은 NativeStrong보다 높았지만 Common0 대비 CI는1을 포함했다.
기존 Strong Stock은 새로운 제안 방법이 아니다.
이 결과를 Layer/Hybrid의 개선이나 새 Dynamic Speedup으로 제시하지 않는다.

새 Main Layer/Stock TTFT 비율은1.32140 [1.20357, 1.42187]였다.
Hybrid/Stock은1.66590 [1.48420, 1.84302]였다.
Replay에서는 각각1.14728 [1.12798, 1.17350]와1.27086 [1.19478, 1.36759]였다.
두 Method의 TTFT 손해는 출력 길이를 통제해도 남았다.
Native를 이기는 Layer/Hybrid의 실현 가능한 Headroom은 이 캠페인에서 입증하지 못했다.

## 새 Main400 Natural N3의 최종 표

아래는 원래 v2와 합치지 않은 새 Warm-runtime Family다.
GC·I/O·Priming 등 공통 Runtime 차이는 One-factor 개입이 아니다.
기존 Stock보다 새 Stock이 좋아진 전체 효과를 그중 하나로 배분하지 않는다.
각 값은 독립 Start3회의 Metric 평균이다.

| Method | TTFT p95 ms | Mean TBT ms | ITL p99 ms | TPOT p95 ms/token | Raw tok/s | SLO tok/s | Joint % |
|---|---:|---:|---:|---:|---:|---:|---:|
| NativeStrong | 8150.324 | 7.899 | 19.250 | 9.059 | 112.734 | 66.704 | 72.08 |
| NativeCommon0 | 8512.282 | 7.686 | 15.263 | 9.910 | 109.591 | 77.969 | 72.50 |
| NativeCommon2 | 8713.865 | 7.355 | 8.975 | 8.183 | 111.116 | 77.121 | 69.42 |
| Strong Stock Token | 7688.311 | 7.463 | 9.358 | 8.380 | 109.280 | 80.536 | 73.83 |
| StockCap2 Control | 10874.534 | 7.324 | 8.712 | 8.130 | 108.175 | 57.337 | 52.75 |
| Layer | 10159.310 | 7.192 | 8.466 | 7.917 | 112.352 | 55.577 | 48.75 |
| Hybrid | 12807.987 | 7.258 | 8.528 | 8.011 | 106.724 | 49.039 | 44.67 |

새 평균 Host 경로도 다음과 같이 분리했다. 단위는 ms다.

| Method | Arrival→P Ready | P Ready→첫 Client | Mean TTFT |
|---|---:|---:|---:|
| NativeStrong | 3427.277 | 73.697 | 3500.975 |
| Stock Token | 793.547 | 2648.133 | 3441.680 |
| Layer | 3087.235 | 2397.752 | 5484.988 |
| Hybrid | 4639.125 | 2248.951 | 6888.075 |

Stock−Native의 P 이전−2633.730ms와 이후+2574.436ms가 거의 상쇄됐다.
Layer−Native는 P 이전−340.042ms였지만 이후+2324.055ms였다.
Layer−Stock은 P 이전+2293.688ms, 이후−250.381ms였다.
따라서 Native-relative 회귀와 Layer-vs-Token 패배를 같은 P 원인 하나로 설명할 수 없다.

## Safety와 남은 한계

새 Main의9개 Split Arm과 Replay의6개 Split Arm은 두 Native Decode Gate를 통과했다.
모두 Frozen Task Quality Gate도 통과했다.
그렇지만 일부 요청은34ms ITL을 초과했다. 실패 요청과 Max를 보존했다.
Main NativeStrong의 평균 Absolute Decode Attainment는93.0833%로95% 기준 미달이었다.
Common0·Common2와 상대 Gate를 별도로 제시했다.
Registration 해제는 Control 대비 ITL5% 비열등성을 일부 통과하지 못했다.
이를 정확한 Common Native Reference로 바꿔 숨기지 않았다.

적격 Serial Kernel 비용과 Main Host Queue를 직접 합산하지 않았다.
Stock Serial Capture의5% 간섭 실패도 유지했다.
Native와 Split의 One-token·Bulk Admission 차이는 Source로 확인했다.
해당 Branch의 요청별 전용 지연과 Main GPU Operator 기여율은 미측정이다.
Best-family는 동일 D Rows가 아니다.
Frozen-path Oracle은 Queue·KV·ITL과 이후 출력 상태를 갱신하지 않는다.
Oracle 결과는 구현 가능한 Speedup이나 전역 성능 상한이 아니다.

## 5초 SLO 회복의 Metric-level 진단

새 Main에서 관측한 한 Host 경계를100% 제거한 Frozen-path 결과다.
아래 수치는 실제 개입의 회복량이 아니다.
분모와 이후 출력·ITL·Cohort 기간을 고정했다.
서로 겹치는 경계의 증가량을 더하지 않는다.

| Method·가상 제거 경계 | 실행당 추가 Joint 통과 요청 | 가상 SLO tok/s |
|---|---:|---:|
| Stock D Accept→Prepare 전체 | 96.000 | 106.630 |
| Layer P Ready 이전 전체 | 144.667 | 97.537 |
| Layer D Accept→Prepare 전체 | 95.000 | 79.931 |
| Hybrid Coordinator 대기 전체 | 189.000 | 99.582 |
| Hybrid D Accept→Prepare 전체 | 34.000 | 58.047 |

이 진단은 Stock의 D 준비, Layer의 P·D Queue, Hybrid의 Source Feedback을 우선 조사할 근거다.
하지만 실제 Registration 해제와 Credit 반환 개입은 같은 가상 이득을 재현하지 못했다.
따라서 Oracle상 큰 Stage가 있다는 사실만으로 간단한 Guard 제거를 제안하지 않는다.
[가상 Stage 제거 CSV](dense_ar_native_ttft_joint_slo_20261124/figures/frozen_path_stage_oracles.csv)에 모든 Fraction과 Start를 보존했다.

새 Main Layer는 Stock보다 Raw Rate가3.071935 tok/s 높았다.
그러나 TTFT-only 제외 Rate가27.686429 tok/s 더 높았다.
TTFT+ITL 제외 Rate도0.344590 tok/s 더 높았다.
따라서 Goodput 차이는−24.959083 tok/s였다.
`3.071935 − 27.686429 − 0.344590 = −24.959083`이다.
반올림 전 값으로 항등식을 검사했다.
이는 Metric 손실의 정확한 분해다. Runtime 인과 기여율은 아니다.

## 보고서와 재현 자료

- [Native-relative TTFT 원인](dense_ar_native_relative_ttft_root_cause.md).
- [Joint SLO 여덟 범주와 Token 가중](dense_ar_joint_slo_failure_attribution.md).
- [Native·Split 요청별 Critical Path](dense_ar_native_token_layer_request_trace.md).
- [Controlled Intervention과 Safety Gate](dense_ar_pd_critical_path_controlled_interventions.md).
- [연구 결정 기록](dense_ar_native_ttft_joint_slo_20261124/research_decision_log.md).
- [원래15개 실행의 재현 JSON](dense_ar_native_ttft_joint_slo_20261124/analysis/historical_main400_metric_reproduction.json).
- [실행별 Metric CSV](dense_ar_native_ttft_joint_slo_20261124/figures/metrics_per_start.csv).
- [8-category Figure](dense_ar_native_ttft_joint_slo_20261124/figures/joint_SLO_eight_category_breakdown.png).
- [Raw Rate와 SLO-goodput Figure](dense_ar_native_ttft_joint_slo_20261124/figures/raw_output_throughput_vs_SLO_goodput.png).
- [새 Main400 TTFT CDF](dense_ar_native_ttft_joint_slo_20261124/figures/fresh_main400_native_split_ttft_CDF.png).
- [새 Main의 One-factor 정책 효과](dense_ar_native_ttft_joint_slo_20261124/figures/policy_effects_current_main400_natural.png).
- [Equal Replay의 정책 효과](dense_ar_native_ttft_joint_slo_20261124/figures/policy_effects_current_heldout192_equalreplay.png).
- [Serial 적격 Kernel·Host 경계](dense_ar_native_ttft_joint_slo_20261124/figures/serial_native_split_kernel_and_host.png).
- [재계산·Archive 안내](dense_ar_native_ttft_joint_slo_20261124/README.md).

모든 Main Figure는 PNG/PDF와 대응 CSV를 제공한다.
Main400 Natural, Heldout Replay, Serial Kernel Control과 Metric Oracle은 구분한다.
Pointwise Start-level CI를 사용한다. 다중 검정 보정은 하지 않는다.
원래 Checkout·Index·이전 Source와 Evidence는 보존한다.
CPU Regression, GPU Composition Gate와 최종 Metric 재계산을 통과했다.
17개 Figure의 PNG·PDF·CSV, 총51개 파일은 두 번 생성한 Byte가 일치했다.
승인된 GPU0–7 Burner를 복원했다.
각 GPU의 소유권과 연속 두 번70% 이상 사용률을 확인했다.
사용률은 Resource 확인용이다. 연구의 Kernel Busy Metric이 아니다.
[Resource Closeout](dense_ar_native_ttft_joint_slo_20261124/state/resource_closeout.json)에 실제 Process와 Sample을 보존했다.

연구 GPU Dispatch는246개였다. Correctness·Prime·Startup·Timing을 포함한다.
예약 Process 시간은21.6003 GPU-hours였다. Kernel-exclusive 시간은 아니다.
마지막 Dispatch 종료는 시작 후9.1370시간이었다.
Burner와 CPU 분석 시간은 이 GPU-hours에서 제외했다.
사용자 Checkout·Index와 기존 Source·Evidence의 보존 검사도 통과했다.
무손실 Archive와 GitHub Remote SHA는 별도 Publication 기록에서 확인한다.

Raw·Config·Log 등62661개 Member의 로컬 Full Archive를 검사했다.
공개 Archive는 원문·Tensor·Full Log 등을 제외한12493개 Member다.
공개 Part6개는 연결된 원래 Archive의 SHA256과 일치했다.
각 압축 해제 Member도 Byte 수와 SHA256이 일치했다.
원래 Raw와 큰 파생 CSV는 로컬에서 삭제하지 않았다.
[Archive Manifest](dense_ar_native_ttft_joint_slo_20261124/archives/evidence_archive_manifest.json)에 모든 Member와 제외 이유가 있다.

Notion Archive는 읽기 전용으로 접근했다.
제한한 Schema Query에서 관련 Dense·LongBench 항목은 반환되지 않았다.
로컬 불변 Raw와 고정 Source를 Primary 근거로 사용했다.
[Notion 접근 기록](dense_ar_native_ttft_joint_slo_20261124/references/notion_archive_audit.json)에 Query 범위를 남겼다.

## GitHub Publication

Science Commit은 `308515a484d0be4a89b5d32498047f9e4f2ef1de`다.
별도 Branch `flashvep/dense-ar-native-ttft-joint-slo-20261124`에 Push했다.
`git ls-remote`로 같은 Remote SHA를 확인했다.
이후 Closeout Commit에는 이 원격 확인 기록과 최종 문서를 추가한다.
따라서 Science Commit SHA와 최종 Branch HEAD가 다를 수 있다.
[Science Publication 확인 기록](dense_ar_native_ttft_joint_slo_20261124/state/publication_commit_science.json)에 File Hash와 Remote SHA가 있다.
사용자의 기존 Checkout과 Staged Index는 변경하지 않았다.

# PD Critical Path의 Controlled Intervention

## 목적과 증거 구분

이번 캠페인은 새로운 최적 Partition을 찾는 Sweep이 아니다.
Native 대비 TTFT 회귀의 위치와 정책의 인과 효과를 구분한다.
기존 Main400 Natural 결과는 `TRACE_ANALYSIS`로 재계산했다.
새 Main400 Natural과 Heldout192 Equal-output은 별도 실측 Family다.
Serial32 Kernel Control은 `ISOLATED_KERNEL_MEASUREMENT`다.
Gate 제거와 Host 경계 제거는 `OFFLINE_ORACLE`이다.
이 네 종류의 수치를 같은 Start 평균에 합치지 않는다.

## 사전에 동결한 개입

| Candidate / Control | 변경한 Factor | 유지한 주요 계약 |
|---|---|---|
| StockCap2 / StockToken | D Running Admission Cap0→2 | Strong Full-model Compiled 경로, 무제한 Source Window |
| StockCap2RegOpen / StockCap2 | D Request Registration 지연 ON→OFF | Cap2, Compute Throttle, Remote READ 지연 |
| NativeRegOpen / NativeCommon2 | 같은 Registration 변경 | Full36 Native 경로와 나머지 Common-D 정책 |
| NativeCommon0 / NativeCommon2 | D Cap2→0 | 나머지 Common-D 정책 |
| LayerRegOpen / Layer | 같은 Registration 변경 | Last2, Source16, Cap2, Compiler 설정 |
| HybridRegOpen / Hybrid | 같은 Registration 변경 | Last2+Suffix1/8, Source8, Cap2, Compiler 설정 |
| LayerRetirement / Layer | Coordinator Credit의 안전한 반환 시점 | Last2, Source16과 실제 Buffer Lifetime |
| HybridRetirement / Hybrid | 같은 Credit 반환 변경 | Last2+Suffix1/8, Source8과 실제 Buffer Lifetime |

각 쌍의 Runtime Signature에서 변경 Field가 하나인지 검사한다.
정책 변경이 실제 Batch와 이후 Queue를 바꿀 수 있다.
따라서 One-factor 설정이 Actual Batch까지 같다는 뜻은 아니다.
이후 Queue Feedback을 포함한 정책의 총 효과로 해석한다.
개별 Stage 차이를 서로 독립적인 인과 기여율로 합산하지 않는다.

NativeStrong은 별도 강한 기준선으로 유지한다.
Uncapped Stock의 Common-D 기준은 NativeCommon0이다.
Cap2 Arm의 기준은 NativeCommon2다.
Registration-open Arm의 정확한 Common-D 기준은 NativeRegOpen이다.
이 Native 대조군의 Source Window는 무제한이다.
유한 Source Window의 Layer·Hybrid와 Common-P Policy까지 같다고 주장하지 않는다.

## Registration Intervention의 의미

기존 Wrapper는 Output을 시작한 Running 요청이 있으면 새 `add` 명령을 보관한다.
Decode Step, Token 전달과 Connector Retirement는 계속 실행한다.
이 정책은 새 요청 등록의 Host 간섭을 줄이기 위한 보수적인 장치다.
동시에 D Submit→Accept를 늘릴 수 있다.

RegOpen은 이 Request Registration 지연만 해제한다.
Bulk Prefill Compute Throttle과 Remote READ 지연은 해제하지 않는다.
따라서 Active Decode 사이에 임의의 GPU Prefill을 삽입하는 실험이 아니다.
실제 Decode-safe Window나 미래 Arrival을 알고 있다고 가정하지 않는다.
Registration 해제가 TTFT를 줄여도 Decode Gate는 별도로 검사한다.

설치한 Scheduler에는 Bulk Prefill과 한 Token의 Native 잔여 Query를 구분하는 조건이 있다.
같은 Throttle Flag라도 Native와 Split의 실행 가능성이 같지 않다.
이 Source 근거는 [Scheduler Audit](dense_ar_native_ttft_joint_slo_20261124/references/installed_scheduler_predicate_audit.json)에 있다.
각 요청이 어떤 Branch 때문에 몇 ms 대기했는지는 직접 계측하지 않았다.
논리적 Decode 상태와 대기 구간의 겹침을 전용 GPU Compute 비용으로 해석하지 않는다.

## Credit Intervention의 안전 조건

원래 Layer·Hybrid는 첫 Client Token을 받으면 Coordinator Credit를 반환한다.
Retirement Arm은 Source Resource가 안전하게 퇴역한 뒤 Credit를 반환할 수 있다.
다음 조건을 모두 유지한다.

1. 원래 Native KV 송신 ACK 처리가 완료된다.
2. 원래 Scheduler의 Request 제거와 GPU Fence가 완료된다.
3. Hidden이 필요하면 수신 ACK가 존재한다.
4. Hidden Owner와 Buffer가 원래 Reap 절차로 해제된다.

Native KV Block을 직접 Free하지 않는다.
GPU Fence보다 먼저 Buffer를 재사용하지 않는다.
첫 Client Token 반환은 Fallback으로 유지한다.
변경 대상은 Coordinator의 독립 Outstanding Credit다.
KV Capacity, Actual Batch와 Hidden Pool 크기를 Credit와 같은 값으로 취급하지 않는다.

## Correctness와 출력 품질

초기 12개 정책과 NativeCommon0 추가 대조군을 새로 검증했다.
총13개 Composition Gate가 통과했다.
순서는 Single1, Prime8, Warm Strict8A/B, Natural32, Natural400이다.
Same-geometry Strict 검사에는 Output, Prompt KV, Hidden과 Full36 Decode가 포함된다.
Query·Layer Coverage, ACK와 Buffer Lifetime도 검사했다.
BI0의 다른 Batch Geometry 사이에서 완전한 Greedy Token 일치를 자동 가정하지 않았다.
원래 Compiled C1 Teacher에 대한 Task Macro 품질 허용 한계2pp를 유지했다.
품질 Bootstrap은 Context Cluster를 사용한다.
이 Qualification 시간은 성능 표에 넣지 않는다.

Main과 Heldout의 각 Timing도 별도 Prime8과 품질 검사를 수행한다.
원시 출력이나 EOS를 덮어쓰지 않는다.
Natural의 실제 출력 작업량 변화는 요청별 Count로 보존한다.
Replay에서는 Frozen C1 Natural 길이를 사용해 출력 작업량을 맞춘다.

## 실측 계약과 반복

모든 Primary Timing은 GPU0/1 한 Pair에서 순차 실행한다.
다른 여섯 GPU는 승인된 Burner를 유지한다.
초기 GPU Qualification만 네 Pair에 병렬 배치했다.
Qualification의 병렬 시간은 Primary 성능으로 사용하지 않는다.

Main400은 Natural EOS, Cap2048, QPS2.0이다.
동결 Seed는 `2026112271/2026112272/2026112273`이다.
13개 Arm의 독립 Start3회를 계획했다.
순서는 Forward, Reverse, 고정 Shuffle이다.
Heldout192는 정확히11208개 출력 Token의 Equal-output Replay다.
Seed는 `2026112281/2026112282/2026112283`이다.
10개 Arm의 전체 N3 일정이 시간 예산에 들어가는지 먼저 검사한다.
들어가지 않으면 D17의 핵심7개 Arm 전체 N3가 들어가는지 검사한다.
핵심은 NativeStrong, Common0, Common2, Stock, StockCap2, Layer, Hybrid다.
Registration-open Cell은 이 시간 Fallback에서만 `NOT_RUN`으로 남긴다.
D17은 Main N3 완료와 최초 Heldout 결과 이전에 정의했다.
이 예산 결정과 핵심 Arm 선정에는 성능 효과의 방향이나 유의성을 사용하지 않는다.
미실행 Cell과 실패 Cell도 계획에서 제거하지 않는다.

모델 Revision은 `b968826d9c46dd6066d109eabc6255188de91218`다.
BF16·BI0·Exact Causal Attention·Full36 Decode·B8192/S1024·KV42GiB·Block128을 유지한다.
Main400 Hash는 `0845a0e44f001acf55dfd16407dc05e80dc56103590a8be3ac772473c1eb6cd9`다.
Heldout192 Hash는 `c00eb4dd16deaa319100b47f9b22cc658e8003bdda08b56611ce544dac5e35f7`다.
GPU UUID, Command와 Source Capsule SHA는 실행별 Raw에 있다.

현재 Timing 결과의 수집 상태와 실패 사유는 다음 파일에 보존한다.

- [Main400 실행 기록](dense_ar_native_ttft_joint_slo_20261124/state/main400_native_relative_progress_v2.json).
- [Heldout 사전 계획](dense_ar_native_ttft_joint_slo_20261124/jobs/heldout192_native_relative_confirmation_v1.json).
- [정책별 독립 Start 분석](dense_ar_native_ttft_joint_slo_20261124/figures/controlled_intervention_per_start.csv).
- [정책 쌍의 효과와 CI](dense_ar_native_ttft_joint_slo_20261124/figures/controlled_intervention_contrasts.csv).
- [두 Native 기준의 Decode Gate](dense_ar_native_ttft_joint_slo_20261124/figures/controlled_intervention_decode_gates.csv).

## 새 Main400 Natural N3

39개 Timing이 모두 완료됐다. 13개 Arm은 각각 독립 Start3회다.
모든 실행은 Frozen Task Quality Gate를 통과했다.
아래 값은 새 Warm-runtime Family다. 원래 v2의 평균과 합치지 않는다.
Controller Metric과 독립 Raw 재계산을 모두 보존했다.
Heldout192 Equal-output Replay도 전체30개 Timing을 완료했다.
사전 시간 검사에서 전체 일정이 가능했다. D17의 핵심7개 Fallback은 사용하지 않았다.

| Method | TTFT p95 ms | ITL p99 ms | TPOT p95 ms/token | SLO tok/s | Joint % |
|---|---:|---:|---:|---:|---:|
| NativeStrong | 8150.324 | 19.250 | 9.059 | 66.704 | 72.08 |
| NativeCommon0 | 8512.282 | 15.263 | 9.910 | 77.969 | 72.50 |
| NativeCommon2 | 8713.865 | 8.975 | 8.183 | 77.121 | 69.42 |
| Strong Stock Token | 7688.311 | 9.358 | 8.380 | 80.536 | 73.83 |
| StockCap2 | 10874.534 | 8.712 | 8.130 | 57.337 | 52.75 |
| Layer | 10159.310 | 8.466 | 7.917 | 55.577 | 48.75 |
| Hybrid | 12807.987 | 8.528 | 8.011 | 49.039 | 44.67 |

새 Stock/NativeStrong TTFT p95 비율은0.94331 [0.75864, 1.35027]이다.
Stock/NativeCommon0은0.90320 [0.75334, 1.17302]다.
따라서 새 Uncapped Stock의 Native-relative TTFT 회귀는 통계적으로 확인되지 않는다.
원래 표의 회귀를 모든 Runtime에서 유지되는 현상으로 일반화하지 않는다.
기존 v2와 새 Family의 차이는 One-factor 비교가 아니다.
GC·I/O·Stable Source·Port·Priming의 전체 차이를 특정 원인 하나로 배분하지 않는다.

Layer/NativeStrong은1.24649 [1.02129, 1.84068]다.
Hybrid/NativeStrong은1.57147 [1.28110, 2.37211]다.
Layer/Stock은1.32140 [1.20357, 1.42187]다.
Hybrid/Stock은1.66590 [1.48420, 1.84302]다.
해당 두 Partial Method의 TTFT 회귀는 새 Family에서도 관측된다.
이 Best-family 쌍은 동일 Offloaded Rows가 아니다.

### 한 Factor의 실제 효과

아래 비율의 분자는 Candidate, 분모는 같은 Start의 Control이다.
모두 새 Main400 Natural N3다. 95% CI는 Pointwise다.

| Candidate / Control | TTFT p95 비율 [95% CI] | SLO-goodput 비율 [95% CI] |
|---|---:|---:|
| StockCap2 / StockToken | 1.41442 [1.10517, 1.67120] | 0.71194 [0.33564, 0.95766] |
| StockCap2RegOpen / StockCap2 | 0.99586 [0.96731, 1.04782] | 1.02884 [0.94857, 1.50855] |
| NativeRegOpen / NativeCommon2 | 0.99955 [0.99599, 1.00159] | 1.00699 [0.98662, 1.02429] |
| LayerRegOpen / Layer | 1.00049 [0.98441, 1.01172] | 1.00976 [0.98023, 1.04463] |
| HybridRegOpen / Hybrid | 0.98484 [0.96564, 1.00419] | 1.01357 [0.97406, 1.05145] |
| LayerRetirement / Layer | 1.00630 [0.99617, 1.02212] | 0.98783 [0.97943, 0.99162] |
| HybridRetirement / Hybrid | 1.15080 [1.01822, 1.23949] | 0.82666 [0.61530, 1.01838] |

D Cap0→2는 같은 Token Arm의 TTFT를 악화시켰다.
반대로 ITL p99 비율은0.93101 [0.90763, 0.95706]으로 개선됐다.
TPOT p95도0.97011 [0.95435, 0.98733]로 개선됐다.
그런데 Joint는73.83%→52.75%, Goodput은80.536→57.337 tok/s로 낮아졌다.
Decode Tail과 Joint SLO의 차이를 같은 Axis의 정책 개입으로도 재현했다.
Strong Stock의 Primary는 원래 Cap0다. Cap2를 약한 Token Primary로 대체하지 않았다.

Cap2와 Cap0의 Natural 출력 합계는63728개와63573개였다.
Aggregate 차이는 약+0.244%다.
요청별 길이와 Batch까지 같다는 뜻은 아니다.
해당 총 효과를 Pure Admission 지연만의 인과 기여로 바꾸지 않는다.
Equal-output Replay로 길이 변화의 영향을 별도로 확인한다.

Registration 지연 해제는 네 Arm에서 TTFT의 재현 가능한 개선을 보이지 못했다.
StockCap2의 RegOpen은 ITL p99 비율1.05402 [1.03804, 1.07327]을 보였다.
NativeRegOpen/Common2는1.04047 [1.00932, 1.06817]이다.
각 Control 대비 5% Tail 비열등성은 이 두 비교에서 통과하지 못한다.
Candidate의 두 Native 기준 Gate와 이 One-factor Control Gate는 다른 비교다.
정확한 Common-D Reference를 바꿨다는 이유로 Control-relative 악화를 숨기지 않는다.

안전한 Credit 반환도 TTFT를 일관되게 줄이지 못했다.
Hybrid에서는 TTFT가 약15.08% 증가했다. CI도1보다 컸다.
Hybrid Goodput 비율의 CI는1을 포함한다. 그 하락을 통계적으로 확정하지 않는다.
Natural 출력과 이후 Queue의 변화를 포함한 정책 총 효과다.
Credit를 빨리 반환하면 항상 Online TTFT가 줄어든다는 설명은 이 조건에서 반증됐다.

[완료 Controller의 N3 Snapshot](dense_ar_native_ttft_joint_slo_20261124/analysis/main400_controller_n3_metrics_v1.json)에 Mean과 모든 paired CI를 보존했다.
최종 Host Stage와 양쪽 Native Decode Gate는 Terminal 분석 파일에서 함께 확인한다.

## Heldout192 Equal-output N3 Confirmation

각 Arm은 같은 요청별 Frozen 출력 길이를 사용했다.
각 Start는11208개, N3 합계는33624개 출력 Token이다.
Main Natural과 다른 요청 집합·Seed이므로 두 Family 차이를 Output-only 개입으로 부르지 않는다.

| Method | TTFT p95 ms | ITL p99 ms | TPOT p95 ms/token | SLO tok/s | Joint % |
|---|---:|---:|---:|---:|---:|
| NativeStrong | 7104.700 | 18.282 | 8.554 | 73.814 | 81.08 |
| NativeCommon0 | 7804.671 | 14.397 | 12.190 | 90.005 | 78.30 |
| NativeCommon2 | 8294.192 | 9.015 | 8.074 | 83.324 | 71.35 |
| Strong Stock Token | 9507.240 | 9.138 | 8.258 | 80.245 | 72.57 |
| StockCap2 | 11133.453 | 8.669 | 8.066 | 70.901 | 63.89 |
| Layer | 10907.458 | 8.427 | 7.909 | 67.231 | 60.24 |
| Hybrid | 12082.362 | 8.440 | 7.927 | 56.510 | 57.99 |

| Candidate / Control | TTFT p95 비율 [95% CI] | SLO-goodput 비율 [95% CI] |
|---|---:|---:|
| StockCap2 / StockToken | 1.17105 [1.13126, 1.22165] | 0.88355 [0.85273, 0.91061] |
| NativeRegOpen / NativeCommon2 | 1.00001 [0.99912, 1.00053] | 1.00384 [0.99666, 1.01206] |
| LayerRegOpen / Layer | 1.01001 [0.99629, 1.02432] | 1.00065 [0.99975, 1.00166] |
| HybridRegOpen / Hybrid | 0.99857 [0.99319, 1.00351] | 1.05762 [0.96334, 1.17595] |
| Layer / StockToken | 1.14728 [1.12798, 1.17350] | 0.83782 [0.77955, 0.90760] |
| Hybrid / StockToken | 1.27086 [1.19478, 1.36759] | 0.70421 [0.66491, 0.73935] |

Cap2의 TTFT 악화와 Goodput 하락은 출력 길이를 맞춰도 재현됐다.
Replay의 P Ready 이전 평균 차이는−3.826ms였다.
P Ready 이후는+944.377ms였다.
D Accept→Prepare는+1107.700ms, D Prepare→First Worker는−140.055ms였다.
짧아진 마지막 D Host 경계만으로 앞 Queue의 손해를 만회하지 못했다.

Registration 해제에서도 같은 Queue 이동을 관측했다.
Replay Layer는 D Submit→Accept−568.674ms, Accept→Prepare+593.029ms였다.
Hybrid는−504.799ms와+475.713ms였다.
앞 경계를 줄여도 전체 TTFT의 재현 가능한 개선은 없었다.
이 값은 Mean Host 경계다. 전용 Kernel 비용이나 p95의 가법 분해가 아니다.

### 실제 Batch는 정책 이름과 다르다

아래 값은 각 Start의 Step-weighted 평균을 다시 N3 평균한 값이다.
Max는 세 Start에서 관측한 최댓값이다.

| Family·Method | P Prefill Batch 평균 | D Decode Batch 평균 | D Decode Batch Max |
|---|---:|---:|---:|
| Main NativeStrong | 1.450 | 1.503 | 6 |
| Main Common0 | 1.449 | 1.260 | 20 |
| Main Common2 | 1.449 | 1.212 | 2 |
| Main Stock | 1.251 | 1.265 | 6 |
| Main StockCap2 | 1.266 | 1.171 | 2 |
| Main Layer | 1.411 | 1.037 | 2 |
| Main Hybrid | 1.298 | 1.081 | 2 |
| Replay NativeStrong | 1.396 | 1.487 | 5 |
| Replay Stock | 1.251 | 1.267 | 5 |
| Replay StockCap2 | 1.250 | 1.185 | 2 |
| Replay Layer | 1.355 | 1.041 | 2 |
| Replay Hybrid | 1.254 | 1.073 | 2 |

이 비교는 C3 Continuous다. Batch1 Mechanism Control이 아니다.
Actual Geometry 변화는 정책 개입의 결과로 보존한다.

### Decode Gate의 최종 결과

새 Main의9개 Split Arm과 Replay의6개 Split Arm은 두 Native-relative Gate를 통과했다.
각 Start의 Absolute Decode Attainment도95% 이상이었다.
이는 모든 ITL이34ms 이내였다는 뜻이 아니다.
Main Layer의 전체 Max ITL은43.812ms, Hybrid는43.391ms였다.
Replay Stock의 Max는102.907ms, Hybrid는46.024ms였다.
실패 요청과 전체 Max는 원시 로그 및 Gate CSV에 그대로 남겼다.

NativeStrong 자체의 Main Absolute Decode Attainment는 평균93.0833%였다.
이는95% 기준보다 낮았다.
따라서 NativeStrong의 상대 Tail 통과와 절대 Decode 보호를 혼동하지 않는다.
Common0·Common2는 Main에서 모든-ITL 및 TPOT를 모두 통과했다.
Registration-open의 Control-relative 악화는 위의 별도 One-factor 결과에 유지한다.
정확한 Native Reference를 바꿔 이 악화를 숨기지 않았다.

## 중복 실행 대신 재사용한 기존 N3 Control

이전 Heldout192의 Credit·GC 개입은 Raw에서 다시 검사했다.
이전 `OriginalLayer/OptimizedLayer`와 `OriginalHybrid/OptimizedHybrid`의 Retirement 쌍도 포함한다.
이번 새 Family의 대응 이름은 `Layer/LayerRetirement`와 `Hybrid/HybridRetirement`다.
이전 Geometry 변경 Family와 Retirement Family의 이름은 같아도 같은 개입이 아니다.
각 Config와 Signature로 구분했다. 이름만으로 두 Family를 합치지 않았다.
같은 쌍은 GPU Pair, Source ZIP, Seed, Arrival Offset과 출력 계약이 일치했다.
Runtime Signature의 변경 Field도 정확히 하나였다.
새 Main400과 합쳐 평균을 계산하지 않는다.

| 이전 Control | 출력 계약 | TTFT p95 비율 [95% CI] | SLO-goodput 비율 [95% CI] |
|---|---|---:|---:|
| Layer 안전 Retirement / 기존 반환 | Equal Replay | 0.98463 [0.96578, 1.00375] | 1.00956 [1.00219, 1.01687] |
| Layer 안전 Retirement / 기존 반환 | Natural | 1.01187 [0.97713, 1.07397] | 1.15750 [1.03274, 1.72789] |
| Hybrid 안전 Retirement / 기존 반환 | Equal Replay | 1.01692 [0.97742, 1.04811] | 0.97606 [0.96245, 0.98760] |
| Hybrid 안전 Retirement / 기존 반환 | Natural | 1.15118 [0.93884, 1.44300] | 1.08339 [0.78270, 1.55774] |
| Hybrid Coordinator GC 유예 / Normal GC | Equal Replay | 0.98563 [0.98113, 0.99097] | 1.04134 [1.01170, 1.08114] |

각 행은 독립 Start3회다. CI는 Pointwise다.
Natural의 실제 출력 작업량 차이는 그대로 보존했다.
Retirement만으로 TTFT 회귀가 일관되게 해소됐다는 증거는 없다.
GC 개입은 해당 Warm Heldout Replay에서 작은 TTFT·Goodput 효과를 보였다.
이 결과로 과거 Main400의 모든 Client Tail 원인을 GC로 확정하지 않는다.
GC 비교의 ITL p99 비율은0.99731 [0.99407, 0.99975]였다.
이는 동일 개입의 Client Metric이다. GPU Kernel 개선 수치가 아니다.

[재사용 Audit](dense_ar_native_ttft_joint_slo_20261124/analysis/reused_one_factor_controls.json)과
[대조 CSV](dense_ar_native_ttft_joint_slo_20261124/figures/reused_one_factor_control_contrasts.csv)에 모든 쌍의 Hash와 Stage 효과가 있다.

## Kernel Control의 결과와 한계

Serial32 Actual Batch1/1/1 Output16에서 Native·Layer·Hybrid의 CPC N3가 통과했다.
Stock 첫 CPC의 ITL p99 간섭은 +20.1861%였다.
따라서 Stock Capture는 `DIAGNOSTIC`이다. 예정 후속 두 Capture는 `NOT_RUN`이다.
Stock Profile을 Primary GPU 비용이나 Native-relative 성공으로 사용하지 않는다.
이 결과는 [Native-relative 보고서](dense_ar_native_relative_ttft_root_cause.md)에 수치로 정리했다.

Main400 Natural의 요청별 전용 GPU Operator 시간은 `NOT_MEASURED`다.
Serial Control의 GEMM·Attention 비율을 Main의 수초 Queue에 적용하지 않는다.
Host READ Enclosure도 GPU DMA 전용 시간으로 대체하지 않는다.
적격 Counter의 Kernel Union은 실제 CUDA Activity다.
해당 경계의 No-observed-kernel 시간은 Physical Idle이나 안전 Window의 증거가 아니다.

## Decode Gate와 통계

동결 SLO는 TTFT≤5000ms, 모든 Client ITL≤34ms, TPOT≤33ms/token이다.
NativeStrong과 정확한 Common-D Native에 대해 ITL p99·TPOT p95의 paired CI를 계산한다.
각 상대 CI 상한은1.05 이하여야 한다.
요청별 Absolute Decode Attainment95%도 별도 조건이다.
품질·TTFT 개선·Decode 보호는 서로 다른 판정이다.
하나의 통과를 다른 Gate의 통과로 대신하지 않는다.

세 독립 Start의 경험적 paired Bootstrap은27개 재표본을 사용한다.
Request나 Token을 독립 Server Start로 계산하지 않는다.
Pointwise CI이며 다중 검정 보정은 하지 않았다.
N3 미만이면 CI를 만들지 않는다.

## 수행하지 않은 개입

Compute Throttle을 제거한 Active-decode Prefill Insertion은 수행하지 않는다.
검증된 Safe Window가 없기 때문이다.
Zero-compute Stub이나 가짜 Transfer를 Native-equivalent로 사용하지 않는다.
모델·Dataset·SLO·Stock Compiler를 바꿔 결과를 개선하지 않는다.
GC·I/O를 함께 바꾼 과거 Runtime Bridge는 One-factor 개입이 아니다.
기존 Gate 제거·Frozen-path Oracle은 실제 Online 개선으로 보고하지 않는다.

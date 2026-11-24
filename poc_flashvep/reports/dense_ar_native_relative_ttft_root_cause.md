# Native 대비 Prefill Split의 TTFT 회귀

## 조사 범위

이 보고서는 기존 Main400 Natural Trace와 새 Serial Kernel Control을 구분한다.
새 Main400 정책 N3는 별도 Family다. 39개 Timing과 품질 Gate가 완료됐다.
Heldout192 Equal-output Confirmation도 30개 Timing과 품질 Gate를 완료했다.

기준 Commit은 `0e2e062260103feb53bd0afbb0ac964ebb027cd9`다.
모델은 Qwen3-8B, Revision `b968826d9c46dd6066d109eabc6255188de91218`다.
Original BF16, BI0, Exact Causal Attention, Full36 Decode를 유지한다.
B8192/S1024, GPU별 KV42GiB 예산, KV Block128을 유지한다.
Prefix Cache와 Thinking은 비활성화한다.
Strong Stock Token의 Full-model Compiled 경로를 유지한다.
NativeStrong과 Candidate의 실제 D 정책에 맞춘 Common-policy Native를 모두 보존한다.

이 비교의 Best-family Action은 서로 같은 Offloaded GEMM Rows가 아니다.
Stock은 Last8-equivalent Token Suffix다. Layer는 Last2다.
Hybrid는 Last2+Suffix1/8이다.
따라서 Layer의 더 짧은 D 경계를 전부 Axis Efficiency로 해석하지 않는다.
Config의 기존 `iso_work_frontier`와 `target_D_rows` 표기도 작업량 일치의 증거로 사용하지 않는다.
실제 요청별 Partition, Model Visit와 Query Coverage를 다시 검사한다.
[Actual Work Audit](dense_ar_native_ttft_joint_slo_20261124/analysis/native_relative_actual_work_audit.json)에 Structural Rows와 실제 실행 Rows를 분리했다.
Native Proxy의 Last-prompt Query 재계산과 정상적인 Preemption 반복도 실제 Rows에 남긴다.

Main400의 명목 작업량은 다음과 같다.
`STRUCTURAL_ACCOUNTING`이다. GPU 시간이나 FLOPs 실측값이 아니다.

| Action | P Token-layer Rows | D Token-layer Rows |
|---|---:|---:|
| Native Full36 | 196325640 | 0, 별도 Last-prompt Recompute 제외 |
| Stock Token Last8-equivalent | 152697672 | 43627968 |
| Layer Last2 | 185418660 | 10906980 |
| Hybrid Last2+Suffix1/8 | 162240996 | 34084644 |

Stock의 명목 D Rows는 Layer의 약4배다.
반대로 Layer의 P Rows가 더 많다.
긴 D 지연과 짧은 P 지연이 동시에 나타날 구조적인 이유가 있다.
이 Rows 비율을 시간 비율로 바꾸지 않는다.
Native의 실제 D 실행에는 Prompt Logit 재계산도 있다.
예를 들어 Start0에서400개 요청의 추가 Rows는14400개였다.
Partition의 Attention Pair도 별도 Audit에 있다. GEMM Rows와 같은 값이 아니다.

## 관측한 회귀

기존 Main400 Natural, QPS2.0, 독립 Start3회의 paired TTFT p95 비율이다.
각 비율의 분모는 같은 Start의 NativeStrong이다.

| Method / NativeStrong | p95 비율 | Start-level 95% CI |
|---|---:|---:|
| Stock Token | 1.40707 | [1.03063, 2.45208] |
| Layer | 1.43098 | [1.25673, 1.89804] |
| Hybrid | 1.44183 | [1.28720, 1.80547] |

각 CI는 paired Start3회의 경험적 Bootstrap27개 재표본을 사용한다.
Pointwise CI다. 여러 가설의 동시 검정 보정은 하지 않았다.
새 정책 Family나 Serial Output16 결과를 이 평균에 합치지 않았다.
원래 15개 실행의 일곱 주요 Metric은 Raw에서 재현했다.
[Raw 재현 근거](dense_ar_native_ttft_joint_slo_20261124/analysis/historical_main400_metric_reproduction.json)에 실행별 값이 있다.

## RQ1의 첫 구분: 연산 이전과 연산 제거

Prefill Split은 원래 필요한 계산을 다른 GPU로 이전한다.
현재 Completed-prefix Runtime은 P 결과 이후 D Continuation을 실행한다.
한 요청의 P와 D 계산이 모두 없어지는 방식이 아니다.
여러 요청 사이의 Pipeline 이득과 한 요청의 Serial 지연은 다른 개념이다.
P Queue 감소가 D의 추가 계산·대기·Handoff보다 커야 TTFT가 줄어든다.
이는 실행 의존성에 관한 설명이다. 그 크기는 아래 실측으로 구분한다.

Native도 D 첫 출력 준비가 필요하다.
해당 D 비용을 0으로 놓지 않았다.
First-token Source Audit에서 검사한 기존 Native와 Split은 모두 D 첫 출력 경로였다.
Native만 P 첫 Sample을 바로 전달해서 이겼다는 설명은 이 Family에서 반증됐다.

## 적격 Serial Kernel Control

`ISOLATED_KERNEL_MEASUREMENT`다.
별도 Frozen Qualification32 Prompt를 사용했다.
이 요청들은 Main400, Heldout192, Discovery208과 Request·Context가 겹치지 않는다.
Calibration32의 평균 Input은 10502.594 Tokens다. Main400은 13633.725 Tokens다.
두 분포가 같다고 가정하지 않는다.
원래 Serving API에서 실제 Batch1/1/1과 정확한 Output16을 사용했다.
다음 값은 각 요청의 관측 경계 안에서 CUDA Kernel Interval을 합집합으로 계산한 평균이다.
동시에 실행된 Kernel은 중복 합산하지 않는다.
원래 요청의 Metadata·Sampling Kernel도 해당 경계에 들어갈 수 있다.
Natural C3 요청별 전용 Forward 비용으로 대체하지 않는다.

각 Arm의 세 prospective Clean/Profile/Clean 묶음을 독립 실행했다.
TTFT p95, TTFT Mean, ITL p99, TPOT p95, Mean TBT의 간섭이 모두 5% 이내여야 적격이다.
수집 Counter의 Drop·Invalid도 검사했다.
Native·Layer·Hybrid는 각각 N3가 통과했다.
Stock은 첫 묶음의 ITL p99 비율1.201861 때문에 진단 자료로 분리했다.
Stock의 예정 후속 두 묶음은 중단했다. 유리한 결과가 나올 때까지 반복하지 않았다.

| Method | 적격 Start | P 경계 Host ms | P Kernel Union ms | D-first Host ms | D-first Kernel Union ms |
|---|---:|---:|---:|---:|---:|
| NativeStrong | 3 | 302.906 | 287.920 | 46.328 | 13.582 |
| Layer Last2 | 3 | 303.672 | 278.956 | 66.755 | 23.226 |
| Hybrid Last2 + Suffix1/8 | 3 | 260.387 | 237.939 | 110.187 | 63.326 |

P 경계는 Arrival→P Ready다.
D-first 경계는 D Submit→첫 Worker 출력이다.
D-first는 Continuation과 첫 출력 준비를 함께 포함한다.
순수 Suffix GEMM만의 시간이 아니다.
두 Host 경계 사이의 Coordinator 전송 경계와 최종 Client 전달은 이 표의 합계에 없다.

Native 대비 paired 차이는 다음과 같다.
단위는 ms다. 부호는 Candidate−Native다.

| 비용 | Layer 차이 [95% CI] | Hybrid 차이 [95% CI] |
|---|---:|---:|
| P Kernel Union | −8.963 [−9.175, −8.698] | −49.980 [−50.294, −49.591] |
| D-first Kernel Union | +9.643 [9.195, 10.066] | +49.744 [49.349, 50.150] |
| P + D-first Kernel Union | +0.680 [0.061, 1.352] | −0.237 [−0.517, 0.022] |
| P Host 경계 | +0.767 [−4.743, 5.295] | −42.519 [−47.663, −37.977] |
| D-first Host 경계 | +20.427 [17.637, 23.569] | +63.859 [59.020, 69.805] |

이 Control에서는 이전한 P Kernel 비용과 추가 D-first Kernel 비용이 거의 상쇄됐다.
Layer의 P Kernel 절약은 약8.963ms였다. P Host 경계의 절약은 확인되지 않았다.
Hybrid는 P Kernel을 약49.980ms 줄였다. D-first Kernel은 약49.744ms 증가했다.
이는 해당 Calibration32와 Batch1/Output16에서의 결과다.
Main400 Natural의 수초 Queue를 이 수치와 직접 합산하지 않는다.
P+D-first Kernel 합계도 TTFT가 아니다.

후속15개 Decode Token의 Host 구간 평균은 Native106.039ms, Layer106.098ms, Hybrid106.531ms였다.
이 값은 해당 Serial Control의 실제 Worker Emission 구간이다.
Natural C3의 ITL p99나 TPOT p95로 이름을 바꾸지 않는다.

Stock의 적격 GPU 절약량은 이 캠페인에서 확정하지 못했다.
진단 N1의 P Kernel213.024ms와 D-first Kernel78.145ms는 별도 CSV에 남겼다.
그 값을 N3 Primary 비교나 Native-relative 성공 판정에 사용하지 않는다.

[Kernel 분석 JSON](dense_ar_native_ttft_joint_slo_20261124/analysis/serial_native_relative_kernel_analysis_v2.json)에 Raw Hash, Clock Mapping, 실제 작업량과 모든 CPC 결과가 있다.
[요청별 비용 CSV](dense_ar_native_ttft_joint_slo_20261124/figures/serial_native_relative_request_phases.csv)와 [Category Partition CSV](dense_ar_native_ttft_joint_slo_20261124/figures/serial_native_relative_operator_partition.csv)를 함께 보존했다.
Kernel 이름으로 GEMM·Attention 등을 분류했다.
공통 GEMM 이름에서 Q/K/V Projection의 개별 시간을 추정하지 않았다.
No-observed-kernel 시간은 DMA, Physical Idle 또는 특정 Host 대기로 확정하지 않았다.

아래는 적격 N3의 Kernel-name Category 평균이다. 단위는 ms/request다.
Category가 겹친 구간은 별도 Partition으로 남겼다.
표에 없는 ROPE·KV Cache·Activation·Metadata 등의 시간도 원본 CSV에 있다.

| Method·경계 | GEMM | Attention | Fused RMSNorm·Residual | Separate RMSNorm | Unknown |
|---|---:|---:|---:|---:|---:|
| Native P | 199.997 | 64.380 | 7.852 | 0.081 | 3.851 |
| Layer P | 186.114 | 59.957 | 7.491 | 9.033 | 0.004 |
| Hybrid P | 162.730 | 46.449 | 6.525 | 7.926 | 0.000 |
| Native D-first | 10.493 | 1.831 | 0.313 | 0.007 | 0.517 |
| Layer D-first | 16.354 | 4.063 | 0.725 | 0.528 | 0.268 |
| Hybrid D-first | 38.497 | 17.971 | 1.536 | 1.800 | 0.233 |

Native와 Partial 경로는 Category 구성이 다르다.
특히 Native의 Unknown Category를 없는 비용으로 취급하지 않는다.
Fused Kernel의 내부 연산을 이름만으로 다시 나누지 않았다.
따라서 Separate RMSNorm 차이 전체를 Compiler의 독립적인 인과 손실로 확정하지 않는다.
이 표를 Natural C3의 Operator별 기여율로 사용하는 것도 금지한다.

같은 Serial Capture의 P Host 경계에서 시작한 CUDA Kernel Record도 셌다.
요청당 N3 평균은 Native768.563개, Layer787.063개, Hybrid742.500개였다.
이는 CUDA 실행 Record 수다. Host Launch API 호출 수나 Launch Overhead 시간이 아니다.
경계는 `[Arrival, P Ready)`로 고정했다.
D-first와 후속 Decode의 Record 수도 CSV에 별도로 보존했다.
Async 준비 작업이 Host Emission 경계를 넘을 수 있다.
따라서 D-first 비용을 순수 Suffix Forward만의 비용으로 바꾸지 않는다.

## 기존 Main400에서 회귀가 생긴 위치

아래 값은 `TRACE_ANALYSIS`다.
각 Start의 요청 평균을 구한 뒤 Start3회의 평균을 계산했다.
단위는 ms다.

| Method | Arrival→P Ready | P Ready→첫 Client | Mean TTFT |
|---|---:|---:|---:|
| NativeStrong | 3443.691 | 70.883 | 3514.573 |
| Stock Token | 864.162 | 4141.997 | 5006.159 |
| Layer | 3525.294 | 2578.503 | 6103.797 |
| Hybrid | 4423.103 | 2199.199 | 6622.302 |

Stock은 Native보다 P Ready 이전이 약2579.529ms 짧았다.
그러나 P Ready 이후는 약4071.115ms 길었다.
결과적으로 Mean TTFT는 약1491.586ms 증가했다.
이것은 평균 Host 경로의 정확한 위치 분해다. GPU 절약량의 인과 분해가 아니다.

Layer는 Native와 P Ready 이전 평균이 비슷했다. 차이는 약+81.603ms다.
반면 P Ready 이후는 약+2507.621ms였다.
따라서 Native 대비 Layer 회귀를 P Attention 하나로 설명할 수 없다.
Layer-vs-Stock에서는 반대로 P 이전 손해가 약2661.132ms였다.
D 이후 이득 약1563.494ms보다 컸다.
비교 기준이 다르면 설명해야 할 Critical Path도 다르다.

## 동일 Flag라도 Admission 기회는 같지 않다

설치한 Scheduler Source도 직접 확인했다.
Compute Deferral은 실행 중인 Decode 상태가 있을 때 Bulk Prefill을 미룬다.
Waiting 요청의 남은 Local Token이 1개보다 많으면 해당 Branch에서 대기한다.
Async Remote-KV Load는 별도 경로로 진행할 수 있다.
Native의 Last-prompt Query처럼 남은 Token이 1개인 경우는 그 Bulk 조건에 해당하지 않는다.
반면 Split Continuation은 여러 Token의 추가 Prefill이 필요하다.
따라서 같은 Throttle Flag가 같은 실행 기회를 보장하지 않는다.

Project Wrapper는 `prefill_capacity_bound` Escape도 비활성화한다.
이 설정은 Decode 보호를 위한 보수적인 정책이다.
GPU Kernel이 없는 짧은 구간마다 Prefill을 넣는 정책이 아니다.
현재 Scheduler의 논리적 Decode 상태와 Physical GPU Idle도 동일하지 않다.

Hybrid에는 정확한 2D 경계에서 Chunk를 자르는 기존 AST 변환도 있다.
KV Allocation 전에 두 `num_new_tokens` 지점만 처리한다.
Hybrid의 Source Audit Hash가 다른 이유를 이 변환 및 Dedent된 Source와 연결해 검사했다.
이를 예기치 않은 Dependency 변경으로 판정하지 않았다.
반대로 Hybrid의 실행 Scheduler가 Native와 Byte-for-byte 같다고 주장하지도 않는다.
기존과 새 실행의 Hash 및 변환 근거는 [설치 Scheduler Audit](dense_ar_native_ttft_joint_slo_20261124/references/installed_scheduler_predicate_audit.json)에 있다.

이 Source 근거는 대기가 가능한 인과 경로를 확인한다.
그 경로가 전체 TTFT 회귀 중 몇 ms를 만들었는지는 정책 개입과 구분해서 판정한다.
Source Predicate만으로 GPU가 포화됐거나 Safe Window가 존재한다고 결론내리지 않는다.

자세한 아홉 Stage와 READ/ACK 세부 경계는 [Request Trace 보고서](dense_ar_native_token_layer_request_trace.md)에 있다.
이 값들을 TTFT p95의 가법 분해로 사용하지 않는다.
관측한 Host Queue를 GPU Compute Stall로 이름 바꾸지 않는다.

## 가설 판정의 기준

새 Warm Runtime에서는 Uncapped Stock의 Native-relative TTFT 회귀가 확인되지 않았다.
Stock/NativeStrong p95 비율은0.94331 [0.75864, 1.35027]다.
반면 같은 Token Arm의 D Cap0→2 개입은 TTFT 비율1.41442 [1.10517, 1.67120]을 보였다.
Cap2 Token/NativeCommon2는1.24796 [1.18925, 1.31432]였다.
따라서 Native-relative 회귀를 Offloading의 보편적인 성질로 볼 수 없다.
D 정책과 실행 Geometry를 분리해야 한다.
새 Layer·Hybrid의 NativeStrong-relative 회귀는 각각1.24649,1.57147이었다.
각 CI 상한·하한과 전체 One-factor 결과는 [정책 보고서](dense_ar_pd_critical_path_controlled_interventions.md)에 있다.

| 가설 | 현재 확인한 근거 | 판정 범위 |
|---|---|---|
| H1 Full-model Compiler Advantage | Stock도 Full-model Compiled 경로다. Layer·Hybrid는 P Kernel을 실제 줄였다. | Compiler 하나의 독립 효과는 `INCONCLUSIVE`. 모든 회귀를 Partial Compiler로 설명할 수 없음. |
| H2 추가 D Continuation·Handoff | Serial N3에서 추가 D-first Kernel·Host 비용 관측. Main에서는 Post-P 경로 증가. | 추가 경로의 존재·비용은 지지됨. 전용 DMA 기여율은 `NOT_MEASURED`. |
| H3 P Queue 누적 | Layer-vs-Stock P 이전 Host 차이와 Source Window Feedback 관측. | Trace 위치는 지지됨. Main GPU Operator의 독립 기여율은 미확정. |
| H4 D Admission·Decode 간섭 | Stock·Layer·Hybrid의 D Accept→Prepare가 길었다. READ Call 자체는 약14–16ms였다. | D 대기 위치는 지지됨. KV 고갈·GPU 포화와 동일시하지 않음. |
| H5 Policy·Concurrency 차이 | 같은 Stock의 Cap0→2가 Main TTFT p95를1.41442배, Equal Replay를1.17105배로 늘렸다. | 검사한 D Cap의 정책 총 효과는 `SUPPORTED`. 모든 Axis 차이의 독립 기여율은 미확정. |

각 가설의 최종 수치와 반증 결과는 [Controlled Intervention 보고서](dense_ar_pd_critical_path_controlled_interventions.md)에 기록한다.

## 최종 원인 판정

전체 판정은 `PARTIAL_ROOT_CAUSE`다.
측정한 D Cap의 정책 총 효과는 반복 개입과 Equal-output Control에서 확인했다.
Bulk Continuation의 추가 의존성과 D 준비 대기도 확인했다.
반면 Main400의 요청별 GPU·DMA 전용 기여율은 식별하지 못했다.
Compiler 하나가 회귀의 원인이라는 설명도 확정하지 못했다.

같은 Stock의 Cap0→2에서 Main 평균 D Accept→Prepare는2638.049ms 증가했다.
Equal Replay에서도1107.700ms 증가했다.
Replay의 P Ready 이전 변화는−3.826ms였다.
이 조건에서는 P 연산량 차이가 없어도 D 정책이 TTFT를 악화시켰다.
그러나 이 Stage 전체를 Cap의 독립적인 전용 인과 지연으로 부르지 않는다.
정책은 이후 Batch와 Queue도 바꿨다.

Registration 지연만 해제하면 D Submit→Accept가 줄었다.
그만큼 D Accept→Prepare가 늘어 총 TTFT는 개선되지 않았다.
예를 들어 Replay Native의 두 변화는−645.657ms와+639.380ms였다.
앞 Queue를 없애는 것과 전체 Critical Path를 줄이는 것은 달랐다.
검증한 Safe Window 없이 Compute Throttle을 제거하지 않았다.

새 Stock/NativeStrong의 Main TTFT CI는1을 포함한다.
Replay에서도1.33816 [0.96813, 1.85331]로1을 포함한다.
따라서 모든 Prefill Offload가 Native보다 항상 느리다는 명제는 지지하지 않는다.
새 Layer와 Hybrid의 회귀는 두 Family에서 모두 확인됐다.
이 결론은 검사한 Cut·정책·Workload에 한정한다.
기존 v2와 새 Warm Runtime 사이의 전체 차이는 One-factor 실험이 아니다.
공통 GC·I/O 설정과 Source Snapshot 차이는 [Runtime Bridge Audit](dense_ar_native_ttft_joint_slo_20261124/analysis/historical_current_runtime_bridge_audit.json)에 명시했다.

## 개선 여지의 해석

요청별 Host Envelope를 제거한 Frozen-path Oracle도 계산한다.
원래 Cohort 길이, ITL, 출력과 이후 Queue를 고정한다.
이 Oracle은 구현 가능한 Speedup이나 전역 최적 상한이 아니다.
실제 개선 가능성은 정합성과 Decode Gate를 통과한 정책의 독립 실측으로만 판정한다.
Native보다 빠른 점이 Oracle에만 있으면 실현 가능한 Serving Headroom으로 보고하지 않는다.
[단계 제거 CSV](dense_ar_native_ttft_joint_slo_20261124/figures/frozen_path_stage_oracles.csv)와 [Native 목표 제거량 CSV](dense_ar_native_ttft_joint_slo_20261124/figures/frozen_native_target_stage_removals.csv)에 가정을 보존했다.

다음은 기존 Main400에서 선택한 Host 경계를 전부 제거한 Metric 진단이다.
각 요청의 이후 Queue, 출력, ITL과 원래 측정 기간은 그대로 둔다.
단위는 Start-mean이다. 복구 요청 수의 분모는 실행당400개다.

| Method·가상 제거 경계 | 추가 Joint 통과 요청 수 | 가상 SLO tok/s | 해석 |
|---|---:|---:|---|
| Stock D Accept→Prepare 전체 | 133.667 | 107.759 | READ·Scheduler·Queue를 함께 없앤 Host 경계 Oracle |
| Layer D Accept→Prepare 전체 | 81.000 | 68.643 | 같은 경계 Oracle |
| Layer P Submit→Prepare 전체 | 74.333 | 68.631 | P Host Admission 경계 Oracle |
| Hybrid Arrival→P Submit 전체 | 180.333 | 97.342 | Coordinator 경계 Oracle |

이는 Stage 개선의 우선순위를 찾는 자료다. 실제 복구량이 아니다.
특히 Credit 제약을 지우면 다른 Queue와 Memory가 그대로 유지되지 않을 수 있다.
전체 D Post-P 경계를 가상 제거해도 Hybrid는 세 Start 모두 Native TTFT p95에 도달하지 못했다.
해당 Hybrid의 P 이전 경로도 함께 바꿔야 한다는 Frozen-trace 조건이다.
반대로 Layer·Stock의 모든 Post-P 제거는 세 Start에서 Native 목표에 도달했다.
필요 제거율은 Start마다 다르다. 이를 공통 구현 가능한 절감률로 해석하지 않는다.
실현 가능한 Native 초과 이득은 아래 정책 개입의 품질·Decode Gate로 별도 판정한다.

# Layer × Token 2D: Dynamic Scheduling의 실행 범위와 Gate

상태: `HOLD_DECODE_SAFE_WINDOW_NOT_IDENTIFIED`.
검증되지 않은 Active-decode 삽입은 실행하지 않는다.
Static 성능 확인과 Safe Window 검증은 다른 Gate다.

## 구현한 기능과 구현하지 않은 기능

| 기능 | 상태 | 증거 수준 |
|---|---|---|
| Pure Layer/Token 및 Genuine 2D의 정합한 Continuation | 별도 N8/N32/N400 Gate | `CORRECTNESS_ONLY` |
| Active Decode가 있을 때 큰 Prefill 계산 지연 | 구현 및 Natural32/400 검증 | `CORRECTNESS_ONLY` |
| Active Decode 중 큰 Remote READ Setup 지연 | 구현 및 Natural32/400 검증 | `CORRECTNESS_ONLY` |
| Active Decode 중 새 Request 등록 지연 | 별도 Source에서 Natural32/400 검증 | `CORRECTNESS_ONLY` |
| 고정 Cut의 Continuous Online 성능 | Discovery와 독립 Confirmation을 구분 | `MEASURED_ONLINE` |
| 다음 Decode Deadline을 예측하는 Window Predictor | `NOT_RUN` | `NOT_RUN` |
| 안전 Budget에 따른 Online Dynamic Split-size | `NOT_RUN` | `NOT_RUN` |
| Online Layer/Token/Hybrid Selector | `NOT_RUN` | `NOT_RUN` |

계산 지연 정책은 다음 Decode가 없을 때 Prefill 실행을 허용한다.
이는 보수적인 Admission이다. 미래의 실행 가능한 Window 길이를 예측하는 Scheduler가 아니다.
고정 Cut을 대기시켰다는 사실만으로 Dynamic Size 또는 Dynamic Axis 이득을 주장하지 않는다.

### Host 등록 Queue를 별도로 계산한다

후속 Runtime은 Output이 시작된 D Request가 있으면 새 Request 등록을 미룬다.
이 정책은 `engine.add_request` 이전의 Host 처리도 Decode 제출을 지연할 수 있다는 관측을 검사한다.
P/D Model, Client Timestamp와 SLO는 변경하지 않는다.
IPC 수신과 기존 Decode Step 및 NIXL ACK는 계속 처리한다.

미룬 Request는 vLLM Scheduler에 아직 등록되지 않았을 수 있다.
따라서 빈 Scheduler Queue만으로 D Admission 대기가 없다고 판단하지 않는다.
별도 Host 등록 Queue의 길이와 Receive→Release 대기를 원시 Event에서 재구성한다.
이 구간은 앞선 Coordinator 또는 IPC 대기를 포함하지 않는다.

이 정책도 미래 Window 길이를 예측하지 않는다.
Admission 시점이 바뀌면 이후 Batch도 바뀔 수 있다.
독립 Start의 ITL/TPOT 개선을 GPU-exclusive 비용 감소로 자동 해석하지 않는다.
원래 동결한 Main, READ-only Repair와 등록-only 후속 검사는 서로 다른 결과로 보존한다.
새 정책의 정합성 통과를 Dynamic Size 또는 Dynamic Axis의 성능 통과로 표시하지 않는다.

## Physical Inactivity와 Executable Window

Kernel Capture가 보이는 Inactivity는 모든 GPU Engine의 Physical Idle과 다르다.
Kernel과 Copy를 함께 관측해도 모든 Engine의 Idle을 보장하지 않는다.
NIXL Host Enclosure와 Kernel Gap이 겹친다는 사실도 순수 DMA 비용의 인과 증명이 아니다.

유효한 Dynamic 후보에는 다음 조건이 필요하다.

1. Partial KV와 Hidden/Residual이 준비돼야 한다.
2. Transfer와 Import가 완료돼야 한다.
3. KV Credit과 Admission이 가능해야 한다.
4. 비선점 Forward의 Tail Cost와 Control Overhead가 Budget 안에 들어가야 한다.
5. 새 Decode-ready 요청의 위험과 Prediction Error를 포함해야 한다.
6. Native-relative Decode Gate와 Absolute SLO를 모두 지켜야 한다.

Native Trace에 준비된 Partial State가 없다면 Gap 길이만으로 Safe-fit Rate를 계산하지 않는다.
현재 Native Activity 재검증은 Clean/Profile/Clean을 독립 3회 검사한다.
Full-trace Metric과 ROI에 노출된 같은 Request ID의 Metric을 함께 검사한다.
Observer CI가 ±5%를 벗어나면 해당 Capture는 `DIAGNOSTIC`이다.
원래 Capture의 실패를 새 Capture의 통과로 지우지 않는다.

공통 Compute/READ Native의 QPS1.8과 2.8 재검증은 각각 3개 Triplet에서 Observer Gate를 통과했다.
P와 D의 같은 Host-time ROI 교집합으로 다시 계산했다.
QPS1.8의 Kernel Busy 평균은 P 87.059%, D 36.692%였다.
QPS2.8에서는 P 99.520%, D 49.028%였다.
같은 ROI의 P Waiting 평균은 각각 1.385개와 20.921개였다.
D Decode-ready Waiting 평균은 두 조건 모두 0개였다.
D Active 평균은 각각 0.390개와 0.548개였다.

이는 같은 짧은 ROI에서 P Backlog와 낮은 D Kernel Busy가 함께 나타난 근거다.
전체 Trace의 Sustainable Capacity나 미래 Safe Window를 입증하지는 않는다.
특히 Native Trace에는 아직 수락 가능한 Partial KV/Hidden State가 없다.
그 Gap에 들어갈 실제 후보가 준비됐다고 가정해 Safe-fit Rate를 만들지 않는다.

근거: [공통 ROI Activity](dense_ar_longbench400_2d_20261121/analysis/recovered_native_shared_window_activity_v1.json).

### 관측한 D Kernel Gap

다음 값은 같은 공통 ROI의 내부 Kernel Gap이다. 경계에서 잘린 Gap은 제외했다.
Copy와 Control Activity가 이 Gap 안에 있을 수 있다. 따라서 모든 Engine의 Physical Idle은 아니다.
각 행은 독립 Profile Start 1개다. Gap 수를 독립 Start 수로 사용하지 않는다.

| QPS | Profile Start | Gap 수 | p50(ms) | p90(ms) | p99(ms) | 최대(ms) |
|---:|---:|---:|---:|---:|---:|---:|
| 1.8 | 0 | 117184 | 0.000096 | 0.000160 | 0.013536 | 1178.178026 |
| 1.8 | 1 | 115262 | 0.000096 | 0.000160 | 0.007840 | 1942.636155 |
| 1.8 | 2 | 101763 | 0.000096 | 0.000160 | 0.013440 | 1165.390966 |
| 2.8 | 0 | 144254 | 0.000096 | 0.000160 | 0.007840 | 1160.015189 |
| 2.8 | 1 | 140454 | 0.000096 | 0.000160 | 0.007727 | 834.942951 |
| 2.8 | 2 | 159068 | 0.000096 | 0.000160 | 0.007872 | 837.194730 |

짧은 Kernel 사이 간격과 드문 큰 간격은 같은 현상이 아니다.
최대 Gap을 평균적인 실행 기회나 예측 가능한 미래 Budget으로 사용하지 않는다.
그 시점에 Partial State가 준비됐는지도 별도로 확인해야 한다.
따라서 위 분포에서 측정된 Layer/Token Safe-fit Rate를 계산하지 않았다.
Safe Window는 `NOT_IDENTIFIED`다. Online Dynamic Insertion은 `NOT_RUN`이다.

분포 Figure는 [D Kernel Gap CDF](dense_ar_longbench400_2d_20261121/figures/recovered_native_observed_kernel_gap_distribution.png)에 있다.

## 상대 5% Gate가 만드는 제한

Absolute Every-ITL SLO 34ms와 Native-relative ITL p99 5%는 다른 조건이다.
예를 들어 Native p99가 9ms라면 상대 기준의 Point 값은 9.45ms다.
이 예시는 `STRUCTURAL_ACCOUNTING`이다. 측정된 Prefill Budget이 아니다.
34−9=25ms를 추가 Prefill의 안전 Budget으로 사용할 수 없다.

Active Decode의 많은 Interval에 긴 Prefill이 들어가면 p99가 증가할 수 있다.
매우 드문 삽입과 빈번한 삽입도 구분해야 한다.
실제 Rate와 Client-visible 분포를 측정하기 전에는 이를 Safety PASS라고 부르지 않는다.
GPU Stream Priority만으로 비선점 Kernel 간섭이 사라진다고 가정하지 않는다.

## C0 State-updating Replay

Calibration은 Serial208-action Start 3회와 Native Serial Pilot 16개를 사용한다.
이는 880개 관측값이다. 서로 다른 Prompt 880개나 독립 Server Start 880회가 아니다.
Model은 P/D Queue, Credit 반환과 Completion을 갱신한다.
Request별 최소 비용을 Frozen State에서 독립적으로 고르는 Oracle은 사용하지 않는다.

Held-out C0 Validation 432개 고유 Server Start에서 TTFT 오차를 검사했다.
TTFT Median APE는 약 3.52%, p90 APE는 약 10.55%였다.
이 TTFT Gate만으로 ITL Tail이나 Joint SLO를 검증했다고 주장하지 않았다.

별도 Tail Replay는 실제 Client ITL Template를 사용한다.
Held-out TTFT Median/p90 APE는 약 3.301/11.371%였다.
ITL p99 오차는 약 2.954/6.143%였다. TPOT p95 오차는 약 3.249/8.974%였다.
이 검사는 선언한 C0 Replay 허용 오차를 통과했다.
그러나 C3 Continuous Batching의 Online Admission을 승인한 것은 아니다.

근거: [TTFT Validation](dense_ar_longbench400_2d_20261121/analysis/c0_queue_model_heldout_validation_v1.json),
[Tail Validation](dense_ar_longbench400_2d_20261121/analysis/c0_tail_replay_heldout_validation_v1.json).

## 제한된 Offline Oracle

25개 Action은 Native, Pure Layer, Pure Token과 Interior Hybrid를 포함한다.
4개 Input Bucket × Controlled Output16/64 × Arrival Trace 3개를 검사한다.
탐색은 제한된 Coordinate Search다. 전역 최적 상한이 아니다.
Whole-prefill Deflection은 이 Action 집합에 없다.

최대 Workload-mean 예측 SLO-goodput 비율은 약 1.04418이었다.
해당 Workload는 I3/Output64다. 개별 Trace 비율은 약 1.0959/1.0071/1.0241이었다.
따라서 개별 Trace의 큰 값을 전체 평균 개선으로 표시하지 않는다.
이 값은 `OFFLINE_ORACLE`이다. Online Speedup이 아니다.

16개 Monte Carlo Draw도 독립 Server Start 16회가 아니다.
Trace별 Min/Max를 95% CI로 표시하지 않는다.
Calibration 불확실성은 이 Oracle의 개선 수치에 완전히 전파하지 않았다.
새 Proper-pipeline-load Regime과 C3 Runtime에서도 검증하지 않았다.

Oracle의 Best Fixed는 Model이 Discovery Trace에서 선택한 후보다.
실제 GPU 반복 측정으로 확인한 최강 Static을 기준으로 삼은 것은 아니다.
Oracle의 Pure Token Cut은 Last2/4/8/12에 해당한다.
별도 25-action GPU Discovery의 Token Fraction은 1/16, 1/8, 1/4, 1/3이다.
두 탐색 공간의 모든 Boundary가 같지는 않다.
따라서 1.04418을 가장 강한 실측 Static 대비 Headroom으로 표현하지 않는다.

검증된 평균 5% 추가 Headroom은 아직 확보하지 못했다.
이는 가능한 모든 Dynamic Policy의 상한이 5% 미만이라는 뜻이 아니다.
근거: [제한된 Oracle 결과](dense_ar_longbench400_2d_20261121/analysis/c0_bounded_axis_hybrid_oracle_v1.json).

## 최소 Knob의 현재 범위

실측을 해석할 때 중요한 후보는 다음 세 가지다.

1. P/D Effective Service Balance.
2. Source Credit와 Backpressure.
3. B8192에서 필요한 D Query Chunk 수.

이 후보는 설명 변수다. 검증된 Online Policy는 아니다.
기존 전체 Replay는 명확한 C0 Winner 38개 중 32개를 예측했다.
최대 Latency Regret은 11.530%였다. 단일 Knob으로 충분하다는 근거는 아니다.
관련 반례는 [Axis Cost Model](dense_ar_longbench400_axis_cost_model.md)에 있다.

추가 단순 Demand Rule에서는 `max(P,D)`가 명확한 Winner 38개 중 34개를 맞췄다.
Serial `P+D`는 16개를 맞췄다. 이 비교는 기존 Matrix 관측 후의 탐색적 분석이다.
최대 Regret은 Bottleneck Rule에서 13.863%였다.
이 Score는 알려진 Controlled Output과 여러 Calibration 입력을 압축한다.
따라서 단일 직접 관측 변수로 Natural EOS C3를 예측했다고 주장하지 않는다.
더 높은 Cell Accuracy를 Dynamic Serving 이득으로 바꾸지도 않는다.

## 다음 GO 조건

먼저 새 Source의 Full400 정합성과 강한 Static 비교를 완료한다.
Unchanged Native, 공통 D 정책 Native와 더 나은 Native Credit 설정을 구분한다.
Layer/Token/Hybrid는 실제 독립 Start CI로 Decode Gate를 통과해야 한다.

그 뒤 실제 준비된 State를 가진 요청에서 미래 Window Predictor를 검증해야 한다.
Candidate Tail과 Prediction Error를 함께 포함해야 한다.
Best Static 및 Window-aware Fixed 후보 대비 추가 이득도 별도로 확인해야 한다.

원래 Held-out200/Main400 Confirmation은 모두 완료했다.
후속 등록 수정도 Natural400/QPS1.8과2.8의 독립 Start3회를 각각 완료했다.
후속 Layer, Token과 Hybrid는 그 두 조건에서 Decode 보호를 회복했다.
그러나 Hybrid는 강한 Fixed Stock Token보다 TTFT와 Goodput에서 우수하지 않았다.
안전한 미래 Window와 준비된 State의 Fit은 여전히 검증하지 못했다.

따라서 Online Dynamic Size와 Axis Selector는 `NOT_RUN`으로 남긴다.
C0 Hybrid의 작은 Iso-work 이득을 C3 Dynamic 이득으로 옮기지 않는다.
현재 판정은 `HOLD_DECODE_SAFE_WINDOW_NOT_IDENTIFIED`다.
이는 모든 미래 Dynamic Policy의 이득이 없다는 증명은 아니다.

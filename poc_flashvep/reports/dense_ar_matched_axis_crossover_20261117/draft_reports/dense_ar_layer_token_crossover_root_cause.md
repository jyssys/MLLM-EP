# Layer/Token Crossover 원인 분석

진행 중 초안이다. 완료하지 않은 조건과 최종 판정을 구분한다.

## 공통 실행 조건

Model: Qwen/Qwen3-8B. Revision: `b968826d9c46dd6066d109eabc6255188de91218`.
Original BF16, Exact Causal Attention, Greedy, Thinking Disabled, Full36 Decode를 유지한다.
Input8000과 Output16/32/64/128/200을 사용한다. `ignore_eos=True`이다.
GPU0/1은 H100 SXM80GB이다. KV Budget은 GPU당42GiB이다.
GPU UUID는 `GPU-f217c8a0-1142-20f4-d84b-af29f3a47a0d`와 `GPU-a77f3471-67d4-20b0-9fab-e502d4de5adb`이다.
Runtime은 vLLM0.26.0+cu129, PyTorch2.11.0+cu129, NIXL1.4.1이다.
모든 Worker에 B8192/S1024를 적용한다. Prefix Cache는 비활성화한다.
PD의 공통 D Admission 상한은8이다. 실제 준비 Group과 Decode Cohort는 다를 수 있다.
실제 모델 Context Limit은40960이다. Worker의 실행 한도는 Input+Output에서 정한다.
따라서 Output16과 Output200의 Worker 한도는 각각8016과8200이다.
같은 Output에서는 모든 비교 방법의 한도가 같다.

Branch: `flashvep/dense-ar-matched-axis-crossover-20261117`.
Publication Parent: `228a041f35cedcee9f45beba6c462b453fb3a2c5`.
Parent SHA를 전체 실행 Source SHA라고 표시하지 않는다.
각 실행의 `command.json`에 실제 `source.zip` SHA256을 저장한다.
Worker별 실제 설정, Prompt Hash, Seed 및 설정 Hash도 원시 Evidence에 저장한다.
[environment.json](../state/environment.json)에 GPU와 Model 설정을 보존한다.
[frozen_new_inference_sources_v3.json](../state/frozen_new_inference_sources_v3.json)에 동결한 Inference Source Hash를 보존한다.

## 표본과 지표

성능 비교는 실제 문서144개를 사용한다. 동일 Prompt를 독립 Start로 세지 않는다.
기존 Held-out128개에 기존 Discovery의 다른 문서16개를 추가했다.
새로운 미관측 Prompt Pool이라고 주장하지 않는다.
Discovery Seed는2026111780이다.
QPS6 Confirmation Seed는2026111790/2026111791/2026111792이다.
Discovery와 Confirmation은 CI 계산에서 합치지 않는다.
CI의 재표본 단위는 독립 Server Start이다. n1에는 CI가 없다.
표의 p95/p99는 Start 안에서 계산한 분위수를 Start 사이에서 평균한 값이다.
기존 CI는 Bootstrap Draw 값의 Type7 선형 보간을 유지한다.
별도 역CDF 분위수 민감도 분석도 같은 독립 Start에 적용한다.
두 방식에서 결론이 바뀌면 확정적인 우위나 비열등성으로 판단하지 않는다.
표는 표시용으로 반올림한다. 반올림하지 않은 값과 CI는 JSON에 보존한다.
Raw Output tokens/s는 전체 유한 Cohort의 Arrival-to-drain 처리율이다.
이를 Maximum Sustainable Capacity라고 표시하지 않는다.
Client-visible ITL에는 첫 Token Interval을 포함한다.
Worker Notice 시간과 Client Delivery 시간은 별도로 보존한다.

현재 집계한 Discovery는172/232회이다.
현재 집계한 Confirmation은0/120회이다.
표에 없는 예정 조건은 `NOT_RUN`이다.
상태 Matrix는 [full_execution_matrix.csv](../analysis/full_execution_matrix.csv)에서 확인한다.
개별 요청은 [per_request_metrics_and_work.csv](../analysis/per_request_metrics_and_work.csv)에서 확인한다.
독립 Start 통계는 [independent_start_statistics.json](../analysis/independent_start_statistics.json)에서 확인한다.

## SLO 계약

기존 Output별 Frozen SLO를 변경하지 않았다.
Output200 Balanced는 TTFT5000ms, 모든 ITL34ms, TPOT33ms이다.
Output Sweep의 공통 보조 SLO도5000/34/33ms이다.
기존 Output64 Balanced의 ITL35ms와 공통 보조34ms를 구분한다.
Goodput은 세 조건을 모두 통과한 요청의 Output Token 수를 분자로 사용한다.
Reference Goodput이0이면 비율은 정의하지 않는다.
Native-relative 보호는 ITL p99와 TPOT p95 비율의95% CI 상한을 검사한다.
두 상한이 모두1.05 이하여야 PASS이다. 최소3개 독립 Start가 필요하다.
Absolute SLO와 Relative 비열등성은 서로 다른 Gate이다.
[frozen_SLO_regimes.json](../configs/frozen_SLO_regimes.json)와 [common_output_sweep_SLO.json](../configs/common_output_sweep_SLO.json)을 보존한다.

## 외부 중단과 재개

제어 세션 재시작으로 중단한 실행1건은 FAILED로 보존한다.
원래 Manifest Slot은 유지하고, name만 다른 대체 실행을 연결한다.
중단한 부분 결과는 독립 표본이나 Clean 성능에 포함하지 않는다.
완료한 기존 GPU 실행은 반복하지 않는다.
[복구 기록](../state/daemon_recovery_20261118.json)에 원래 Receipt Hash와 대체 이름을 저장한다.

## 실행 Geometry와 해석 범위

B8192는 Iteration Token Budget이다. S1024는 Scheduler Sequence 상한이다.
둘 다 실제 GPU Batch 수를 뜻하지 않는다.
현재 D Runtime은 준비 Group과 이후 Decode Cohort를 결합한다.
새 준비 Group은 이전 Decode Cohort가 종료한 뒤 수용한다.
Layer의 Prompt Query8000개 때문에 준비 Group은1개 요청이다.
Token≈889/1778/2667의 준비 Group은 각각 최대8/4/3개 요청이다.
Native의 Cached Query는1개이며 공통 상한8개 요청을 사용할 수 있다.
각 Group의 실제 크기는 Ready-work와 종료 경계에 따라 더 작을 수 있다.
따라서 공통 Admission Policy를 사용했지만 실제 Geometry는 일치하지 않는다.
이 비교를 `GEOMETRY_MATCHED`라고 표시하지 않는다.

이 구조는 현재 구현의 제한이다.
Layer-axis가 원리적으로 Batch1만 사용할 수 있다는 뜻은 아니다.
여러 Iteration에서 Layer 요청을 준비한 뒤 큰 Decode Cohort로 합치는 경로는 구현하지 않았다.
이번 캠페인에서 이 실행 구조를 조용히 변경하지 않는다.
P의 외부 Queue 증가만으로 P Compute Saturation을 주장하지 않는다.
D Reservation, Source Window32 및 Cohort Backpressure가 P 제출을 제한할 수 있다.

## 같은 GEMM Rows와 다른 Attention 분배 — STRUCTURAL_ACCOUNTING

| Last | Token Suffix Histogram (N144) | D GEMM Rows / method | D Token/Layer causal Attention pairs | Extra Layer Hidden/Residual bytes |
| --- | --- | --- | --- | --- |
| 4 | {"880": 64, "896": 80} | 4608000 | 1.888769 | 18874368000 |
| 8 | {"1776": 128, "1792": 16} | 9216000 | 1.777679 | 18874368000 |
| 12 | {"2656": 48, "2672": 96} | 13824000 | 1.666581 | 18874368000 |

전체 모델의 Causal Attention Pair 합은 두 방법에서 같다.
P와 D의 작업 분배는 다르다.
Token/Layer Attention Pair 비율을 GPU 시간 비율로 바꾸지 않는다.
Layer는 각 요청에 Hidden/Residual131072000bytes를 추가 전송한다.
정확한 N144 Matched Plan에서는 유효 KV Byte 합이 두 방법에서 같다.
이 값은 Source KV Credit 점유량이 같다는 뜻이 아니다.

## Admission 통제 — MEASURED

기존 공통 cap8 Matrix를 유지한다.
별도 cap1 실험에서 두 방법의 실제 Decode Cohort Geometry 차이를 줄인다.
Last4/8은 Output16/200에서 각1개 Start이다.
Last12는 사전 선정한 Seed2026111787/1788/1789에서 확인한다.
동일한 Seed의 Native cap1/cap8도 비교한다.
독립 반복을 끝내지 않은 결과에는 확정적 원인 판정을 적용하지 않는다.

| Last | Output | Admission cap | Starts | Layer/Token TTFT p95 [95% CI] |
| --- | --- | --- | --- | --- |
| 4 | 16 | 1 | 1 | 0.972260 [CI: NOT_MEASURED] |
| 8 | 16 | 1 | 1 | 1.029545 [CI: NOT_MEASURED] |
| 12 | 16 | 1 | 3 | 0.974817 [0.956518, 0.999547] |
| 4 | 16 | 8 | 1 | 2.017273 [CI: NOT_MEASURED] |
| 8 | 16 | 8 | 1 | 3.038383 [CI: NOT_MEASURED] |
| 12 | 16 | 8 | 3 | 3.125943 [2.793851, 3.570556] |
| 4 | 200 | 1 | 1 | 0.998433 [CI: NOT_MEASURED] |
| 8 | 200 | 1 | 1 | 0.996151 [CI: NOT_MEASURED] |
| 12 | 200 | 1 | 3 | 0.997022 [0.993398, 1.001305] |
| 4 | 200 | 8 | 1 | 7.538413 [CI: NOT_MEASURED] |
| 8 | 200 | 8 | 1 | 3.830243 [CI: NOT_MEASURED] |
| 12 | 200 | 8 | 3 | 2.992460 [2.970642, 3.011819] |

## Host Critical Path — TRACE_ANALYSIS

Arrival → P Host Submit → Source Ready Notice → D Host Submit → First Worker Notice → First Client Output을 추적한다.
이 다섯 구간의 합은 해당 요청의 TTFT와 일치해야 한다.
평균 구간 차이는 Host Clock의 분해이다. Exclusive GPU Compute 원인으로 표시하지 않는다.
p95 구간을 더해서 TTFT p95를 설명하지 않는다.
자기 요청의 첫 Token 이후 Decode는 자기 TTFT에 포함되지 않는다.
이전 Cohort의 Decode Residency는 다음 요청의 D 대기를 늘릴 수 있다.
Output Length는 이 대기 증폭을 바꿀 수 있다.
[root_cause_components.json](../analysis/root_cause_components.json)에 같은 Start의 구간을 저장한다.

| Method | Output | Starts | D prep ms / request | Decode residency ms / request | D Host-span rate ceiling req/s | Observed finite-cohort req/s |
| --- | --- | --- | --- | --- | --- | --- |
| Native0 | 16 | 1 | 24.182 | 269.499 | 3.405 | 3.368 |
| Layer4 | 16 | 1 | 48.539 | 352.408 | 2.494 | 2.478 |
| Token4 | 16 | 1 | 62.784 | 200.063 | 3.804 | 3.770 |
| Layer8 | 16 | 1 | 77.294 | 353.391 | 2.322 | 2.306 |
| Token8 | 16 | 1 | 92.820 | 141.738 | 4.263 | 4.224 |
| Layer12 | 16 | 1 | 122.651 | 353.285 | 2.101 | 2.088 |
| Token12 | 16 | 1 | 127.343 | 121.952 | 4.011 | 3.982 |
| Native0 | 32 | 1 | 19.098 | 278.976 | 3.355 | 3.322 |
| Layer4 | 32 | 1 | 46.782 | 727.357 | 1.292 | 1.286 |
| Token4 | 32 | 1 | 55.586 | 215.021 | 3.695 | 3.668 |
| Layer8 | 32 | 1 | 76.072 | 727.223 | 1.245 | 1.240 |
| Token8 | 32 | 1 | 87.165 | 190.629 | 3.600 | 3.575 |
| Layer12 | 32 | 1 | 108.947 | 729.157 | 1.193 | 1.189 |
| Token12 | 32 | 1 | 124.651 | 246.673 | 2.693 | 2.680 |
| Native0 | 64 | 1 | 26.163 | 285.310 | 3.211 | 3.181 |
| Layer4 | 64 | 1 | 55.377 | 1477.145 | 0.653 | 0.651 |
| Token4 | 64 | 1 | 58.282 | 219.514 | 3.600 | 3.569 |
| Layer8 | 64 | 1 | 78.582 | 1475.198 | 0.644 | 0.642 |
| Token8 | 64 | 1 | 93.713 | 378.168 | 2.119 | 2.109 |
| Layer12 | 64 | 1 | 109.033 | 1476.017 | 0.631 | 0.630 |
| Token12 | 64 | 1 | 120.629 | 497.810 | 1.617 | 1.612 |
| Native0 | 128 | 1 | 24.467 | 401.065 | 2.350 | 2.334 |
| Layer4 | 128 | 1 | 65.629 | 2976.573 | 0.329 | 0.328 |
| Token4 | 128 | 1 | 58.266 | 401.407 | 2.175 | 2.165 |
| Layer8 | 128 | 1 | 94.371 | 2978.007 | 0.325 | 0.325 |
| Token8 | 128 | 1 | 88.268 | 763.864 | 1.174 | 1.171 |
| Layer12 | 128 | 1 | 124.413 | 2981.211 | 0.322 | 0.322 |
| Token12 | 128 | 1 | 128.653 | 1006.777 | 0.881 | 0.879 |
| Native0 | 200 | 1 | 21.590 | 627.953 | 1.540 | 1.533 |
| Layer4 | 200 | 1 | 48.362 | 4651.269 | 0.213 | 0.213 |
| Token4 | 200 | 1 | 61.408 | 629.050 | 1.448 | 1.444 |
| Layer8 | 200 | 1 | 77.242 | 4666.520 | 0.211 | 0.211 |
| Token8 | 200 | 1 | 91.094 | 1193.924 | 0.778 | 0.777 |
| Layer12 | 200 | 1 | 124.127 | 4670.254 | 0.209 | 0.208 |
| Token12 | 200 | 1 | 128.637 | 1581.956 | 0.585 | 0.584 |

D Host-span Rate Ceiling은 요청 수를 서로 겹치지 않는 D Host Span 합으로 나눈 값이다.
여기에는 Host와 Transfer 대기도 포함된다.
이는 해당 실행의 시간 회계 상한이다. 물리적 GPU 처리 용량이나 Sustainable Capacity가 아니다.
실제 Group 크기가 달라지면 요청당 Decode Residency의 분모도 달라진다.
Raw 처리율과 이 상한의 차이는 Idle, 경계 및 Host Gap도 포함한다.
이를 Exclusive GPU Compute 효율로 표시하지 않는다.

## GEMM과 Handoff의 증거 한계

고립 실험의 원래 Layer0 Projection Median 합은 구조적 성분 추정이다.
이를 실제 전체 Suffix 실행시간으로 표시하지 않는다.
별도 Last2/Last4 Serial Diagnostic은 실제 P→D Continuation의 CUDA Event Enclosure를 측정했다.
그 측정에는 Attention, Norm, Activation 및 Launch/Stream 간격도 포함된다.
그 측정을 순수 GEMM 시간이나 QPS Sweep의 TTFT 이득으로 대체하지 않는다.
NIXL Telemetry와 제출 Byte는 실제 기록이다.
동시 Transfer Duration을 더해서 요청별 Wire Time으로 표시하지 않는다.
Handoff, Admission 및 Host Scheduling 비용을 구분한다.

## Profiler와 Crossover 판정

온라인 Profiler의 Clean 대비 전체5% 간섭 Gate를 통과하지 못했다.
Raw Failure와 별도 사후 PID 매핑을 보존한다.
개선한 API 배치 Pilot도 Maximum ITL Screen을 통과하지 못했다.
따라서 Kernel Timeline은 `DIAGNOSTIC`이다.
신뢰 가능한 물리적 P Busy, D Idle 및 SM/HBM Counter는 `NOT_MEASURED`이다.
Host State를 물리적 GPU Idle Budget으로 표시하지 않는다.

| Stage | Last | QPS | Admission | 판정 | Layer favored Outputs | Token favored Outputs |
| --- | --- | --- | --- | --- | --- | --- |
| admission_control | 4 | 6.0 | 1 | INCONCLUSIVE | [] | [] |
| admission_control | 8 | 6.0 | 1 | INCONCLUSIVE | [] | [] |
| admission_control | 12 | 6.0 | 1 | LAYER_FAVORED_ON_CONFIRMED_POINTS | [16] | [] |
| admission_control | 4 | 6.0 | 8 | INCONCLUSIVE | [] | [] |
| admission_control | 8 | 6.0 | 8 | INCONCLUSIVE | [] | [] |
| admission_control | 12 | 6.0 | 8 | TOKEN_FAVORED_ON_CONFIRMED_POINTS | [] | [16, 200] |
| discovery | 4 | 2.0 | 8 | INCONCLUSIVE | [] | [] |
| discovery | 8 | 2.0 | 8 | INCONCLUSIVE | [] | [] |
| discovery | 12 | 2.0 | 8 | INCONCLUSIVE | [] | [] |
| discovery | 4 | 4.0 | 8 | INCONCLUSIVE | [] | [] |
| discovery | 8 | 4.0 | 8 | INCONCLUSIVE | [] | [] |
| discovery | 12 | 4.0 | 8 | INCONCLUSIVE | [] | [] |
| discovery | 4 | 6.0 | 8 | INCONCLUSIVE | [] | [] |
| discovery | 8 | 6.0 | 8 | INCONCLUSIVE | [] | [] |
| discovery | 12 | 6.0 | 8 | INCONCLUSIVE | [] | [] |
| discovery | 12 | 8.0 | 8 | INCONCLUSIVE | [] | [] |
| discovery | 4 | 10.0 | 8 | INCONCLUSIVE | [] | [] |
| discovery | 8 | 10.0 | 8 | INCONCLUSIVE | [] | [] |
| discovery | 12 | 10.0 | 8 | INCONCLUSIVE | [] | [] |
| discovery | 4 | 8.0 | 8 | INCONCLUSIVE | [] | [] |
| discovery | 8 | 8.0 | 8 | INCONCLUSIVE | [] | [] |

이 표의 TTFT 판정은 기존 Type7 CI 기준이다.
역CDF 민감도 분석과 Native-relative Decode 보호는 별도로 검사한다.
이 표만으로 연구 GO를 선언하지 않는다.

TTFT의 통계적 Crossover와 Decode-safe Serving 우위는 서로 다른 판정이다.
공통 cap8의 결론을 모든 Token Streaming Pipeline으로 확대하지 않는다.

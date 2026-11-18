# Unified-budget DuetServe Motivation 결과

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

## 네 Panel과 데이터 선택

X축은 Offered QPS2–10이다.
Panel은 Joint SLO Attainment, Mean TTFT, Mean TBT, Generated Output tokens/s이다.
Last4/8/12 Pair를 각각 같은 수준으로 제시한다.
기본 Last12 Pair는 결과를 본 뒤 유리한 Cut으로 선정한 것이 아니다.
B512/S16은 이 Figure에 포함하지 않는다.
Agg의 주황색 Curve도 새 Prompt/Arrival 계약으로 재측정한다.
기존 Historical Figure는 덮어쓰지 않는다.
QPS6에서 Confirmation3회가 완료되면 해당 평균과 CI를 사용한다.
다른 점은 같은 Discovery Seed의 n1이다.
따라서 모든 Curve 점을 동일 Seed의 단일 Sweep이라고 표시하지 않는다.
미측정 점은 비워 둔다. 실측 점처럼 보간하지 않는다.

## Output200 완료 조건

| Method | Output | QPS | Stage | Starts | Mean TTFT ms | TTFT p95 ms | Mean TBT ms | ITL p99 ms | TPOT p95 ms | Output tokens/s | Joint SLO % | SLO tokens/s |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Agg | 200 | 2.0 | discovery | 1 | 326.788 | 371.650 | 30.289 | 237.346 | 34.678 | 359.765 | 1.389 | 4.997 |
| Native | 200 | 2.0 | discovery | 1 | 7818.681 | 15909.833 | 23.931 | 24.643 | 24.004 | 290.541 | 36.111 | 104.918 |
| Layer4 | 200 | 2.0 | discovery | 1 | 297926.201 | 570254.673 | 23.457 | 24.051 | 23.777 | 42.309 | 1.389 | 0.588 |
| Token4 | 200 | 2.0 | discovery | 1 | 10048.523 | 20347.921 | 23.982 | 24.468 | 24.013 | 275.741 | 27.778 | 76.595 |
| Layer8 | 200 | 2.0 | discovery | 1 | 298442.701 | 571326.219 | 23.389 | 24.074 | 23.781 | 42.245 | 1.389 | 0.587 |
| Token8 | 200 | 2.0 | discovery | 1 | 51697.900 | 99973.788 | 23.305 | 24.256 | 23.936 | 154.952 | 3.472 | 5.380 |
| Layer12 | 200 | 2.0 | discovery | 1 | 301180.135 | 576557.974 | 23.430 | 24.115 | 23.785 | 41.891 | 1.389 | 0.582 |
| Token12 | 200 | 2.0 | discovery | 1 | 81072.859 | 156681.748 | 23.282 | 24.027 | 23.697 | 117.312 | 2.778 | 3.259 |
| Agg | 200 | 4.0 | discovery | 1 | 400.887 | 633.061 | 50.223 | 294.642 | 64.949 | 663.109 | 0.694 | 4.605 |
| Native | 200 | 4.0 | discovery | 1 | 26198.807 | 48865.484 | 24.017 | 24.946 | 23.997 | 306.337 | 6.250 | 19.146 |
| Layer4 | 200 | 4.0 | discovery | 1 | 315931.868 | 602703.589 | 23.357 | 24.012 | 23.773 | 42.536 | 1.389 | 0.591 |
| Token4 | 200 | 4.0 | discovery | 1 | 29092.549 | 54286.398 | 23.985 | 24.494 | 24.016 | 289.294 | 6.250 | 18.081 |
| Layer8 | 200 | 4.0 | discovery | 1 | 320418.677 | 610657.891 | 23.462 | 24.090 | 23.784 | 42.015 | 1.389 | 0.584 |
| Token8 | 200 | 4.0 | discovery | 1 | 71822.206 | 135682.507 | 23.406 | 24.209 | 23.935 | 154.928 | 3.472 | 5.379 |
| Layer12 | 200 | 4.0 | discovery | 1 | 322998.768 | 615387.936 | 23.442 | 24.113 | 23.784 | 41.727 | 1.389 | 0.580 |
| Token12 | 200 | 4.0 | discovery | 1 | 101651.733 | 194334.467 | 23.350 | 24.086 | 23.708 | 116.891 | 2.778 | 3.247 |
| Agg | 200 | 6.0 | discovery | 1 | 1940.938 | 5556.912 | 66.814 | 300.347 | 81.377 | 791.703 | 0.000 | 0.000 |
| Native | 200 | 6.0 | discovery | 1 | 33043.213 | 60767.268 | 23.967 | 24.708 | 24.015 | 306.549 | 6.250 | 19.159 |
| Layer4 | 200 | 6.0 | discovery | 1 | 322923.401 | 614908.180 | 23.373 | 24.006 | 23.776 | 42.529 | 1.389 | 0.591 |
| Token4 | 200 | 6.0 | discovery | 1 | 35970.711 | 66232.122 | 23.990 | 24.647 | 24.021 | 288.743 | 5.556 | 16.041 |
| Layer8 | 200 | 6.0 | discovery | 1 | 326278.024 | 621004.091 | 23.450 | 24.091 | 23.798 | 42.135 | 1.389 | 0.585 |
| Token8 | 200 | 6.0 | discovery | 1 | 78142.520 | 146919.605 | 23.350 | 24.211 | 23.918 | 155.376 | 3.472 | 5.395 |
| Layer12 | 200 | 6.0 | discovery | 1 | 329904.150 | 627729.255 | 23.469 | 24.134 | 23.785 | 41.691 | 1.389 | 0.579 |
| Token12 | 200 | 6.0 | discovery | 1 | 108508.429 | 206877.018 | 23.365 | 24.036 | 23.716 | 116.790 | 2.778 | 3.244 |
| Agg | 200 | 8.0 | discovery | 1 | 3982.586 | 9151.043 | 69.153 | 303.011 | 82.931 | 831.273 | 0.000 | 0.000 |
| Native | 200 | 8.0 | discovery | 1 | 36327.765 | 66521.003 | 23.967 | 24.721 | 24.000 | 306.998 | 6.250 | 19.187 |
| Layer4 | 200 | 8.0 | discovery | 1 | 328591.458 | 625438.819 | 23.460 | 24.052 | 23.779 | 42.219 | 1.389 | 0.586 |
| Token4 | 200 | 8.0 | discovery | 1 | 39046.243 | 71502.881 | 23.986 | 24.526 | 24.012 | 290.598 | 4.167 | 12.108 |
| Layer8 | 200 | 8.0 | discovery | 1 | 330375.467 | 628294.606 | 23.487 | 24.110 | 23.780 | 42.034 | 0.694 | 0.292 |
| Token8 | 200 | 8.0 | discovery | 1 | 82173.963 | 154200.246 | 23.401 | 24.284 | 23.951 | 154.127 | 2.778 | 4.281 |
| Layer12 | 200 | 8.0 | discovery | 1 | 332331.610 | 632236.784 | 23.455 | 24.111 | 23.799 | 41.798 | 1.389 | 0.581 |
| Token12 | 200 | 8.0 | discovery | 1 | 111786.688 | 212838.391 | 23.380 | 24.058 | 23.710 | 116.745 | 1.389 | 1.621 |
| Agg | 200 | 10.0 | discovery | 1 | 5543.551 | 11761.060 | 69.431 | 304.125 | 81.863 | 849.084 | 0.000 | 0.000 |
| Native | 200 | 10.0 | discovery | 1 | 38842.862 | 71017.517 | 23.978 | 24.710 | 24.037 | 303.688 | 6.250 | 18.981 |
| Layer4 | 200 | 10.0 | discovery | 1 | 330825.661 | 629335.845 | 23.462 | 24.049 | 23.778 | 42.213 | 1.389 | 0.586 |
| Token4 | 200 | 10.0 | discovery | 1 | 41384.735 | 75703.354 | 23.984 | 24.528 | 24.020 | 288.566 | 0.694 | 2.004 |
| Layer8 | 200 | 10.0 | discovery | 1 | 333081.091 | 632962.263 | 23.454 | 24.084 | 23.777 | 41.972 | 1.389 | 0.583 |
| Token8 | 200 | 10.0 | discovery | 1 | 83552.855 | 156340.348 | 23.334 | 24.187 | 23.931 | 155.352 | 0.694 | 1.079 |
| Layer12 | 200 | 10.0 | discovery | 1 | 335007.308 | 637069.572 | 23.444 | 24.109 | 23.783 | 41.726 | 1.389 | 0.580 |
| Token12 | 200 | 10.0 | discovery | 1 | 112967.524 | 214586.543 | 23.302 | 24.014 | 23.699 | 117.748 | 0.694 | 0.818 |

## 기존 Native와의 차이

새 Native는 실제 양 Worker의 B8192/S1024를 검사한다.
예전32000/4와128000/16 설정으로 실행한 시간을 새 결과에 합치지 않는다.
별도 Bridge는 같은 과거 Prompt와 Arrival Trace에서 새 Native를 재측정한다.
이 비교에는 B/S 외에 Admission, Source Packing, Reservation 및 측정 Epoch 차이도 있다.
따라서 Bridge를 B만의 인과 효과라고 주장하지 않는다.
DuetServe Figure2는 DP2이다. TP2 결과로 바꾸어 설명하지 않는다.
우리 vLLM/NIXL 경로가 원 논문의 Runtime과 완전히 같다고 주장하지 않는다.
원문은 [DuetServe](https://pages.cs.wisc.edu/~markhill/papers/icml2026_DuetServe.pdf)이다.

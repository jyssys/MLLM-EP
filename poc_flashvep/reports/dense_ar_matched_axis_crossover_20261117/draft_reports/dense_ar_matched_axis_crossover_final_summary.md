# Dense AR Matched-axis Crossover 최종 요약

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

## 판정

현재 판정은 `INCONCLUSIVE_PENDING_CONFIRMATION`이다.
완료한 Confirmation에서 Type7 TTFT CI가 Layer를 지지하는 점은0개이다.
Type7 TTFT CI가 Token을 지지하는 점은0개이다.
Layer TTFT 우위와 Native-relative Decode 보호를 함께 통과한 점은0개이다.
공통 SLO Goodput 우위와 Native-relative Decode 보호를 함께 통과한 점은0개이다.
두 통과 수에는 역CDF 민감도 점검도 적용한다.
이 수는 독립적인 전체 Workload 개수를 뜻하지 않는다.
조건별 CI는 Pointwise이며 다중 비교를 보정하지 않았다.
Absolute SLO와 Goodput 비율은 별도 표에서 확인한다.
모든 조건의 전역적 축 우위나 Sustainable Capacity를 주장하지 않는다.

## QPS6 전체 Output 비교

| Method | Output | QPS | Stage | Starts | Mean TTFT ms | TTFT p95 ms | Mean TBT ms | ITL p99 ms | TPOT p95 ms | Output tokens/s | Joint SLO % | SLO tokens/s |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Agg | 16 | 6.0 | discovery | 1 | 515.903 | 967.771 | 100.969 | 294.787 | 182.621 | 87.633 | 5.556 | 4.868 |
| Native | 16 | 6.0 | discovery | 1 | 8005.596 | 16230.893 | 22.888 | 23.370 | 22.926 | 53.890 | 38.194 | 20.583 |
| Layer4 | 16 | 6.0 | discovery | 1 | 15577.764 | 30807.720 | 23.493 | 24.167 | 23.849 | 39.653 | 15.972 | 6.333 |
| Token4 | 16 | 6.0 | discovery | 1 | 5986.859 | 12174.156 | 23.411 | 24.269 | 23.818 | 60.325 | 45.139 | 27.230 |
| Layer8 | 16 | 6.0 | discovery | 1 | 17759.812 | 34940.856 | 23.558 | 24.472 | 23.905 | 36.894 | 13.194 | 4.868 |
| Token8 | 16 | 6.0 | discovery | 1 | 3986.296 | 7919.565 | 23.463 | 24.589 | 23.874 | 67.591 | 65.972 | 44.591 |
| Layer12 | 16 | 6.0 | discovery | 1 | 21036.633 | 41129.815 | 23.552 | 24.581 | 23.922 | 33.414 | 11.111 | 3.713 |
| Token12 | 16 | 6.0 | discovery | 1 | 4755.896 | 9916.258 | 23.406 | 24.638 | 23.865 | 63.714 | 56.944 | 36.282 |
| Agg | 32 | 6.0 | discovery | 1 | 528.583 | 988.349 | 103.820 | 299.279 | 172.907 | 172.394 | 2.083 | 3.592 |
| Native | 32 | 6.0 | discovery | 1 | 8246.963 | 16366.639 | 22.716 | 23.303 | 22.851 | 106.310 | 35.417 | 37.651 |
| Layer4 | 32 | 6.0 | discovery | 1 | 42327.080 | 81661.553 | 23.469 | 24.168 | 23.797 | 41.154 | 5.556 | 2.286 |
| Token4 | 32 | 6.0 | discovery | 1 | 6306.347 | 12548.155 | 23.241 | 24.192 | 23.615 | 117.361 | 44.444 | 52.161 |
| Layer8 | 32 | 6.0 | discovery | 1 | 44405.987 | 85592.865 | 23.459 | 24.479 | 23.829 | 39.672 | 5.556 | 2.204 |
| Token8 | 32 | 6.0 | discovery | 1 | 6856.013 | 13452.383 | 23.301 | 24.244 | 23.841 | 114.388 | 40.972 | 46.867 |
| Layer12 | 32 | 6.0 | discovery | 1 | 46877.518 | 90317.988 | 23.515 | 24.536 | 23.829 | 38.041 | 4.861 | 1.849 |
| Token12 | 32 | 6.0 | discovery | 1 | 13378.652 | 26256.325 | 23.377 | 24.568 | 23.768 | 85.745 | 17.361 | 14.886 |
| Agg | 64 | 6.0 | discovery | 1 | 550.645 | 1035.152 | 100.008 | 302.529 | 165.407 | 331.535 | 0.000 | 0.000 |
| Native | 64 | 6.0 | discovery | 1 | 8776.196 | 16832.016 | 24.241 | 26.222 | 24.448 | 203.559 | 29.861 | 60.785 |
| Layer4 | 64 | 6.0 | discovery | 1 | 96354.657 | 184659.692 | 23.447 | 24.077 | 23.759 | 41.676 | 2.778 | 1.158 |
| Token4 | 64 | 6.0 | discovery | 1 | 7168.976 | 13248.187 | 23.955 | 26.378 | 24.600 | 228.445 | 38.194 | 87.253 |
| Layer8 | 64 | 6.0 | discovery | 1 | 98055.587 | 187436.234 | 23.416 | 24.241 | 23.772 | 41.106 | 2.778 | 1.142 |
| Token8 | 64 | 6.0 | discovery | 1 | 20546.908 | 39356.808 | 23.361 | 24.583 | 23.898 | 135.001 | 10.417 | 14.063 |
| Layer12 | 64 | 6.0 | discovery | 1 | 100249.397 | 191751.059 | 23.429 | 24.323 | 23.774 | 40.303 | 2.778 | 1.120 |
| Token12 | 64 | 6.0 | discovery | 1 | 30938.296 | 59551.805 | 23.223 | 24.133 | 23.671 | 103.153 | 6.944 | 7.163 |
| Agg | 128 | 6.0 | discovery | 1 | 984.641 | 2956.441 | 78.124 | 301.458 | 107.147 | 583.878 | 0.000 | 0.000 |
| Native | 128 | 6.0 | discovery | 1 | 17169.500 | 31920.233 | 23.985 | 25.197 | 24.021 | 298.764 | 10.417 | 31.121 |
| Layer4 | 128 | 6.0 | discovery | 1 | 204987.573 | 390681.336 | 23.438 | 24.065 | 23.766 | 41.930 | 1.389 | 0.582 |
| Token4 | 128 | 6.0 | discovery | 1 | 19713.376 | 36480.710 | 23.992 | 24.523 | 24.020 | 277.154 | 9.722 | 26.946 |
| Layer8 | 128 | 6.0 | discovery | 1 | 206529.590 | 393936.077 | 23.454 | 24.157 | 23.781 | 41.608 | 1.389 | 0.578 |
| Token8 | 128 | 6.0 | discovery | 1 | 47565.452 | 89673.824 | 23.407 | 24.267 | 23.906 | 149.842 | 3.472 | 5.203 |
| Layer12 | 128 | 6.0 | discovery | 1 | 208909.133 | 398267.411 | 23.474 | 24.239 | 23.791 | 41.181 | 1.389 | 0.572 |
| Token12 | 128 | 6.0 | discovery | 1 | 67813.823 | 129226.468 | 23.301 | 24.205 | 23.741 | 112.545 | 2.778 | 3.126 |
| Agg | 200 | 6.0 | discovery | 1 | 1940.938 | 5556.912 | 66.814 | 300.347 | 81.377 | 791.703 | 0.000 | 0.000 |
| Native | 200 | 6.0 | discovery | 1 | 33043.213 | 60767.268 | 23.967 | 24.708 | 24.015 | 306.549 | 6.250 | 19.159 |
| Layer4 | 200 | 6.0 | discovery | 1 | 322923.401 | 614908.180 | 23.373 | 24.006 | 23.776 | 42.529 | 1.389 | 0.591 |
| Token4 | 200 | 6.0 | discovery | 1 | 35970.711 | 66232.122 | 23.990 | 24.647 | 24.021 | 288.743 | 5.556 | 16.041 |
| Layer8 | 200 | 6.0 | discovery | 1 | 326278.024 | 621004.091 | 23.450 | 24.091 | 23.798 | 42.135 | 1.389 | 0.585 |
| Token8 | 200 | 6.0 | discovery | 1 | 78142.520 | 146919.605 | 23.350 | 24.211 | 23.918 | 155.376 | 3.472 | 5.395 |
| Layer12 | 200 | 6.0 | discovery | 1 | 329904.150 | 627729.255 | 23.469 | 24.134 | 23.785 | 41.691 | 1.389 | 0.579 |
| Token12 | 200 | 6.0 | discovery | 1 | 108508.429 | 206877.018 | 23.365 | 24.036 | 23.716 | 116.790 | 2.778 | 3.244 |

## 동일 작업량과 실행 구조

| Last | Token Suffix Histogram (N144) | D GEMM Rows / method | D Token/Layer causal Attention pairs | Extra Layer Hidden/Residual bytes |
| --- | --- | --- | --- | --- |
| 4 | {"880": 64, "896": 80} | 4608000 | 1.888769 | 18874368000 |
| 8 | {"1776": 128, "1792": 16} | 9216000 | 1.777679 | 18874368000 |
| 12 | {"2656": 48, "2672": 96} | 13824000 | 1.666581 | 18874368000 |

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


## 성공·실패·미실행의 구분

Full-state 정합성은 N8→N32→N128과 Output200 N8에서 검사했다.
Full Prompt KV, Final Hidden, Greedy Output, Layer 방문, Full36 Decode, ACK와 정상 Retirement를 검사했다.
성능 Start마다 실제 B/S, Output, Query Coverage, Matched Rows와 Credit Lifetime도 검사한다.
정상 종료만으로 계측 간섭 Gate를 통과했다고 표시하지 않는다.
실패한 Profiler와 Optional Projection Observer 기록은 보존한다.
정합성 실패 결과를 적격 성능에 합치지 않는다.
Dynamic Scheduler 개선, TP2 Strict Parity 및 새 Streaming Token Pipeline은 `NOT_RUN`이다.
Last2 Full Sweep은 사용자가 취소했다. 해당 추가 GPU 실행은0회이다.
별도 완료한 Serial Last2/4 Diagnostic은 원래 Sweep과 합치지 않는다.

## 다음 연구에서 필요한 검증

큰 D Decode Cohort를 유지하면서 Layer 준비를 여러 Iteration으로 분리할 수 있는지 검토할 수 있다.
이는 이번 구현에서 측정한 결과가 아니다.
그 구조를 구현하면 Token도 같은 Runtime과 Admission에서 다시 비교해야 한다.
현재 결과만으로 Layer×Token Dynamic Selection의 Serving 이득을 주장하지 않는다.

## 보존과 전달

이 보고서는 기존 사용자 Checkout과 Index를 바꾸지 않는 별도 Branch에 저장한다.
Per-job Command, Source ZIP, Hash, Worker Log, Prompt/Arrival 및 실패 기록을 보존한다.
큰 Evidence는 원본 SHA256과 압축 해제 SHA256을 검사한 Archive로 전달한다.
최종 Remote SHA와 GPU 복원 Receipt는 별도로 검사한다.
GPU0–7의 승인 Burner가 실제 실행 중인지 확인한 뒤 종료한다.

## 요청한15개 질문의 답

1. B8192/S1024: 완료된 Start의 실제 양 Worker 설정을 검사한다. 미실행 조건까지 통과했다고 표시하지 않는다.
2. D Admission: 공통 상한은8이다. 실제 Layer 준비 Group1과 Token 최대8/4/3은 같지 않다.
3. 기존 Native와의 차이: 별도 Historical Bridge로 비교한다. B/S만의 단독 인과 효과는 아니다.
4. Last4/Token≈889: 아래 Pair별 TTFT 비율과 CI를 사용한다.
5. Last8/Token≈1778: 아래 Pair별 TTFT 비율과 CI를 사용한다.
6. Last12/Token≈2667: 아래 Pair별 TTFT 비율과 CI를 사용한다.
7. Output Sweep: Output16/32/64/128/200의 모든 완료 결과를 같은 고정 Partition에서 비교한다.
8. Crossover: 독립 Start의 TTFT CI가 양 방향 우위를 각각 지지해야 통계적 역전을 확인한다.
9. 원인 분리: 같은 B/S와 cap8은 실제 Cohort Geometry를 일치시키지 못한다. cap1 통제를 별도로 검사한다.
10. GEMM/Attention/Transfer/Queue: 구조적 작업량과 Host 경로를 구분한다. Exclusive GPU 성분별 원인 비율은 확인하지 못했다.
11. 같은 SLO의 Goodput: Common SLO 비율과 CI를 사용한다. Reference0인 비율을 개선0%로 처리하지 않는다.
12. Output200 Motivation: 새 Agg와 Native를 같은 Trace에서 측정한다. 아직 측정하지 않은 QPS는 비워 둔다.
13. 새 네 Panel: Last4/8/12의 모든 고정 Pair를 각각 제시한다. 기본 Alias는 Last12이다.
14. 과거 Output16 Last12 우위: 이번 통일 조건의 비율로 다시 판단한다. 과거13.95%를 새 측정값으로 사용하지 않는다.
15. 관측값: 실제 D Cohort 크기, Residency, Ready-to-admit 대기, Source Window/KV Credit과 Suffix 비용을 함께 봐야 한다.

이 관측값만으로 새로운 Axis Predictor가 검증됐다는 뜻은 아니다.
물리적 GPU Idle과 SM/HBM Counter가 없으면 해당 항목은 `NOT_MEASURED`로 남긴다.

## Pair별 상세 비율

| Stage | Last | Output | QPS | Starts | Layer/Token TTFT p95 [95% CI] | Layer/Token common-SLO tokens/s [95% CI] |
| --- | --- | --- | --- | --- | --- | --- |
| discovery | 4 | 16 | 2.0 | 1 | 1.772570 [CI: NOT_MEASURED] | 0.991664 [CI: NOT_MEASURED] |
| discovery | 8 | 16 | 2.0 | 1 | 1.924198 [CI: NOT_MEASURED] | 0.978100 [CI: NOT_MEASURED] |
| discovery | 12 | 16 | 2.0 | 1 | 2.309570 [CI: NOT_MEASURED] | 0.972085 [CI: NOT_MEASURED] |
| discovery | 4 | 16 | 4.0 | 1 | 5.676597 [CI: NOT_MEASURED] | 0.230327 [CI: NOT_MEASURED] |
| discovery | 8 | 16 | 4.0 | 1 | 11.315581 [CI: NOT_MEASURED] | 0.125889 [CI: NOT_MEASURED] |
| discovery | 12 | 16 | 4.0 | 1 | 11.116064 [CI: NOT_MEASURED] | 0.103688 [CI: NOT_MEASURED] |
| discovery | 4 | 16 | 6.0 | 1 | 2.530584 [CI: NOT_MEASURED] | 0.232587 [CI: NOT_MEASURED] |
| discovery | 8 | 16 | 6.0 | 1 | 4.411966 [CI: NOT_MEASURED] | 0.109168 [CI: NOT_MEASURED] |
| discovery | 12 | 16 | 6.0 | 1 | 4.147715 [CI: NOT_MEASURED] | 0.102328 [CI: NOT_MEASURED] |
| discovery | 12 | 16 | 8.0 | 1 | 3.049675 [CI: NOT_MEASURED] | 0.144594 [CI: NOT_MEASURED] |
| discovery | 4 | 16 | 10.0 | 1 | 1.963645 [CI: NOT_MEASURED] | 0.349959 [CI: NOT_MEASURED] |
| discovery | 8 | 16 | 10.0 | 1 | 2.734754 [CI: NOT_MEASURED] | 0.175238 [CI: NOT_MEASURED] |
| discovery | 12 | 16 | 10.0 | 1 | 2.579651 [CI: NOT_MEASURED] | 0.189819 [CI: NOT_MEASURED] |
| discovery | 4 | 32 | 2.0 | 1 | 23.607648 [CI: NOT_MEASURED] | 0.084035 [CI: NOT_MEASURED] |
| discovery | 8 | 32 | 2.0 | 1 | 24.508753 [CI: NOT_MEASURED] | 0.071836 [CI: NOT_MEASURED] |
| discovery | 12 | 32 | 2.0 | 1 | 21.440977 [CI: NOT_MEASURED] | 0.061060 [CI: NOT_MEASURED] |
| discovery | 4 | 32 | 4.0 | 1 | 18.291024 [CI: NOT_MEASURED] | 0.023972 [CI: NOT_MEASURED] |
| discovery | 8 | 32 | 4.0 | 1 | 14.903385 [CI: NOT_MEASURED] | 0.025227 [CI: NOT_MEASURED] |
| discovery | 12 | 32 | 4.0 | 1 | 5.142751 [CI: NOT_MEASURED] | 0.057887 [CI: NOT_MEASURED] |
| discovery | 4 | 32 | 6.0 | 1 | 6.507853 [CI: NOT_MEASURED] | 0.043833 [CI: NOT_MEASURED] |
| discovery | 8 | 32 | 6.0 | 1 | 6.362654 [CI: NOT_MEASURED] | 0.047027 [CI: NOT_MEASURED] |
| discovery | 12 | 32 | 6.0 | 1 | 3.439856 [CI: NOT_MEASURED] | 0.124224 [CI: NOT_MEASURED] |
| discovery | 4 | 32 | 10.0 | 1 | 4.287113 [CI: NOT_MEASURED] | 0.087650 [CI: NOT_MEASURED] |
| discovery | 8 | 32 | 10.0 | 1 | 3.771588 [CI: NOT_MEASURED] | 0.111210 [CI: NOT_MEASURED] |
| discovery | 12 | 32 | 10.0 | 1 | 2.711274 [CI: NOT_MEASURED] | 0.196641 [CI: NOT_MEASURED] |
| discovery | 4 | 64 | 2.0 | 1 | 60.087022 [CI: NOT_MEASURED] | 0.012117 [CI: NOT_MEASURED] |
| discovery | 8 | 64 | 2.0 | 1 | 39.315981 [CI: NOT_MEASURED] | 0.012381 [CI: NOT_MEASURED] |
| discovery | 12 | 64 | 2.0 | 1 | 10.235198 [CI: NOT_MEASURED] | 0.026867 [CI: NOT_MEASURED] |
| discovery | 4 | 64 | 4.0 | 1 | 36.738366 [CI: NOT_MEASURED] | 0.005959 [CI: NOT_MEASURED] |
| discovery | 8 | 64 | 4.0 | 1 | 6.610608 [CI: NOT_MEASURED] | 0.036973 [CI: NOT_MEASURED] |
| discovery | 12 | 64 | 4.0 | 1 | 3.733542 [CI: NOT_MEASURED] | 0.143247 [CI: NOT_MEASURED] |
| discovery | 4 | 64 | 6.0 | 1 | 13.938488 [CI: NOT_MEASURED] | 0.013268 [CI: NOT_MEASURED] |
| discovery | 8 | 64 | 6.0 | 1 | 4.762486 [CI: NOT_MEASURED] | 0.081197 [CI: NOT_MEASURED] |
| discovery | 12 | 64 | 6.0 | 1 | 3.219903 [CI: NOT_MEASURED] | 0.156286 [CI: NOT_MEASURED] |
| discovery | 4 | 64 | 10.0 | 1 | 8.805824 [CI: NOT_MEASURED] | 0.032446 [CI: NOT_MEASURED] |
| discovery | 8 | 64 | 10.0 | 1 | 4.115261 [CI: NOT_MEASURED] | 0.092137 [CI: NOT_MEASURED] |
| discovery | 12 | 64 | 10.0 | 1 | 2.888876 [CI: NOT_MEASURED] | 0.223670 [CI: NOT_MEASURED] |
| discovery | 4 | 128 | 2.0 | 1 | 75.400065 [CI: NOT_MEASURED] | 0.002828 [CI: NOT_MEASURED] |
| discovery | 8 | 128 | 2.0 | 1 | 7.931426 [CI: NOT_MEASURED] | 0.080546 [CI: NOT_MEASURED] |
| discovery | 12 | 128 | 2.0 | 1 | 4.312071 [CI: NOT_MEASURED] | 0.105459 [CI: NOT_MEASURED] |
| discovery | 4 | 128 | 4.0 | 1 | 15.712287 [CI: NOT_MEASURED] | 0.012514 [CI: NOT_MEASURED] |
| discovery | 8 | 128 | 4.0 | 1 | 4.899644 [CI: NOT_MEASURED] | 0.111317 [CI: NOT_MEASURED] |
| discovery | 12 | 128 | 4.0 | 1 | 3.319725 [CI: NOT_MEASURED] | 0.182348 [CI: NOT_MEASURED] |
| discovery | 4 | 128 | 6.0 | 1 | 10.709258 [CI: NOT_MEASURED] | 0.021612 [CI: NOT_MEASURED] |
| discovery | 8 | 128 | 6.0 | 1 | 4.392989 [CI: NOT_MEASURED] | 0.111070 [CI: NOT_MEASURED] |
| discovery | 12 | 128 | 6.0 | 1 | 3.081934 [CI: NOT_MEASURED] | 0.182953 [CI: NOT_MEASURED] |
| discovery | 4 | 128 | 10.0 | 1 | 8.659378 [CI: NOT_MEASURED] | 0.033892 [CI: NOT_MEASURED] |
| discovery | 8 | 128 | 10.0 | 1 | 4.084829 [CI: NOT_MEASURED] | 0.110797 [CI: NOT_MEASURED] |
| discovery | 12 | 128 | 10.0 | 1 | 2.937654 [CI: NOT_MEASURED] | 0.182943 [CI: NOT_MEASURED] |
| discovery | 4 | 200 | 2.0 | 1 | 28.025206 [CI: NOT_MEASURED] | 0.007672 [CI: NOT_MEASURED] |
| discovery | 8 | 200 | 2.0 | 1 | 5.714760 [CI: NOT_MEASURED] | 0.109052 [CI: NOT_MEASURED] |
| discovery | 12 | 200 | 2.0 | 1 | 3.679803 [CI: NOT_MEASURED] | 0.178547 [CI: NOT_MEASURED] |
| discovery | 4 | 200 | 4.0 | 1 | 11.102295 [CI: NOT_MEASURED] | 0.032674 [CI: NOT_MEASURED] |
| discovery | 8 | 200 | 4.0 | 1 | 4.500638 [CI: NOT_MEASURED] | 0.108476 [CI: NOT_MEASURED] |
| discovery | 12 | 200 | 4.0 | 1 | 3.166643 [CI: NOT_MEASURED] | 0.178485 [CI: NOT_MEASURED] |
| discovery | 4 | 200 | 6.0 | 1 | 9.284138 [CI: NOT_MEASURED] | 0.036823 [CI: NOT_MEASURED] |
| discovery | 8 | 200 | 6.0 | 1 | 4.226829 [CI: NOT_MEASURED] | 0.108472 [CI: NOT_MEASURED] |
| discovery | 12 | 200 | 6.0 | 1 | 3.034311 [CI: NOT_MEASURED] | 0.178486 [CI: NOT_MEASURED] |
| discovery | 4 | 200 | 8.0 | 1 | 8.747044 [CI: NOT_MEASURED] | 0.048428 [CI: NOT_MEASURED] |
| discovery | 8 | 200 | 8.0 | 1 | 4.074537 [CI: NOT_MEASURED] | 0.068181 [CI: NOT_MEASURED] |
| discovery | 12 | 200 | 8.0 | 1 | 2.970502 [CI: NOT_MEASURED] | 0.358027 [CI: NOT_MEASURED] |
| discovery | 4 | 200 | 10.0 | 1 | 8.313183 [CI: NOT_MEASURED] | 0.292574 [CI: NOT_MEASURED] |
| discovery | 8 | 200 | 10.0 | 1 | 4.048617 [CI: NOT_MEASURED] | 0.540349 [CI: NOT_MEASURED] |
| discovery | 12 | 200 | 10.0 | 1 | 2.968824 [CI: NOT_MEASURED] | 0.708737 [CI: NOT_MEASURED] |

## Native-relative 상세 판정

| Stage | Candidate | Last | Output | QPS | Starts | ITL p99 ratio [95% CI] | TPOT p95 ratio [95% CI] | Native-relative Gate |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| discovery | Layer | 4 | 16 | 2.0 | 1 | 1.030153 [CI: NOT_MEASURED] | 1.037146 [CI: NOT_MEASURED] | NOT_ESTABLISHED |
| discovery | Token | 4 | 16 | 2.0 | 1 | 1.031553 [CI: NOT_MEASURED] | 1.036722 [CI: NOT_MEASURED] | NOT_ESTABLISHED |
| discovery | Layer | 8 | 16 | 2.0 | 1 | 1.047170 [CI: NOT_MEASURED] | 1.040389 [CI: NOT_MEASURED] | NOT_ESTABLISHED |
| discovery | Token | 8 | 16 | 2.0 | 1 | 1.043755 [CI: NOT_MEASURED] | 1.040324 [CI: NOT_MEASURED] | NOT_ESTABLISHED |
| discovery | Layer | 12 | 16 | 2.0 | 1 | 1.049422 [CI: NOT_MEASURED] | 1.041540 [CI: NOT_MEASURED] | NOT_ESTABLISHED |
| discovery | Token | 12 | 16 | 2.0 | 1 | 1.046945 [CI: NOT_MEASURED] | 1.041132 [CI: NOT_MEASURED] | NOT_ESTABLISHED |
| discovery | Layer | 4 | 16 | 4.0 | 1 | 1.036126 [CI: NOT_MEASURED] | 1.039128 [CI: NOT_MEASURED] | NOT_ESTABLISHED |
| discovery | Token | 4 | 16 | 4.0 | 1 | 1.034557 [CI: NOT_MEASURED] | 1.039419 [CI: NOT_MEASURED] | NOT_ESTABLISHED |
| discovery | Layer | 8 | 16 | 4.0 | 1 | 1.048486 [CI: NOT_MEASURED] | 1.043305 [CI: NOT_MEASURED] | NOT_ESTABLISHED |
| discovery | Token | 8 | 16 | 4.0 | 1 | 1.048239 [CI: NOT_MEASURED] | 1.041972 [CI: NOT_MEASURED] | NOT_ESTABLISHED |
| discovery | Layer | 12 | 16 | 4.0 | 1 | 1.055713 [CI: NOT_MEASURED] | 1.044107 [CI: NOT_MEASURED] | NOT_ESTABLISHED |
| discovery | Token | 12 | 16 | 4.0 | 1 | 1.050839 [CI: NOT_MEASURED] | 1.041250 [CI: NOT_MEASURED] | NOT_ESTABLISHED |
| discovery | Layer | 4 | 16 | 6.0 | 1 | 1.034092 [CI: NOT_MEASURED] | 1.040276 [CI: NOT_MEASURED] | NOT_ESTABLISHED |
| discovery | Token | 4 | 16 | 6.0 | 1 | 1.038467 [CI: NOT_MEASURED] | 1.038926 [CI: NOT_MEASURED] | NOT_ESTABLISHED |
| discovery | Layer | 8 | 16 | 6.0 | 1 | 1.047136 [CI: NOT_MEASURED] | 1.042727 [CI: NOT_MEASURED] | NOT_ESTABLISHED |
| discovery | Token | 8 | 16 | 6.0 | 1 | 1.052154 [CI: NOT_MEASURED] | 1.041370 [CI: NOT_MEASURED] | NOT_ESTABLISHED |
| discovery | Layer | 12 | 16 | 6.0 | 1 | 1.051785 [CI: NOT_MEASURED] | 1.043446 [CI: NOT_MEASURED] | NOT_ESTABLISHED |
| discovery | Token | 12 | 16 | 6.0 | 1 | 1.054257 [CI: NOT_MEASURED] | 1.040973 [CI: NOT_MEASURED] | NOT_ESTABLISHED |
| discovery | Layer | 12 | 16 | 8.0 | 1 | 1.034222 [CI: NOT_MEASURED] | 1.042573 [CI: NOT_MEASURED] | NOT_ESTABLISHED |
| discovery | Token | 12 | 16 | 8.0 | 1 | 1.038913 [CI: NOT_MEASURED] | 1.036508 [CI: NOT_MEASURED] | NOT_ESTABLISHED |
| discovery | Layer | 4 | 16 | 10.0 | 1 | 1.037293 [CI: NOT_MEASURED] | 1.039877 [CI: NOT_MEASURED] | NOT_ESTABLISHED |
| discovery | Token | 4 | 16 | 10.0 | 1 | 1.042071 [CI: NOT_MEASURED] | 1.036650 [CI: NOT_MEASURED] | NOT_ESTABLISHED |
| discovery | Layer | 8 | 16 | 10.0 | 1 | 1.051460 [CI: NOT_MEASURED] | 1.044032 [CI: NOT_MEASURED] | NOT_ESTABLISHED |
| discovery | Token | 8 | 16 | 10.0 | 1 | 1.042428 [CI: NOT_MEASURED] | 1.038094 [CI: NOT_MEASURED] | NOT_ESTABLISHED |
| discovery | Layer | 12 | 16 | 10.0 | 1 | 1.052043 [CI: NOT_MEASURED] | 1.043424 [CI: NOT_MEASURED] | NOT_ESTABLISHED |
| discovery | Token | 12 | 16 | 10.0 | 1 | 1.059058 [CI: NOT_MEASURED] | 1.041469 [CI: NOT_MEASURED] | NOT_ESTABLISHED |
| discovery | Layer | 4 | 32 | 2.0 | 1 | 1.019988 [CI: NOT_MEASURED] | 1.036777 [CI: NOT_MEASURED] | NOT_ESTABLISHED |
| discovery | Token | 4 | 32 | 2.0 | 1 | 1.028539 [CI: NOT_MEASURED] | 1.035816 [CI: NOT_MEASURED] | NOT_ESTABLISHED |
| discovery | Layer | 8 | 32 | 2.0 | 1 | 1.027639 [CI: NOT_MEASURED] | 1.038680 [CI: NOT_MEASURED] | NOT_ESTABLISHED |
| discovery | Token | 8 | 32 | 2.0 | 1 | 1.037688 [CI: NOT_MEASURED] | 1.043900 [CI: NOT_MEASURED] | NOT_ESTABLISHED |
| discovery | Layer | 12 | 32 | 2.0 | 1 | 1.034032 [CI: NOT_MEASURED] | 1.038669 [CI: NOT_MEASURED] | NOT_ESTABLISHED |
| discovery | Token | 12 | 32 | 2.0 | 1 | 1.033271 [CI: NOT_MEASURED] | 1.037565 [CI: NOT_MEASURED] | NOT_ESTABLISHED |
| discovery | Layer | 4 | 32 | 4.0 | 1 | 1.015248 [CI: NOT_MEASURED] | 1.039536 [CI: NOT_MEASURED] | NOT_ESTABLISHED |
| discovery | Token | 4 | 32 | 4.0 | 1 | 1.047822 [CI: NOT_MEASURED] | 1.048087 [CI: NOT_MEASURED] | NOT_ESTABLISHED |
| discovery | Layer | 8 | 32 | 4.0 | 1 | 1.028243 [CI: NOT_MEASURED] | 1.040638 [CI: NOT_MEASURED] | NOT_ESTABLISHED |
| discovery | Token | 8 | 32 | 4.0 | 1 | 1.045488 [CI: NOT_MEASURED] | 1.047173 [CI: NOT_MEASURED] | NOT_ESTABLISHED |
| discovery | Layer | 12 | 32 | 4.0 | 1 | 1.031460 [CI: NOT_MEASURED] | 1.041389 [CI: NOT_MEASURED] | NOT_ESTABLISHED |
| discovery | Token | 12 | 32 | 4.0 | 1 | 1.037037 [CI: NOT_MEASURED] | 1.039809 [CI: NOT_MEASURED] | NOT_ESTABLISHED |
| discovery | Layer | 4 | 32 | 6.0 | 1 | 1.037142 [CI: NOT_MEASURED] | 1.041402 [CI: NOT_MEASURED] | NOT_ESTABLISHED |
| discovery | Token | 4 | 32 | 6.0 | 1 | 1.038153 [CI: NOT_MEASURED] | 1.033443 [CI: NOT_MEASURED] | NOT_ESTABLISHED |
| discovery | Layer | 8 | 32 | 6.0 | 1 | 1.050486 [CI: NOT_MEASURED] | 1.042794 [CI: NOT_MEASURED] | NOT_ESTABLISHED |
| discovery | Token | 8 | 32 | 6.0 | 1 | 1.040364 [CI: NOT_MEASURED] | 1.043305 [CI: NOT_MEASURED] | NOT_ESTABLISHED |
| discovery | Layer | 12 | 32 | 6.0 | 1 | 1.052921 [CI: NOT_MEASURED] | 1.042796 [CI: NOT_MEASURED] | NOT_ESTABLISHED |
| discovery | Token | 12 | 32 | 6.0 | 1 | 1.054291 [CI: NOT_MEASURED] | 1.040126 [CI: NOT_MEASURED] | NOT_ESTABLISHED |
| discovery | Layer | 4 | 32 | 10.0 | 1 | 1.036929 [CI: NOT_MEASURED] | 1.040936 [CI: NOT_MEASURED] | NOT_ESTABLISHED |
| discovery | Token | 4 | 32 | 10.0 | 1 | 1.048107 [CI: NOT_MEASURED] | 1.050249 [CI: NOT_MEASURED] | NOT_ESTABLISHED |
| discovery | Layer | 8 | 32 | 10.0 | 1 | 1.049492 [CI: NOT_MEASURED] | 1.042655 [CI: NOT_MEASURED] | NOT_ESTABLISHED |
| discovery | Token | 8 | 32 | 10.0 | 1 | 1.072303 [CI: NOT_MEASURED] | 1.048280 [CI: NOT_MEASURED] | NOT_ESTABLISHED |
| discovery | Layer | 12 | 32 | 10.0 | 1 | 1.048440 [CI: NOT_MEASURED] | 1.043123 [CI: NOT_MEASURED] | NOT_ESTABLISHED |
| discovery | Token | 12 | 32 | 10.0 | 1 | 1.050521 [CI: NOT_MEASURED] | 1.038609 [CI: NOT_MEASURED] | NOT_ESTABLISHED |
| discovery | Layer | 4 | 64 | 2.0 | 1 | 0.979121 [CI: NOT_MEASURED] | 0.991003 [CI: NOT_MEASURED] | NOT_ESTABLISHED |
| discovery | Token | 4 | 64 | 2.0 | 1 | 1.004057 [CI: NOT_MEASURED] | 1.001847 [CI: NOT_MEASURED] | NOT_ESTABLISHED |
| discovery | Layer | 8 | 64 | 2.0 | 1 | 0.985435 [CI: NOT_MEASURED] | 0.991513 [CI: NOT_MEASURED] | NOT_ESTABLISHED |
| discovery | Token | 8 | 64 | 2.0 | 1 | 0.992773 [CI: NOT_MEASURED] | 0.996614 [CI: NOT_MEASURED] | NOT_ESTABLISHED |
| discovery | Layer | 12 | 64 | 2.0 | 1 | 0.991359 [CI: NOT_MEASURED] | 0.992649 [CI: NOT_MEASURED] | NOT_ESTABLISHED |
| discovery | Token | 12 | 64 | 2.0 | 1 | 0.991339 [CI: NOT_MEASURED] | 0.988547 [CI: NOT_MEASURED] | NOT_ESTABLISHED |
| discovery | Layer | 4 | 64 | 4.0 | 1 | 0.958197 [CI: NOT_MEASURED] | 0.972799 [CI: NOT_MEASURED] | NOT_ESTABLISHED |
| discovery | Token | 4 | 64 | 4.0 | 1 | 1.048252 [CI: NOT_MEASURED] | 0.999760 [CI: NOT_MEASURED] | NOT_ESTABLISHED |
| discovery | Layer | 8 | 64 | 4.0 | 1 | 0.964734 [CI: NOT_MEASURED] | 0.973750 [CI: NOT_MEASURED] | NOT_ESTABLISHED |
| discovery | Token | 8 | 64 | 4.0 | 1 | 0.974936 [CI: NOT_MEASURED] | 0.978598 [CI: NOT_MEASURED] | NOT_ESTABLISHED |
| discovery | Layer | 12 | 64 | 4.0 | 1 | 0.966697 [CI: NOT_MEASURED] | 0.974586 [CI: NOT_MEASURED] | NOT_ESTABLISHED |
| discovery | Token | 12 | 64 | 4.0 | 1 | 0.970800 [CI: NOT_MEASURED] | 0.971886 [CI: NOT_MEASURED] | NOT_ESTABLISHED |
| discovery | Layer | 4 | 64 | 6.0 | 1 | 0.918189 [CI: NOT_MEASURED] | 0.971827 [CI: NOT_MEASURED] | NOT_ESTABLISHED |
| discovery | Token | 4 | 64 | 6.0 | 1 | 1.005949 [CI: NOT_MEASURED] | 1.006229 [CI: NOT_MEASURED] | NOT_ESTABLISHED |
| discovery | Layer | 8 | 64 | 6.0 | 1 | 0.924445 [CI: NOT_MEASURED] | 0.972351 [CI: NOT_MEASURED] | NOT_ESTABLISHED |
| discovery | Token | 8 | 64 | 6.0 | 1 | 0.937511 [CI: NOT_MEASURED] | 0.977510 [CI: NOT_MEASURED] | NOT_ESTABLISHED |
| discovery | Layer | 12 | 64 | 6.0 | 1 | 0.927600 [CI: NOT_MEASURED] | 0.972447 [CI: NOT_MEASURED] | NOT_ESTABLISHED |
| discovery | Token | 12 | 64 | 6.0 | 1 | 0.920339 [CI: NOT_MEASURED] | 0.968209 [CI: NOT_MEASURED] | NOT_ESTABLISHED |
| discovery | Layer | 4 | 64 | 10.0 | 1 | 0.960325 [CI: NOT_MEASURED] | 0.973195 [CI: NOT_MEASURED] | NOT_ESTABLISHED |
| discovery | Token | 4 | 64 | 10.0 | 1 | 1.002548 [CI: NOT_MEASURED] | 1.016942 [CI: NOT_MEASURED] | NOT_ESTABLISHED |
| discovery | Layer | 8 | 64 | 10.0 | 1 | 0.964247 [CI: NOT_MEASURED] | 0.974237 [CI: NOT_MEASURED] | NOT_ESTABLISHED |
| discovery | Token | 8 | 64 | 10.0 | 1 | 0.964066 [CI: NOT_MEASURED] | 0.977716 [CI: NOT_MEASURED] | NOT_ESTABLISHED |
| discovery | Layer | 12 | 64 | 10.0 | 1 | 0.968129 [CI: NOT_MEASURED] | 0.974766 [CI: NOT_MEASURED] | NOT_ESTABLISHED |
| discovery | Token | 12 | 64 | 10.0 | 1 | 0.961060 [CI: NOT_MEASURED] | 0.970890 [CI: NOT_MEASURED] | NOT_ESTABLISHED |
| discovery | Layer | 4 | 128 | 2.0 | 1 | 0.961366 [CI: NOT_MEASURED] | 0.961499 [CI: NOT_MEASURED] | NOT_ESTABLISHED |
| discovery | Token | 4 | 128 | 2.0 | 1 | 0.977635 [CI: NOT_MEASURED] | 0.971032 [CI: NOT_MEASURED] | NOT_ESTABLISHED |
| discovery | Layer | 8 | 128 | 2.0 | 1 | 0.965143 [CI: NOT_MEASURED] | 0.962127 [CI: NOT_MEASURED] | NOT_ESTABLISHED |
| discovery | Token | 8 | 128 | 2.0 | 1 | 0.971124 [CI: NOT_MEASURED] | 0.967567 [CI: NOT_MEASURED] | NOT_ESTABLISHED |
| discovery | Layer | 12 | 128 | 2.0 | 1 | 0.966687 [CI: NOT_MEASURED] | 0.962292 [CI: NOT_MEASURED] | NOT_ESTABLISHED |
| discovery | Token | 12 | 128 | 2.0 | 1 | 0.965950 [CI: NOT_MEASURED] | 0.959064 [CI: NOT_MEASURED] | NOT_ESTABLISHED |
| discovery | Layer | 4 | 128 | 4.0 | 1 | 0.951339 [CI: NOT_MEASURED] | 0.988215 [CI: NOT_MEASURED] | NOT_ESTABLISHED |
| discovery | Token | 4 | 128 | 4.0 | 1 | 0.970305 [CI: NOT_MEASURED] | 0.997486 [CI: NOT_MEASURED] | NOT_ESTABLISHED |
| discovery | Layer | 8 | 128 | 4.0 | 1 | 0.953254 [CI: NOT_MEASURED] | 0.988387 [CI: NOT_MEASURED] | NOT_ESTABLISHED |
| discovery | Token | 8 | 128 | 4.0 | 1 | 0.956433 [CI: NOT_MEASURED] | 0.994586 [CI: NOT_MEASURED] | NOT_ESTABLISHED |
| discovery | Layer | 12 | 128 | 4.0 | 1 | 0.956346 [CI: NOT_MEASURED] | 0.988068 [CI: NOT_MEASURED] | NOT_ESTABLISHED |
| discovery | Token | 12 | 128 | 4.0 | 1 | 0.949182 [CI: NOT_MEASURED] | 0.985985 [CI: NOT_MEASURED] | NOT_ESTABLISHED |
| discovery | Layer | 4 | 128 | 6.0 | 1 | 0.955095 [CI: NOT_MEASURED] | 0.989385 [CI: NOT_MEASURED] | NOT_ESTABLISHED |
| discovery | Token | 4 | 128 | 6.0 | 1 | 0.973278 [CI: NOT_MEASURED] | 0.999956 [CI: NOT_MEASURED] | NOT_ESTABLISHED |
| discovery | Layer | 8 | 128 | 6.0 | 1 | 0.958747 [CI: NOT_MEASURED] | 0.989980 [CI: NOT_MEASURED] | NOT_ESTABLISHED |
| discovery | Token | 8 | 128 | 6.0 | 1 | 0.963106 [CI: NOT_MEASURED] | 0.995183 [CI: NOT_MEASURED] | NOT_ESTABLISHED |
| discovery | Layer | 12 | 128 | 6.0 | 1 | 0.961987 [CI: NOT_MEASURED] | 0.990424 [CI: NOT_MEASURED] | NOT_ESTABLISHED |
| discovery | Token | 12 | 128 | 6.0 | 1 | 0.960656 [CI: NOT_MEASURED] | 0.988319 [CI: NOT_MEASURED] | NOT_ESTABLISHED |
| discovery | Layer | 4 | 128 | 10.0 | 1 | 0.957849 [CI: NOT_MEASURED] | 0.989171 [CI: NOT_MEASURED] | NOT_ESTABLISHED |
| discovery | Token | 4 | 128 | 10.0 | 1 | 0.978706 [CI: NOT_MEASURED] | 0.999741 [CI: NOT_MEASURED] | NOT_ESTABLISHED |
| discovery | Layer | 8 | 128 | 10.0 | 1 | 0.962008 [CI: NOT_MEASURED] | 0.989754 [CI: NOT_MEASURED] | NOT_ESTABLISHED |
| discovery | Token | 8 | 128 | 10.0 | 1 | 0.964627 [CI: NOT_MEASURED] | 0.994075 [CI: NOT_MEASURED] | NOT_ESTABLISHED |
| discovery | Layer | 12 | 128 | 10.0 | 1 | 0.965316 [CI: NOT_MEASURED] | 0.990041 [CI: NOT_MEASURED] | NOT_ESTABLISHED |
| discovery | Token | 12 | 128 | 10.0 | 1 | 0.963661 [CI: NOT_MEASURED] | 0.987862 [CI: NOT_MEASURED] | NOT_ESTABLISHED |
| discovery | Layer | 4 | 200 | 2.0 | 1 | 0.975963 [CI: NOT_MEASURED] | 0.990544 [CI: NOT_MEASURED] | NOT_ESTABLISHED |
| discovery | Token | 4 | 200 | 2.0 | 1 | 0.992895 [CI: NOT_MEASURED] | 1.000410 [CI: NOT_MEASURED] | NOT_ESTABLISHED |
| discovery | Layer | 8 | 200 | 2.0 | 1 | 0.976898 [CI: NOT_MEASURED] | 0.990737 [CI: NOT_MEASURED] | NOT_ESTABLISHED |
| discovery | Token | 8 | 200 | 2.0 | 1 | 0.984310 [CI: NOT_MEASURED] | 0.997170 [CI: NOT_MEASURED] | NOT_ESTABLISHED |
| discovery | Layer | 12 | 200 | 2.0 | 1 | 0.978587 [CI: NOT_MEASURED] | 0.990873 [CI: NOT_MEASURED] | NOT_ESTABLISHED |
| discovery | Token | 12 | 200 | 2.0 | 1 | 0.974989 [CI: NOT_MEASURED] | 0.987239 [CI: NOT_MEASURED] | NOT_ESTABLISHED |
| discovery | Layer | 4 | 200 | 4.0 | 1 | 0.962549 [CI: NOT_MEASURED] | 0.990642 [CI: NOT_MEASURED] | NOT_ESTABLISHED |
| discovery | Token | 4 | 200 | 4.0 | 1 | 0.981874 [CI: NOT_MEASURED] | 1.000783 [CI: NOT_MEASURED] | NOT_ESTABLISHED |
| discovery | Layer | 8 | 200 | 4.0 | 1 | 0.965658 [CI: NOT_MEASURED] | 0.991113 [CI: NOT_MEASURED] | NOT_ESTABLISHED |
| discovery | Token | 8 | 200 | 4.0 | 1 | 0.970450 [CI: NOT_MEASURED] | 0.997396 [CI: NOT_MEASURED] | NOT_ESTABLISHED |
| discovery | Layer | 12 | 200 | 4.0 | 1 | 0.966581 [CI: NOT_MEASURED] | 0.991095 [CI: NOT_MEASURED] | NOT_ESTABLISHED |
| discovery | Token | 12 | 200 | 4.0 | 1 | 0.965529 [CI: NOT_MEASURED] | 0.987942 [CI: NOT_MEASURED] | NOT_ESTABLISHED |
| discovery | Layer | 4 | 200 | 6.0 | 1 | 0.971586 [CI: NOT_MEASURED] | 0.990044 [CI: NOT_MEASURED] | NOT_ESTABLISHED |
| discovery | Token | 4 | 200 | 6.0 | 1 | 0.997512 [CI: NOT_MEASURED] | 1.000246 [CI: NOT_MEASURED] | NOT_ESTABLISHED |
| discovery | Layer | 8 | 200 | 6.0 | 1 | 0.975010 [CI: NOT_MEASURED] | 0.990973 [CI: NOT_MEASURED] | NOT_ESTABLISHED |
| discovery | Token | 8 | 200 | 6.0 | 1 | 0.979881 [CI: NOT_MEASURED] | 0.995984 [CI: NOT_MEASURED] | NOT_ESTABLISHED |
| discovery | Layer | 12 | 200 | 6.0 | 1 | 0.976745 [CI: NOT_MEASURED] | 0.990447 [CI: NOT_MEASURED] | NOT_ESTABLISHED |
| discovery | Token | 12 | 200 | 6.0 | 1 | 0.972809 [CI: NOT_MEASURED] | 0.987578 [CI: NOT_MEASURED] | NOT_ESTABLISHED |
| discovery | Layer | 4 | 200 | 8.0 | 1 | 0.972959 [CI: NOT_MEASURED] | 0.990791 [CI: NOT_MEASURED] | NOT_ESTABLISHED |
| discovery | Token | 4 | 200 | 8.0 | 1 | 0.992135 [CI: NOT_MEASURED] | 1.000492 [CI: NOT_MEASURED] | NOT_ESTABLISHED |
| discovery | Layer | 8 | 200 | 8.0 | 1 | 0.975314 [CI: NOT_MEASURED] | 0.990806 [CI: NOT_MEASURED] | NOT_ESTABLISHED |
| discovery | Token | 8 | 200 | 8.0 | 1 | 0.982320 [CI: NOT_MEASURED] | 0.997968 [CI: NOT_MEASURED] | NOT_ESTABLISHED |
| discovery | Layer | 12 | 200 | 8.0 | 1 | 0.975323 [CI: NOT_MEASURED] | 0.991601 [CI: NOT_MEASURED] | NOT_ESTABLISHED |
| discovery | Token | 12 | 200 | 8.0 | 1 | 0.973201 [CI: NOT_MEASURED] | 0.987908 [CI: NOT_MEASURED] | NOT_ESTABLISHED |
| discovery | Layer | 4 | 200 | 10.0 | 1 | 0.973268 [CI: NOT_MEASURED] | 0.989233 [CI: NOT_MEASURED] | NOT_ESTABLISHED |
| discovery | Token | 4 | 200 | 10.0 | 1 | 0.992642 [CI: NOT_MEASURED] | 0.999284 [CI: NOT_MEASURED] | NOT_ESTABLISHED |
| discovery | Layer | 8 | 200 | 10.0 | 1 | 0.974669 [CI: NOT_MEASURED] | 0.989196 [CI: NOT_MEASURED] | NOT_ESTABLISHED |
| discovery | Token | 8 | 200 | 10.0 | 1 | 0.978850 [CI: NOT_MEASURED] | 0.995606 [CI: NOT_MEASURED] | NOT_ESTABLISHED |
| discovery | Layer | 12 | 200 | 10.0 | 1 | 0.975670 [CI: NOT_MEASURED] | 0.989435 [CI: NOT_MEASURED] | NOT_ESTABLISHED |
| discovery | Token | 12 | 200 | 10.0 | 1 | 0.971832 [CI: NOT_MEASURED] | 0.985941 [CI: NOT_MEASURED] | NOT_ESTABLISHED |

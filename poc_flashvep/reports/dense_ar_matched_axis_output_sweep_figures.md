# Matched-axis Figure와 재현 위치

## 환경과 증거 계약

Model은 Qwen/Qwen3-8B이다. Revision은 `b968826d9c46dd6066d109eabc6255188de91218`이다.
Original BF16, Exact Causal Attention, Greedy, Thinking Disabled와 Full36 Decode를 유지했다.
Input은 8000 Tokens이다. `ignore_eos=True`를 사용했다.
GPU0/1은 H100 SXM 80GB이다. GPU당 KV Budget은 42GiB이다.
GPU UUID는 `GPU-f217c8a0-1142-20f4-d84b-af29f3a47a0d`와 `GPU-a77f3471-67d4-20b0-9fab-e502d4de5adb`이다.
vLLM은 0.26.0+cu129이다. PyTorch는 2.11.0+cu129이다. NIXL은 1.4.1이다.
Prefix Cache는 비활성화했다. 모든 Worker는 실제 B8192/S1024 Gate를 통과했다.

Publication Parent는 `228a041f35cedcee9f45beba6c462b453fb3a2c5`이다. Branch는 `flashvep/dense-ar-matched-axis-crossover-20261117`이다.
Parent SHA를 모든 실행의 실제 Source SHA라고 표시하지 않는다.
각 `raw/<name>/command.json`에 실행한 Source ZIP의 SHA256을 저장했다.
Model Revision, Config Hash, GPU Mapping, Dataset Hash와 SLO Hash도 실행별로 저장했다.
기존 Inference Adapter 8개와 Predecessor Source 523개는 변경하지 않았다.
사용자 Checkout, Staged Index와 설치 Runtime도 보존했다.

## 사용자 요청에 따른 범위 변경

초기 공통 Admission8 Discovery는 198/232회 완료했다.
초기 Admission 대조는 52/52회 완료했다.
실제 배치 일치 요청에 따라 나머지 Discovery 34회와 기존 Confirmation 120회는 보류했다.
이 조건은 `NOT_RUN`이다. 완료로 표시하지 않았다.
기존 Native 설정 Bridge 6회도 `NOT_RUN`이다.
따라서 기존 Native 32000/4·128000/16 대비 B/S만의 인과 효과는 `NOT_MEASURED`이다.
Last2 전체 Sweep은 취소 상태를 유지했다.
이전 소규모 Last2 Forward 진단은 별도 증거로 보존했다.

새 Batch1 성능은 98/98회 완료했다.
새 준비1/Decode2 성능은 42/42회 완료했다.
두 Protocol 모두 Native와 Last4/8/12의 Layer/Token, 총 7개 Arm을 사용했다.
P 준비 배치와 D Suffix 준비 배치는 항상 실제 요청 1개였다.
Decode 배치만 Protocol에 따라 실제 요청 1개 또는 2개였다.
두 축의 실제 Query 수, Padding, Source/Destination Group과 작업량을 검사했다.
B8192/S1024만 같다는 이유로 실제 배치도 같다고 판단하지 않았다.

## 정확한 Partition과 작업량 — STRUCTURAL_ACCOUNTING

| Pair | Layer P/D layers | Token Suffix histogram / N144 | P GEMM rows/request | D GEMM rows/request |
| --- | --- | --- | --- | --- |
| 4 | 32/4 | {"880": 64, "896": 80} | 256000 | 32000 |
| 8 | 28/8 | {"1776": 128, "1792": 16} | 224000 | 64000 |
| 12 | 24/12 | {"2656": 48, "2672": 96} | 192000 | 96000 |

표의 Token Suffix는 16-Token 경계에 맞춘 인접 길이 조합이다.
N72는 같은 분포를 절반의 요청 수로 사용한다.
평균 Suffix는 각각 888.888889/1777.777778/2666.666667 Tokens이다.
실행별 Query Coverage로 전체 P/D Rows 일치를 다시 검사했다.
같은 GEMM Rows는 같은 Attention FLOPs나 GPU 시간을 뜻하지 않는다.

## 표본, 정합성과 SLO

Discovery는 N72이며 조건별 독립 Start 1회이다.
Batch1은 Output16의 QPS2/4/6/8/10과 Output64의 QPS2/6/10을 사용했다.
Decode2는 Output16의 QPS2/6/10을 사용했다.
Discovery Arrival Seed는 2026111810이다.
Confirmation은 QPS6, N144이며 조건별 독립 Start 3회이다.
Confirmation Seed는 2026111820/2026111821/2026111822이다.
Discovery와 Confirmation을 CI 계산에서 합치지 않았다.
기존 실제 문서 Pool인 `matched144.json`을 사용했다.
새 Arrival Seed 검증이며, 미관측 Prompt Pool 검증은 아니다.

각 새 Protocol은 N18 Full-state Gate를 7개 Arm에서 통과했다.
Full Prompt KV, Hidden, Greedy Output, Layer 방문과 Lifetime을 검사했다.
성능 N144에서는 Output, Query/Layer Coverage, 작업량과 Retirement를 검사했다.
모든 N144 Timing 요청의 Full KV를 CPU로 복사하여 Hash한 것은 아니다.
Decode2의 초기 Version1/2 실패는 삭제하지 않았다.
Version3은 모든 State/Geometry Gate 뒤에만 성능을 실행했다.
첫 Token Cohort Barrier 대기는 Client TTFT에 포함했다.
원래 Generation Timestamp와 Token Count는 변경하지 않았다.

기존 Frozen SLO를 변경하지 않았다.
표의 공통 Balanced SLO는 TTFT≤5000ms, 모든 ITL≤34ms, TPOT≤33ms이다.
Output별 기존 Frozen SLO도 별도 JSON에 보존했다.
Goodput의 분자는 세 조건을 모두 통과한 요청의 실제 Output Tokens이다.
ITL은 첫 Interval을 포함하는 Client-visible 연속 Token 간격이다.
Mean TBT는 모든 해당 Client-visible Interval의 산술 평균이다.
Request TPOT는 요청의 첫 Token부터 마지막 Token까지의 시간을 Output Tokens−1로 나눈 값이다.
Raw token/s와 SLO token/s의 분모는 유한 Cohort의 Arrival-to-drain 시간이다.
이를 Sustainable Capacity라고 주장하지 않는다.

CI의 재표본 단위는 독립 Server Start이다.
요청이나 Token을 독립 Start로 취급하지 않았다.
표의 p95/p99는 Start 안의 분위수를 Start 사이에서 평균한 값이다.
Type7 Bootstrap CI와 역CDF 민감도 CI를 함께 제시한다.
3개 Start의 CI는 모집단 Coverage나 WCET 보장을 뜻하지 않는다.
CI는 Pointwise이다. 다중 비교 보정은 하지 않았다.
Native-relative 보호는 ITL p99와 TPOT p95의 두 CI 상한이 모두 1.05 이하여야 한다.
Absolute SLO 통과와 Native-relative 보호는 다른 Gate이다.

원시 결과는 `dense_ar_matched_axis_crossover_20261117/`에 보존했다.
`analysis/full_execution_matrix.csv`는 실행 상태 전체를 포함한다.
`analysis/per_request_metrics_and_work.csv`는 390개 성능 Start의 45,000개 요청을 포함한다.
`analysis/independent_start_statistics.json`에 반올림하지 않은 값과 CI를 저장했다.
검증은 `analysis/final_integrity_validation.json`에서 확인한다.

## Figure 목록

- [dense_ar_cohort2_batch_matched_four_metrics_o16.png](dense_ar_matched_axis_crossover_20261117/figures/dense_ar_cohort2_batch_matched_four_metrics_o16.png) — MEASURED
- [dense_ar_actual_batch_matched_confirmation_ratios.png](dense_ar_matched_axis_crossover_20261117/figures/dense_ar_actual_batch_matched_confirmation_ratios.png) — MEASURED
- [dense_ar_cohort2_confirmation_metrics.png](dense_ar_matched_axis_crossover_20261117/figures/dense_ar_cohort2_confirmation_metrics.png) — MEASURED
- [dense_ar_matched_cohort2_host_critical_path.png](dense_ar_matched_axis_crossover_20261117/figures/dense_ar_matched_cohort2_host_critical_path.png) — TRACE_ANALYSIS
- [dense_ar_actual_serving_projection_shape_efficiency.png](dense_ar_matched_axis_crossover_20261117/figures/dense_ar_actual_serving_projection_shape_efficiency.png) — MEASURED
- [dense_ar_unified_four_metrics_o200_last4.png](dense_ar_matched_axis_crossover_20261117/figures/dense_ar_unified_four_metrics_o200_last4.png) — MEASURED
- [dense_ar_unified_four_metrics_o200_last8.png](dense_ar_matched_axis_crossover_20261117/figures/dense_ar_unified_four_metrics_o200_last8.png) — MEASURED
- [dense_ar_unified_four_metrics_o200_last12.png](dense_ar_matched_axis_crossover_20261117/figures/dense_ar_unified_four_metrics_o200_last12.png) — MEASURED

PNG와 PDF를 함께 저장했다.
주요 Figure는 사용자 요청 경로인 `poc_flashvep/`에도 저장했다.
실제 측정 점만 Marker로 표시했다.
단일 Start의 점에는 독립 Start CI가 없다.
Ratio Figure의 넓은 회색 범위는 역CDF 민감도 CI이다.
색 범위는 원래 Type7 CI이다.
Host 평균 구간의 누적 그림은 GPU Kernel Timeline이 아니다.
고립 GEMM 확장 모델은 Online Serving 실측과 분리했다.
기존 계측 Gate 실패 Timeline을 성공한 Physical Idle 그림으로 바꾸지 않았다.

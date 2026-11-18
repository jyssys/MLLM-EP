# 배치 일치 Layer/Token 순위와 원인 분석

## 핵심 판정

원했던 작은 Cut Layer 우위·큰 Cut Token 우위는 이번 Online 대조에서 입증하지 못했다.
실제 배치를 맞추면 이전 자유 배치 결과와 다른 순위가 나타났다.
따라서 배치 Geometry는 중요한 혼동 변수였다.
그러나 이것만으로 모든 차이의 유일한 원인을 확정하지 않는다.

Batch1의 Output16/QPS6에서 Last4 차이는 불확실했다.
Last8과 Last12에서는 Layer의 TTFT p95와 SLO-goodput이 우세했다.
Output64에서는 차이가 대체로 작아졌다.
Last4 TTFT의 경계 판정은 CI 분위수 정의에 민감했다.
Last12의 TTFT 우위는 존재했지만 Native-relative 보호가 모든 Gate를 통과하지는 못했다.

준비1/Decode2의 Output16/QPS6에서는 반대 방향의 Partition-dependent 역전이 나타났다.
Last4는 Token의 TTFT p95가 낮았다.
Last8과 Last12는 Layer의 TTFT p95가 낮았다.
이 방향은 사용자가 처음 제시한 작은 Layer·큰 Token 가설과 반대이다.
세 Pair의 Native-relative Decode 보호는 Type7과 역CDF Gate를 모두 통과했다.
하지만 Joint SLO Attainment는 모두 95% 미만이었다.
따라서 성공한 지속 가능 SLO Serving으로 표현하지 않는다.

판정은 `MEASURED_CONDITIONAL_AXIS_BENEFIT`과 `DESIRED_CROSSOVER_NOT_ESTABLISHED`이다.
동일 배치에서 Layer의 조건부 Online 이득은 존재한다.
그 이득을 순수 GEMM 효율이나 Loop-specific 효과로 환원할 근거는 부족하다.
Strong Colocated 대비 우위나 Dynamic 정책의 독립 기여는 이번 대조에서 검증하지 않았다.

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

## 계산 이득과 TTFT의 차이

작은 Cut의 실제 D Forward 이득은 이전 직렬 진단에서 확인했다.
Last4는 Layer30.3130ms, Token38.3212ms였다.
그러나 같은 진단의 Client TTFT p95는 318.0027ms와329.9976ms였다.
각 방식은 독립 Start 1회이다. 95% CI는 없다.
Forward의 비율을 그대로 전체 TTFT 비율로 바꾸지 않는다.

현재 TTFT에는 P 제출 대기와 D Admission 대기가 포함된다.
이전 Cohort의 Decode Residency는 다음 요청의 D 대기를 늘릴 수 있다.
준비1/Decode2에서는 준비와 Decode Group을 분리했다.
그 변경은 Batch1과 다른 실행 구조이다.
실제 같은 배치를 사용해도 Suffix Shape와 P/D Pipeline Balance는 다르다.

아래 표는 Host Clock의 평균 구간 회계이다.
처음 다섯 구간의 합은 Mean TTFT와 일치한다.
마지막 두 Cohort 지표는 그 합에 추가하는 항목이 아니다.
특히 Decode2 Worker Residency에는 남은 준비와 Barrier 전 시간이 섞일 수 있다.
Exclusive Decode Kernel Time으로 해석하지 않는다.
구간별 p95를 더하거나 빼서 TTFT p95를 분해하지 않는다.

| Stage | Output | Method | Last | Arrival→P ms | P→Ready ms | Ready→D ms | D→First worker ms | Worker→Client ms | D prep/cohort request ms | Worker residency/cohort request ms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| batch1_confirmation | 16 | Native | 0 | 9513.885 | 292.474 | 4524.224 | 24.417 | 0.291 | 24.417 | 343.879 |
| batch1_confirmation | 16 | Layer | 4 | 8592.722 | 260.370 | 8363.721 | 57.356 | 0.285 | 57.356 | 352.524 |
| batch1_confirmation | 16 | Token | 4 | 8427.104 | 258.322 | 8350.410 | 55.497 | 0.246 | 55.497 | 352.314 |
| batch1_confirmation | 16 | Layer | 8 | 8537.172 | 228.466 | 9801.653 | 75.929 | 0.237 | 75.929 | 354.081 |
| batch1_confirmation | 16 | Token | 8 | 9445.567 | 226.013 | 10589.506 | 101.723 | 0.255 | 101.723 | 352.367 |
| batch1_confirmation | 16 | Layer | 12 | 9923.310 | 202.608 | 11276.649 | 116.498 | 0.286 | 116.498 | 354.181 |
| batch1_confirmation | 16 | Token | 12 | 10377.479 | 191.868 | 11971.283 | 131.250 | 0.272 | 131.250 | 354.199 |
| batch1_confirmation | 64 | Native | 0 | 53192.232 | 286.998 | 38731.259 | 15.395 | 0.242 | 15.395 | 1443.450 |
| batch1_confirmation | 64 | Layer | 4 | 56139.789 | 262.261 | 40821.888 | 53.921 | 0.264 | 53.921 | 1476.286 |
| batch1_confirmation | 64 | Token | 4 | 56905.666 | 255.286 | 41361.533 | 72.871 | 0.262 | 72.871 | 1475.177 |
| batch1_confirmation | 64 | Layer | 8 | 57492.521 | 225.961 | 42049.936 | 90.503 | 0.295 | 90.503 | 1477.137 |
| batch1_confirmation | 64 | Token | 8 | 57573.067 | 224.346 | 42084.570 | 96.646 | 0.248 | 96.646 | 1471.810 |
| batch1_confirmation | 64 | Layer | 12 | 58417.258 | 196.372 | 42826.216 | 113.388 | 0.270 | 113.388 | 1476.812 |
| batch1_confirmation | 64 | Token | 12 | 59235.196 | 192.699 | 43354.455 | 131.381 | 0.258 | 131.381 | 1477.910 |
| cohort2_confirmation | 16 | Native | 0 | 9255.193 | 291.484 | 153.928 | 33.064 | 0.390 | 16.519 | 171.830 |
| cohort2_confirmation | 16 | Layer | 4 | 7726.850 | 266.994 | 142.604 | 84.983 | 16.243 | 34.589 | 191.525 |
| cohort2_confirmation | 16 | Token | 4 | 7369.846 | 262.767 | 139.417 | 121.273 | 7.209 | 57.214 | 182.445 |
| cohort2_confirmation | 16 | Layer | 8 | 5694.588 | 232.308 | 391.854 | 125.347 | 32.220 | 46.755 | 207.797 |
| cohort2_confirmation | 16 | Token | 8 | 5461.801 | 227.760 | 1595.631 | 151.272 | 37.390 | 57.128 | 212.246 |
| cohort2_confirmation | 16 | Layer | 12 | 3875.079 | 203.241 | 4290.993 | 171.676 | 48.143 | 61.985 | 223.328 |
| cohort2_confirmation | 16 | Token | 12 | 3942.396 | 194.373 | 5871.607 | 211.562 | 54.079 | 78.944 | 230.083 |

## GEMM, Attention과 전송

기존 고립 GEMM은 F.linear를 사용했다.
현재 Serving은 Batch-invariant Linear를 직접 호출한다.
H100에서 두 호출 경로는 같지 않았다.
새 고립 대조는 실제 Serving API를 사용했다.
이 과정에서 Model Weight, Precision과 Serving 수학을 바꾸지 않았다.

Layer0 Projection 중앙값을 확장한 GPU1 비용 모델에서 Token/Layer 비율은 다음과 같다.
Last4는 1.1044이다. Last8은 1.0749이다. Last12는 1.0486이다.
이는 `STRUCTURAL_ACCOUNTING`이며 Online Exclusive GEMM 실측이 아니다.
이 모델은 이득 감소를 보이지만 큰 Cut의 Token 역전을 예측하지 않는다.
이전 Loop의 Token64 결과를 이번 Dense Token889에 그대로 적용할 수 없다.

동일 GEMM Rows에서도 D의 Attention 분배는 다르다.
Token/Layer의 D Causal Pair 비율은 Last4 약1.889, Last8 약1.778, Last12 약1.667이다.
전체 시스템의 Attention Pair 합은 같다.
Pair 비율을 GPU 시간 비율로 표시하지 않는다.

같은 Pair의 유효 KV Payload 합은 Layer와 Token에서 같다.
Layer는 Request당 Hidden/Residual131072000Bytes를 추가 전송한다.
전송 API 경과시간을 순수 DMA 시간으로 표시하지 않는다.
KV Payload 합이 같아도 Credit 점유 기간과 Admission 대기는 다를 수 있다.

Clean Online Profiler의 전체 5% 간섭 Gate는 통과하지 못했다.
따라서 Online Exclusive GEMM, Attention, SM/HBM Counter는 `NOT_MEASURED`이다.
기존 Kernel Timeline은 `DIAGNOSTIC`으로 보존한다.
새 고립 API 대조의 16개 Profiler Gate 통과는 Online Gate 통과가 아니다.

현재 원인 판정은 `PARTIAL_ROOT_CAUSE_IDENTIFICATION`이다.
실제 배치와 Cohort 구조의 영향 및 긴 Host 대기는 확인했다.
전체 Online 차이를 GEMM·Attention·Transfer의 독점적 GPU 시간으로 분해하지는 못했다.
원하는 방향의 Crossover를 GEMM 효율만으로 설명할 수 있다는 주장은 미입증이다.

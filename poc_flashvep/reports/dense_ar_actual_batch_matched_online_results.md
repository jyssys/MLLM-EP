# 실제 Batch 일치 Layer/Token Online 결과

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

## 독립 Confirmation 결과 — MEASURED

| Stage | Output | Method | Last | Starts | TTFT p95 ms | ITL p99 ms | TPOT p95 ms | Raw output tokens/s | Joint SLO % | SLO output tokens/s |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| batch1_confirmation | 16 | Layer | 4 | 3 | 31204.6286 | 24.2213 | 23.8537 | 38.7110 | 13.4259 | 5.2097 |
| batch1_confirmation | 64 | Layer | 4 | 3 | 183328.1296 | 24.0706 | 23.7596 | 41.7421 | 3.0093 | 1.2564 |
| batch1_confirmation | 16 | Layer | 8 | 3 | 33852.6493 | 24.5479 | 23.9367 | 36.9462 | 12.5000 | 4.6183 |
| batch1_confirmation | 64 | Layer | 8 | 3 | 188398.9310 | 24.2333 | 23.7763 | 40.7478 | 2.7778 | 1.1319 |
| batch1_confirmation | 16 | Layer | 12 | 3 | 39356.1560 | 24.6507 | 23.9594 | 33.8135 | 10.4167 | 3.5172 |
| batch1_confirmation | 64 | Layer | 12 | 3 | 191511.7528 | 24.3863 | 23.7791 | 40.1757 | 2.7778 | 1.1160 |
| batch1_confirmation | 16 | Native | 0 | 3 | 25665.3896 | 23.4822 | 22.9478 | 42.9633 | 15.0463 | 6.4810 |
| batch1_confirmation | 64 | Native | 0 | 3 | 173656.8610 | 23.2629 | 22.9186 | 43.7590 | 3.0093 | 1.3169 |
| batch1_confirmation | 16 | Token | 4 | 3 | 30878.7974 | 24.1453 | 23.8375 | 38.9276 | 13.1944 | 5.1346 |
| batch1_confirmation | 64 | Token | 4 | 3 | 185816.3574 | 24.0710 | 23.7583 | 41.2579 | 3.0093 | 1.2416 |
| batch1_confirmation | 16 | Token | 8 | 3 | 37147.0691 | 24.2941 | 23.9069 | 35.0043 | 11.3426 | 3.9697 |
| batch1_confirmation | 64 | Token | 8 | 3 | 188512.7420 | 24.0819 | 23.7708 | 40.7283 | 2.7778 | 1.1313 |
| batch1_confirmation | 16 | Token | 12 | 3 | 41406.3038 | 24.5657 | 23.9497 | 32.7740 | 9.9537 | 3.2612 |
| batch1_confirmation | 64 | Token | 12 | 3 | 194049.4412 | 24.3249 | 23.7987 | 39.6998 | 2.7778 | 1.1028 |
| cohort2_confirmation | 16 | Layer | 4 | 3 | 13865.2538 | 24.3006 | 23.8327 | 56.1156 | 28.0093 | 15.6910 |
| cohort2_confirmation | 16 | Layer | 8 | 3 | 10446.6958 | 24.4725 | 23.8548 | 61.6289 | 33.7963 | 20.8301 |
| cohort2_confirmation | 16 | Layer | 12 | 3 | 14543.7588 | 24.3349 | 23.8038 | 55.1764 | 26.8519 | 14.8041 |
| cohort2_confirmation | 16 | Native | 0 | 3 | 16800.0625 | 23.5421 | 22.9436 | 52.1911 | 24.7685 | 12.9276 |
| cohort2_confirmation | 16 | Token | 4 | 3 | 13203.4391 | 24.1894 | 23.7731 | 57.0993 | 28.4722 | 16.2261 |
| cohort2_confirmation | 16 | Token | 8 | 3 | 12425.8396 | 24.1492 | 23.7419 | 58.3001 | 30.0926 | 17.5359 |
| cohort2_confirmation | 16 | Token | 12 | 3 | 17768.6428 | 24.5549 | 23.8847 | 51.0088 | 23.6111 | 12.0434 |

## 동일 작업량의 Layer/Token 비율

| Stage | Output | Last | Starts | Layer/Token TTFT p95 [Type7 95% CI] | TTFT 역CDF 95% CI | Layer/Token raw token/s [95% CI] | Layer/Token common-SLO token/s [95% CI] |
| --- | --- | --- | --- | --- | --- | --- | --- |
| batch1_confirmation | 16 | 4 | 3 | 1.010552 [0.974517, 1.037809] | [0.957404909295221, 1.0382677440039785] | 0.994436 [0.979684, 1.014287] | 1.014640 [0.979665, 1.055354] |
| batch1_confirmation | 16 | 8 | 3 | 0.911314 [0.907757, 0.915121] | [0.9070835911438708, 0.9164667741126453] | 1.055477 [1.054114, 1.056581] | 1.163394 [1.143570, 1.177830] |
| batch1_confirmation | 16 | 12 | 3 | 0.950487 [0.921580, 0.979349] | [0.9123321092601908, 0.9861961703384439] | 1.031716 [1.012211, 1.052647] | 1.078487 [1.027480, 1.131814] |
| batch1_confirmation | 64 | 4 | 3 | 0.986609 [0.977698, 0.996619] | [0.976106791906767, 1.000251040918798] | 1.011736 [1.002826, 1.019791] | 1.011940 [1.003317, 1.019560] |
| batch1_confirmation | 64 | 8 | 3 | 0.999396 [0.995788, 1.003943] | [0.9955492503380471, 1.0059681924159618] | 1.000478 [0.996838, 1.003230] | 1.000478 [0.996838, 1.003230] |
| batch1_confirmation | 64 | 12 | 3 | 0.986922 [0.982243, 0.990557] | [0.9801326382313703, 0.9907466398227036] | 1.011987 [1.008728, 1.016283] | 1.011987 [1.008728, 1.016283] |
| cohort2_confirmation | 16 | 4 | 3 | 1.050124 [1.033581, 1.063857] | [1.0261124774248518, 1.0650060835414916] | 0.982772 [0.978537, 0.988331] | 0.967024 [0.952629, 0.976421] |
| cohort2_confirmation | 16 | 8 | 3 | 0.840724 [0.823533, 0.856483] | [0.8184656045074666, 0.861568591855603] | 1.057098 [1.052382, 1.061578] | 1.187849 [1.139845, 1.213652] |
| cohort2_confirmation | 16 | 12 | 3 | 0.818507 [0.792987, 0.845625] | [0.7881140260540115, 0.8560399142343355] | 1.081705 [1.069309, 1.091063] | 1.229236 [1.218895, 1.240764] |

## Native-relative Decode Gate

| Stage | Output | Candidate | Last | ITL p99 / Native [95% CI] | TPOT p95 / Native [95% CI] | Type7 Gate | 역CDF Gate |
| --- | --- | --- | --- | --- | --- | --- | --- |
| batch1_confirmation | 16 | Layer | 4 | 1.031474 [1.028192, 1.035079] | 1.039476 [1.039041, 1.039928] | PASS | PASS |
| batch1_confirmation | 16 | Token | 4 | 1.028240 [1.025509, 1.031194] | 1.038773 [1.038593, 1.038948] | PASS | PASS |
| batch1_confirmation | 16 | Layer | 8 | 1.045381 [1.041135, 1.049237] | 1.043095 [1.042484, 1.043714] | PASS | PASS |
| batch1_confirmation | 16 | Token | 8 | 1.034577 [1.029600, 1.039388] | 1.041795 [1.040862, 1.042712] | PASS | PASS |
| batch1_confirmation | 16 | Layer | 12 | 1.049760 [1.044449, 1.055573] | 1.044082 [1.043488, 1.044764] | NOT_ESTABLISHED | NOT_ESTABLISHED |
| batch1_confirmation | 16 | Token | 12 | 1.046139 [1.041108, 1.050433] | 1.043659 [1.042920, 1.044506] | NOT_ESTABLISHED | NOT_ESTABLISHED |
| batch1_confirmation | 64 | Layer | 4 | 1.034720 [1.034131, 1.035318] | 1.036695 [1.036414, 1.036937] | PASS | PASS |
| batch1_confirmation | 64 | Token | 4 | 1.034739 [1.034007, 1.035491] | 1.036637 [1.036246, 1.037095] | PASS | PASS |
| batch1_confirmation | 64 | Layer | 8 | 1.041713 [1.040964, 1.042466] | 1.037425 [1.037037, 1.037889] | PASS | PASS |
| batch1_confirmation | 64 | Token | 8 | 1.035208 [1.034207, 1.036518] | 1.037181 [1.036838, 1.037436] | PASS | PASS |
| batch1_confirmation | 64 | Layer | 12 | 1.048290 [1.046047, 1.050249] | 1.037547 [1.037256, 1.037872] | NOT_ESTABLISHED | NOT_ESTABLISHED |
| batch1_confirmation | 64 | Token | 12 | 1.045651 [1.044390, 1.046812] | 1.038399 [1.038111, 1.038752] | PASS | PASS |
| cohort2_confirmation | 16 | Layer | 4 | 1.032217 [1.028966, 1.035290] | 1.038754 [1.037784, 1.039712] | PASS | PASS |
| cohort2_confirmation | 16 | Token | 4 | 1.027494 [1.023368, 1.031521] | 1.036157 [1.034913, 1.037569] | PASS | PASS |
| cohort2_confirmation | 16 | Layer | 8 | 1.039519 [1.034999, 1.043229] | 1.039718 [1.038263, 1.041098] | PASS | PASS |
| cohort2_confirmation | 16 | Token | 8 | 1.025787 [1.019768, 1.032156] | 1.034795 [1.033368, 1.036122] | PASS | PASS |
| cohort2_confirmation | 16 | Layer | 12 | 1.033674 [1.030408, 1.037012] | 1.037496 [1.036465, 1.038486] | PASS | PASS |
| cohort2_confirmation | 16 | Token | 12 | 1.043019 [1.039002, 1.047222] | 1.041021 [1.040181, 1.042044] | PASS | PASS |

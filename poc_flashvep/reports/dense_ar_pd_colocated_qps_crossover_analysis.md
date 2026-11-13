# Dense AR PD와 Colocated의 QPS 탐색 분석

## 판정과 범위

**`NO_CONFIRMED_PD_COLOCATED_CROSSOVER_ON_TESTED_GRID`**.

단일 Seed의 성능 Trade-off 후보는 존재한다.
그러나 요청한 물리적·인과적 Crossover는 입증하지 못했다.
이 판정은 검사한 Runtime과 Workload에 한정한다.
일반 Dense AR에서 Crossover가 없다는 뜻은 아니다.

이 문서는 Discovery 실행만 분석한다.
Held-out 반복 결과는 [최종 보고서](dense_ar_dynamic_layer_pd_crossover_final_summary.md)에 분리한다.
Discovery의 좋은 결과를 독립 Confirmation에 포함하지 않는다.

## 동일 조건과 측정 정의

- Qwen3-8B Revision은 `b968826d9c46dd6066d109eabc6255188de91218`이다.
- 원본 BF16과 36개 Layer를 유지했다.
- GPU0/1의 1P1D와 Full-replica DP2를 비교했다.
- GPU당 KV Budget은 42GiB이다.
- Input은 Disabled-thinking Suffix를 포함한 실제 8000 Tokens이다.
- 각 Cell은 N64이다. 같은 Output과 QPS의 Arm은 같은 Prompt와 Poisson Arrival을 사용했다.
- Output은 Ignore EOS로 고정했다.
- Prefix Caching을 사용하지 않았다.

TTFT와 ITL은 Coordinator의 Token Delivery 시각으로 계산했다.
TPOT는 첫 Token부터 마지막 Token까지의 시간을 `Output−1`로 나눈 값이다.
ITL에는 첫 Interval을 포함했다.
Service TTFT는 `NOT_MEASURED`이다.
Completed QPS와 Token/s는 Finite-cohort 관측값이다.
이 수치를 Sustainable Capacity로 해석하지 않는다.

## QPS와 Output 탐색 결과

아래 값의 단위는 모두 ms이다.
각 값은 단일 독립 Start의 Percentile이다.
Seed 간 신뢰구간은 이 표에 없다.

| Output | Offered QPS | Native TTFT p95 | DP2 TTFT p95 | Native ITL p99 | DP2 ITL p99 | Native TPOT p95 | DP2 TPOT p95 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 16 | 2.477 | 1281.77 | 886.04 | 28.64 | 27.49 | 25.96 | 27.03 |
| 16 | 4.777 | 3948.74 | 1714.29 | 29.65 | 29.53 | 24.29 | 27.11 |
| 16 | 6.016 | 7041.77 | 2674.31 | 30.85 | 32.84 | 26.23 | 27.52 |
| 32 | 2.477 | 1638.40 | 1512.50 | 29.00 | 27.74 | 26.02 | 26.98 |
| 32 | 4.777 | 4671.29 | 3134.69 | 32.29 | 34.04 | 27.47 | 27.91 |
| 32 | 6.016 | 7024.26 | 4350.72 | 31.74 | 34.55 | 26.41 | 28.91 |
| 64 | 2.477 | 2643.98 | 3196.11 | 29.92 | 29.04 | 27.53 | 27.16 |
| 64 | 4.777 | 6108.41 | 5715.75 | 33.67 | 34.38 | 28.23 | 28.81 |
| 64 | 6.016 | 7940.26 | 6760.20 | 33.91 | 38.49 | 28.28 | 29.61 |
| 128 | 2.477 | 4491.79 | 6107.81 | 26.51 | 29.68 | 26.04 | 27.97 |
| 128 | 4.777 | 8510.46 | 10022.93 | 31.61 | 32.51 | 30.82 | 29.79 |
| 128 | 6.016 | 11321.09 | 10386.19 | 31.47 | 32.77 | 30.88 | 31.06 |
| 200 | 2.477 | 7631.52 | 9424.56 | 31.29 | 29.02 | 30.75 | 28.63 |
| 200 | 4.777 | 15105.42 | 13040.25 | 31.43 | 31.67 | 30.72 | 30.60 |
| 200 | 6.016 | 15684.36 | 14393.32 | 28.70 | 32.57 | 28.17 | 30.96 |

이 표의 Native는 같은 실행 계열의 Discovery Anchor이다.
나중에 측정한 더 강한 No-migration Controller는 포함하지 않았다.
최종 비교에서는 Discovery에서 선정한 가장 강한 적격 Native를 사용한다.

기계가 읽을 수 있는 원문은 [Crossover Accounting](dense_ar_dynamic_layer_pd_crossover_20261113/analysis/crossover_discovery_accounting.json)에 저장했다.
각 Cell의 Configuration과 Raw Namespace도 같은 Evidence Directory에 저장했다.

## Baseline 강화와 정합성 제약

Colocated에서 Cohort4/8/16, Round-robin, Load-aware Routing, HND/NHD Layout을 비교했다.
Native-relative Decode 5% 조건을 Colocated의 Tuning 제약으로 사용하지 않았다.
Frozen Balanced Joint Goodput과 TTFT로 적격 설정을 선정했다.
Native와 Colocated에 같은 출력 정합성 조건을 적용했다.

Continuous Chunked Prefill도 실행했다.
Chunk128/256/512/1024/2048을 검사했다.
적격 Native와 Greedy Output이 일치하지 않은 설정은 성능 비교에서 제외했다.
HND의 Chunk512/1024/2048에서는 각각 8/64, 6/64, 3/64 Requests의 Output이 달랐다.
Prompt Query는 요청마다 8000 Tokens를 한 번씩 처리했다.
불일치의 정확한 Kernel-level 원인은 `NOT_IDENTIFIED`이다.
이 결과는 Chunked Prefill 일반의 수학적 오류를 뜻하지 않는다.

Compile3에서도 Native N8과 Colocated N8의 Output이 일치하지 않았다.
본 평가에는 Compile0의 적격 수치 실행 계열을 사용했다.
본 비교 대상은 이 정합성 조건에서 적격 판정을 받은 가장 강한 Cohort DP2이다.
최적의 Continuous Colocated를 이겼다는 증거는 아니다.

## P의 계산 압력

P-only Prefill Capacity는 Batch4/8/16에서 3.5978/3.5940/3.5857 Requests/s였다.
각 값은 같은 Start 안의 2개 Cohort 반복으로 구했다.
독립 Start 사이의 Capacity 신뢰구간은 아니다.
큰 P Batch가 Capacity를 높인다는 증거는 없었다.

Output16, QPS4.777의 Lean Native에서 관측한 P-ready Queue의 최댓값은 8이었다.
Queue 평균은 관측 전반부 0.364에서 후반부 5.571로 증가했다.
P/D의 KV Credit Refusal은 0이었다.
관측한 최대 KV 사용률은 P31.4%, D10.5%였다.
Host Source-production Rate는 3.485 Requests/s였다.
이는 독립 P Capacity의 약 96.9%이다.

이 값들은 Host Queue와 Service Pressure의 증거이다.
Qualified Clean P Kernel Busy가 80% 이상이었다는 증거는 아니다.
다른 Start나 Diagnostic Capture의 높은 Busy 값을 합쳐 Condition P를 통과시키지 않았다.

## 물리 계측의 제약

Native Output32에서 4초와 0.75초의 짧은 CUPTI Capture를 실행했다.
두 Capture의 Drop과 Invalid Record는 모두 0이었다.
그러나 둘 다 5%의 Profiler/A–A Variation Gate를 통과하지 못했다.

| Capture | 공통 관측 구간 | Diagnostic P Busy | Diagnostic D Busy | Diagnostic D Inactivity |
|---|---:|---:|---:|---:|
| v1 | 3.8126s | 96.30% | 53.97% | 46.03% |
| v2 | 0.520807s | 99.10% | 36.73% | 63.27% |

이 값들은 `DIAGNOSTIC` Kernel Interval Union이다.
Clean Capacity는 아니다.
v2의 273.607ms Inactive Fragment는 Capture 경계에서 잘렸다.
v2의 관측 구간 안에서 완결된 최대 Gap은 7.223ms였다.
경계 Fragment를 완전한 Idle Window의 길이 분포에 섞지 않았다.

Clean Physical Idle, Spatial Compute Slack, Request-feasible Safe Capacity는 `NOT_MEASURED`이다.
알 수 없는 Capacity를 0으로 대체하지 않았다.
NVML Utilization의 보수도 Physical Idle로 사용하지 않았다.

## Crossover를 확정하지 않은 이유

적격 Cohort DP2는 기존 Decode Cohort가 끝날 때까지 다음 Prefill을 시작하지 않는다.
따라서 그 ITL 차이를 Prefill–Decode Kernel Overlap의 직접 증거로 볼 수 없다.
Batch Composition, Cohort Queueing, Graph Shape, Host Delivery가 달라진다.
각 요인의 인과적 기여율은 `NOT_IDENTIFIED`이다.

반대로 실패한 Streaming Arm을 무시하고 Cohort만 보편적인 최강 Baseline으로 부를 수도 없다.
Native의 P Compute Saturation도 Clean Kernel 계측으로 인정하지 못했다.
이 두 제약 때문에 성능비만으로 요청한 Crossover Gate를 통과시키지 않았다.

## 선행연구와의 관계

[DuetServe](https://proceedings.mlr.press/v306/gao26m.html)는 Qwen3-8B, 2GPU, Input8000/Output200의 QPS Sweep을 다룬다.
본 실험의 vLLM/NIXL과 Cohort 제어는 그 Runtime과 같지 않다.
본 표를 DuetServe 재현 결과로 보지 않는다.
이번 분석은 이 Exact Runtime에서 관측한 Trade-off와 미해결 Baseline 제약을 분리했다.

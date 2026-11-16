# DuetServe Figure 2 및 Aggregated Decode 간섭 최종 요약

## 핵심 판정

판정은 `QPS_DEPENDENT_MEAN_TBT_DEGRADATION_REPRODUCED`다.
동일한 B8192/S1024에서 QPS만 변경한 결과다.
모든 Decode 지표가 2배 증가했다고 묶어서 표현하지 않는다.
지표별 판정은 다음과 같다.

| Metric | QPS10 / QPS2 [95% CI] | 2배 판정 |
| --- | --- | --- |
| mean_TBT_ms | 2.114 [2.092, 2.138] | PASS |
| ITL_p99_ms | 1.263 [1.252, 1.273] | NOT_OBSERVED_AT_TESTED_ENDPOINTS |
| TPOT_p95_ms | 2.141 [2.016, 2.253] | PASS |

| Metric | 최대 비율의 QPS | 최대 / QPS2 [95% CI] | 점 추정 2배 최초 QPS | CI 하한 2배 최초 QPS |
| --- | --- | --- | --- | --- |
| mean_TBT_ms | 7 | 2.116 [2.088, 2.144] | 5 | 5 |
| ITL_p99_ms | 9 | 1.263 [1.255, 1.271] | NOT_OBSERVED | NOT_ESTABLISHED |
| TPOT_p95_ms | 7 | 2.161 [2.040, 2.271] | 5 | 7 |

최초 교차와 최대값은 검사한 QPS Grid의 탐색적 요약이다.
여러 QPS 비교에 대한 동시 신뢰구간으로 해석하지 않는다.
기존 Discovery 단일 Start의 Mean TBT 비율은 1.956배였다.
그 값은2배 미만이며 기존 결과를 그대로 보존했다.
따라서2배 경계를 모든 Prompt Pool과 Arrival에서 보장한다고 주장하지 않는다.
이 결과는 Qwen3-8B와 현재 Runtime에 한정한다.
B8192/S1024는 논문 재현용 기준선이다.
B는 `max_num_batched_tokens`다.
이는 Replica별 한 Iteration에서 Prefill과 Decode를 합한 Token 상한이다.
두 GPU 전체를 합친 상한이 아니다.
S는 `max_num_seqs`이며 동시 Sequence 상한이다.
QPS나 실제 Decode Batch 크기와 같지 않다.
사용자 지시에 따라 Primary 비교를 B8192/S1024와 Native로 한정했다.
B512/S16의 추가 실행은 중단했다.
이미 완료한 작은 Budget 결과는 보충 자료로 보존했다.
이는 모든 튜닝된 Colocated 설정보다 우수하다는 검증이 아니다.
이번 측정으로 Dynamic Layer PD의 우위를 주장하지 않는다.

## 환경과 논문 대응

Model은 `Qwen/Qwen3-8B`다.
Revision은 `b968826d9c46dd6066d109eabc6255188de91218`이다.
Original BF16과 Exact Attention을 유지했다.
Greedy와 `ignore_eos=True`를 사용했다.
Thinking-disabled 생성 Suffix를 Token ID로 확인했다.
36개 Layer를 유지했다.
GPU0/1의 H100 SXM 80GB 두 대를 사용했다.
GPU당 KV Budget은 42 GiB다.
Prefix Caching은 비활성이다.
Compile Mode는0이다.
Aggregated는 FULL Mixed CUDA Graph를 사용한다.
실제 Capture 범위를 벗어난 Prefill Shape는 Eager 경로를 사용할 수 있다.
Batch-invariant 실행과 기존 Numerical Correction을 모든 적격 비교에서 유지했다.
vLLM Package Version은 `0.26.0`다.
PyTorch는 `2.11.0+cu129`다.
Torch CUDA는 `12.9`다.
NIXL은 `1.4.1`다.
GPU UUID와 실제 구성은 Evidence의 `state/environment.json`에 있다.

논문 Figure 2는 DP2다.
Figure 2는 Budget8192, Round-robin과 Input8000/Output200을 명시한다.
공통 평가 설정은 최대 Batch Size1024를 명시한다.
우리는 이를 `max_num_seqs=1024`로 대응했다.
Figure 2의 실제 CLI가 동일하다고 확인한 것은 아니다.
논문의 vLLM 0.10.1과 현재 vLLM은 다르다.
논문의 KV 메모리 비율 0.9와 현재 고정 42 GiB도 다르다.
논문 Figure 7은 Qwen3-14B TP2다.
이번 Qwen3-8B TP2는 Figure 7의 정확한 재현이 아니다.
근거는 [DuetServe 원문](https://pages.cs.wisc.edu/~markhill/papers/icml2026_DuetServe.pdf)이다.

| 항목 | 논문 | 현재 실행 |
| --- | --- | --- |
| Figure 2 Model | Qwen3-8B, Revision 미기재 | Frozen Revision b968826d |
| GPU | H100 두 대 | GPU0/1 H100 SXM 80GB 두 대 |
| vLLM | 0.10.1–v1 | 0.26.0 |
| PyTorch / CUDA | 2.8.0 / 13.0 | 2.11.0+cu129 / 12.9 |
| Driver | 580.95.05 | 570.211.01 |
| KV 설정 | Memory Utilization 0.9 | GPU당 KV 42 GiB 고정 |
| Figure 2 Budget / Router | 8192 / Round-robin | 8192 / Round-robin |
| 최대 Sequence | 공통 평가 Batch1024, Figure 2 CLI 미확인 | S1024 명시 |
| Workload | 8000/200, 공식 Demo 기반 | 실제 문서 Prefix 8000/200 |
| 생성 처리량 | Figure 2 Total Tokens/s | Output-only와 Input+Output를 분리 |

Figure 2의 Total Tokens/s가 입력을 포함하는지는 해당 설명만으로 확정하지 않는다.
우리 주 그래프는 출력 Token만 센다.
Input+Output 처리율도 원시 Metrics에 보존했다.
고정 8000/200에서는 Input+Output가 Output-only의 41배다.
이는 Token 수의 구조적 환산이며 별도의 Speedup 측정이 아니다.

## 측정 설계와 정합성

Discovery는 실제 원문 64개를 N128로 순환했다.
Fresh Confirmation은 이전 튜닝 문서와 분리한 LongBench 원문 128개를 사용했다.
도착 Seed는 2026111622, 2026111623, 2026111624다.
Primary 비교군의 축소는 일부 Confirmation 이후의 사용자 결정이다.
이미 완료한 동일한 B8192/S1024와 Native 실행도 모두 포함했다.
Seed1624는 이 범위 변경 이후에 처음 실행했다.
모든 Arm은 같은 QPS에서 동일한 Prompt와 예정 Arrival Trace를 사용했다.
각 QPS와 Method에 독립 Server Start 3개가 있다.
각 Start는 128개 요청과 요청당 200개 출력 Token을 포함한다.
Fresh Pool SHA256은 `ecdc5e1d21d0f058d376e6c4353a826909fdbcd6d6f913e26dc65b90585e8a44`다.
Reference 반복의 128개 출력이 일치했다.
독립 Full-model N32에서 출력, 모든 유효 Prompt KV와 최종 Hidden을 비교했다.
Native N32에서는 실제 D에 Import한 모든 Prompt KV도 비교했다.
N128 성능 실행의 출력은 전체 Reference와 비교했다.
모든 N128의 KV를 직접 Hash 비교했다고 확대하지 않는다.

기존 RMSNorm Phase Correction을 유지했다.
Aggregated는 표준 Continuous Scheduler를 사용했다.
수정 없는 vLLM Binary와 완전히 같다고 표현하지 않는다.
Native는 기존 NIXL 및 Cohort-isolated D Ingress를 사용했다.
이는 논문 Dynamo와 다른 Runtime Contract다.
Native D 상한16과 Aggregated 상한1024의 차이도 Isolation 비율에 영향을 줄 수 있다.
Native의 TTFT에는 Cohort Ingress 대기가 크게 포함될 수 있다.
그 Host 경로의 요청별 분해는 별도 Native 보고서에 제시했다.
따라서 Native의 순수 P Compute Saturation은 별도의 Gate다.
Clean P Kernel Busy와 현재 독립 P Capacity Reference의 공동 검증은 미확정이다.
그 물리적 조건의 판정은 `INCONCLUSIVE_MEASUREMENT`다.
D의 Clean 추가 Compute Capacity는 `NOT_MEASURED`다.
TTFT Queue 증가만으로 두 조건을 PASS로 처리하지 않았다.
추가 API28의 짧은 Kernel 측정은 원인 보고서에서 별도로 판정한다.
짧은 구간이 적격이어도 전체 실행의 Busy나 추가 Offloading Capacity를 뜻하지 않는다.

## 주요 결과

다음 값은 Start별 통계의 평균이다.
ITL p99는 세 Start의 p99 평균이다.
전체 Token을 합친 p99와 다르다.
Output Token/s는 전체 요청을 Drain한 Finite Cohort의 처리율이다.
지속 가능한 최대 처리 용량이 아니다.

| Method | Offered QPS | Start 수 | Mean TTFT (s) | Mean TBT (ms) | ITL p99 (ms) | TPOT p95 (ms) | Output Token/s |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Agg-DP2 B8192/S1024 | 2 | 3 | 0.337 | 31.474 | 240.296 | 38.176 | 369.819 |
| Agg-DP2 B8192/S1024 | 6 | 3 | 2.266 | 65.400 | 303.111 | 81.058 | 796.055 |
| Agg-DP2 B8192/S1024 | 10 | 3 | 5.579 | 66.539 | 303.413 | 81.732 | 832.044 |
| Native PD 1P1D | 2 | 3 | 3.463 | 24.784 | 26.663 | 26.078 | 351.150 |
| Native PD 1P1D | 6 | 3 | 14.089 | 26.115 | 27.656 | 26.264 | 491.144 |
| Native PD 1P1D | 10 | 3 | 17.949 | 26.148 | 27.528 | 26.259 | 492.923 |

전체 QPS, Seed별 결과와 CI는 [DP2 상세 보고서](aggregated_dp2_qps_degradation_results.md)에 있다.
고정된 QPS에서 Aggregated/Native 비율은 별도 Isolation 지표다.
이 비율을 High-QPS/Low-QPS 악화 비율과 혼동하지 않는다.

## 600초 관측

Warmup60초, 측정600초, 추가 도착90초를 분리했다.
측정 구간의 모든 도착 요청을 최종 완료까지 추적했다.
별도로 측정 시간 안에서 관측한 Token Interval도 계산했다.
두 Latency 분모를 섞지 않았다.
긴 Window의 B8192/S1024는 QPS별 독립 Start 3개다.
Native의 긴 Window는 QPS별 1개다.
B512/S16의 긴 Window는 사용자 지시에 따라 실행하지 않았다.

| 600초 Metric | QPS6 / QPS2 [95% CI] |
| --- | --- |
| mean_TBT_ms | 2.497 [2.471, 2.529] |
| active_window_observed_mean_TBT_ms | 2.486 [2.456, 2.521] |
| ITL_p99_ms | 1.256 [1.250, 1.262] |
| TPOT_p95_ms | 2.073 [2.031, 2.121] |

Queue가 증가하는 결과는 Overload Diagnostic이다.
600초를 실행했다는 이유만으로 정상상태나 Sustainable Capacity를 주장하지 않는다.
상세 값은 [Native 및 긴 구간 비교](native_pd_decode_isolation_comparison.md)에 있다.

## Output16 및 Output64의 보조 측정

Input8000과 Primary의 B8192/S1024를 유지했다.
Output별로 N32 출력, 전체 Prompt KV와 Hidden 정합성을 먼저 확인했다.
그 뒤 QPS2/6/10에서 Method별 N128을 실행했다.
Arrival Seed는 2026111615다.
기존 Discovery Prompt Pool을 사용했다.
각 점은 단일 Start이며 독립 반복 CI는 없다.
Output200의 Fresh Confirmation과 같은 증거 강도로 해석하지 않는다.

### Output16

| Method | Offered QPS | Start 수 | Mean TTFT (s) | Mean TBT (ms) | ITL p99 (ms) | TPOT p95 (ms) | Output Token/s |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Agg-DP2 B8192/S1024 | 2 | 1 | 0.311 | 29.636 | 250.695 | 43.304 | 29.818 |
| Agg-DP2 B8192/S1024 | 6 | 1 | 0.484 | 99.347 | 294.767 | 195.345 | 87.519 |
| Agg-DP2 B8192/S1024 | 10 | 1 | 3.080 | 259.310 | 298.305 | 295.989 | 107.979 |
| Native PD 1P1D | 2 | 1 | 0.556 | 22.913 | 23.745 | 22.978 | 29.670 |
| Native PD 1P1D | 6 | 1 | 8.012 | 22.922 | 24.771 | 22.990 | 55.869 |
| Native PD 1P1D | 10 | 1 | 12.334 | 22.939 | 24.762 | 22.973 | 55.958 |

### Output64

| Method | Offered QPS | Start 수 | Mean TTFT (s) | Mean TBT (ms) | ITL p99 (ms) | TPOT p95 (ms) | Output Token/s |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Agg-DP2 B8192/S1024 | 2 | 1 | 0.327 | 30.085 | 235.748 | 36.486 | 117.317 |
| Agg-DP2 B8192/S1024 | 6 | 1 | 0.503 | 83.282 | 295.671 | 123.941 | 332.930 |
| Agg-DP2 B8192/S1024 | 10 | 1 | 3.633 | 138.956 | 305.435 | 184.007 | 372.991 |
| Native PD 1P1D | 2 | 1 | 1.259 | 23.680 | 25.058 | 24.347 | 114.724 |
| Native PD 1P1D | 6 | 1 | 8.971 | 23.778 | 25.279 | 24.059 | 212.575 |
| Native PD 1P1D | 10 | 1 | 13.177 | 23.753 | 26.189 | 24.107 | 209.121 |

Output Length가 다른 점들을 하나의 Curve로 연결하지 않았다.
이 측정으로 새로운 Dynamic Offloading 효과를 주장하지 않는다.

## 원인과 한계

큰 Prefill Chunk가 들어간 Mixed Iteration에서 Decode Token 방출이 늦어졌다.
QPS가 증가하면 Mixed Interval의 비중과 Decode Batch 구성도 변했다.
Mean TBT와 ITL p99는 서로 다른 양상을 보였다.
ITL p99는 드문 긴 Interval에 민감하다.
Request TPOT는 요청 안의 199개 Interval을 평균한다.
고정 Output200에서는 Mean TBT와 Mean Request TPOT가 같은 평균을 갖는다.
두 지표의 상위 분위는 같지 않다.
QPS7–10에서는 약3.5–5.1초의 드문 Maximum ITL도 관측했다.
1초 이상인24개 Interval은 이미 Decode 중이던 요청의 Prompt 재예약과 모두 겹쳤다.
이 재계산·대기는 약300ms Mixed Iteration과 별도로 분석했다.
자세한 기록은 [간섭 원인 보고서](aggregated_decode_interference_root_cause.md)에 있다.

Stock Aggregated Scheduler는 Token Budget을 사용한다.
별도의 Decode-safe 시간 Budget이나 Deadline을 집행하지 않는다.
따라서 이번 현상을 시간 예측기의 Budget 초과로 설명하지 않는다.
GPU Kernel은 일반 Scheduler가 즉시 선점할 수 없다고 취급했다.
SM, HBM, Cache, Launch의 배타적 원인 기여도는 `NOT_IDENTIFIED`다.
Clean GPU Kernel Busy와 Idle은 Profiler Gate 결과를 따라 판정했다.
NVML 사용률의 보수값을 Physical Idle로 사용하지 않았다.

TP2는 TP1 Reference 출력과 불일치했다.
Within-TP2 Diagnostic을 주 성능 비교와 분리했다.
자세한 한계는 [TP2 보고서](aggregated_tp2_qps_degradation_results.md)에 있다.
Static/Dynamic Layer 개선은 이번 Characterization에서 `NOT_RUN`이다.
기존 결과를 새 Offloading 성과로 재사용하지 않았다.
적격 TP2 기준선과 Clean D 추가 실행 예산을 확보하지 못했다.
현재 단계의 결과를 새로운 Layer Offloading 순이익으로 확대하지 않았다.
후속 Layer Admission과 Decode 보호는 별도 검증이 필요하다.

## 실패 실행과 복구

초기 DP2 Qualification은 CPU Fixture 누락으로 시작 전에 실패했다.
초기 Native QPS2는 Canonical KV Contract 누락으로 Worker 준비에 실패했다.
두 실패는 GPU 성능 결과로 집계하지 않았다.
의존 파일을 확인한 뒤 새 이름의 실행으로 검증했다.
최초 Nsight 실행은 감시기가 Profiler의 새 Process Session을 외부 작업으로 분류했다.
기존 실패 Receipt를 PASS로 고치지 않았다.
Live Process Ancestry를 확인하는 별도 감시기로 새 실행을 수행했다.
그 실행도 Profiler 성능 간섭 Gate는 통과하지 못했다.
별도 Normalization Calibration에서는 Lease 해제와 CUDA 자식 종료 사이의 경합이 발생했다.
Resource Guard가 다음 Model 실행을 시작 전에 거절했다.
확인되지 않은 GPU Process를 종료하지 않았다.
이후 CPU 부모가 CUDA 자식의 실제 종료와 NVML 해제를 확인할 때까지 Lease를 보유했다.
원시 실패 로그와 재시도 이름을 모두 보존했다.
실패는 `FAILED`이며 알고리즘적 성능 악화와 구분한다.
범위 변경 당시의 B512/S16 실행은 사용자 지시에 따른 취소다.
이 부분 실행의 Latency를 최종 표에 포함하지 않았다.

## 산출물과 보존

사용자가 요청한 4개 Discovery 그림은 `poc_flashvep/` 바로 아래에 저장했다.
[4개 Discovery 그림 합본](../duetserve_qps_four_metrics_discovery_o200.png)을 제공한다.
이는 단일 Start의 예비 그림이다.
최종 반복 그림은 Evidence의 `figures/`에서 구분한다.
[독립 반복 4개 그림 합본](../duetserve_qps_four_metrics_confirmed_o200.png)도 `poc_flashvep/` 바로 아래에 저장했다.
Source는 `poc_flashvep/duetserve_agg_decode/`다.
Evidence는 `duetserve_agg_dp2_tp2_tbt_crossover_20261116/`다.
Publication Parent는 `db300024e29e6adec845bec96f90d6efcdbd0f72`다.
Branch는 `flashvep/duetserve_agg_dp2_tp2_tbt_crossover_20261116`다.
사용자 Checkout, Index와 외부 Runtime 보존을 확인한다.
Commit 및 Remote SHA는 최종 Publication Receipt로 확인한다.
실패한 실행과 원시 기록을 삭제하지 않는다.

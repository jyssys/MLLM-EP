# Aggregated Decode 간섭 원인 분석

## 가장 강한 관측 근거

Budget과 QPS를 별도로 통제했다.
실제 Model Input의 Prefill/Decode 구성을 기록했다.
Output Frontier를 개별 Token Count와 연결했다.
Host Enclosure, CUDA Event Enclosure와 Kernel Interval을 구분했다.
별도 Decode-safe 시간 Budget이나 Deadline은 Stock Scheduler에 없다.
Token Budget을 30ms 같은 시간 Budget으로 해석하지 않았다.
기존 vLLM은 Prefill Row와 Decode Row를 같은 Model Forward에 넣을 수 있다.
별도의 Prefill Stream을 새로 추가해 동시 실행을 강제하지 않았다.
따라서 두 독립 Stream의 자원 경쟁으로 전체 악화를 설명하지 않는다.

## QPS2와 QPS6의 Interval 구성

같은 Host-only 계측과 Arrival Seed2026111616을 사용했다.
Prefill을 포함한 Interval과 Prefill이 없는 Interval을 실제 Model Input으로 분류했다.
Prefill 포함 분류에는 KV 압력으로 생성 Prefix를 재계산한 경우도 포함한다.
이를 모두 새로운 Prefill/Decode Mixed Batch라고 해석하지 않는다.
다음 항등식은 측정 Mean TBT 변화의 설명이다.

| 측정 평균 변화의 항등식 | 변화 (ms) | 평균 변화 중 비율 (%) |
| --- | --- | --- |
| mixed_frequency_ms | 23.696 | 67.492 |
| mixed_class_latency_ms | 2.929 | 8.342 |
| decode_only_class_latency_ms | 8.485 | 24.166 |

총 Mean TBT 변화는 35.110ms다.
이 분해는 `TRACE_ANALYSIS`다.
이는 배타적인 GPU 원인 기여도가 아니다.
Decode-only로 분류한 Interval도 앞선 Mixed Iteration 뒤에서 기다릴 수 있다.
작업 구성과 실행 지연은 서로 영향을 준다.

## 원인별 판정

1. 큰 Prefill Chunk와 긴 Mixed Iteration은 실제 로그에서 확인했다.
2. 높은 QPS의 Decode Batch와 Mixed Interval 비중 변화도 확인했다.
3. Host 수신 지연은 Worker 방출 간격과 별도로 계산했다.
4. Graph/Eager Coverage는 실제 Replay Counter와 Shape로 구분했다.
5. KV 압력에 따른 중복 Scheduled Prompt Rows는 Host Accounting으로 구분했다.
6. SM, HBM Bandwidth, Cache와 Launch의 배타적 기여도는 `NOT_IDENTIFIED`다.

RMSNorm Correction의 별도 Primitive 비용도 측정했다.
CUDA Event Enclosure이며 Kernel Union이나 Serving Speedup이 아니다.
Mask 준비와 H2D 비용을 제외한 범위를 명시했다.
그 측정만으로 전체 Serving 비용을 빼거나 더하지 않았다.

## 수 초짜리 Maximum ITL의 별도 원인

QPS7–10의 Fresh Confirmation 12개 Start를 추가로 확인했다.
각 Start에는 25,472개 ITL이 있다.
1초 이상인 Interval은 Start당 2개이며 전체 24개다.
24개 모두 같은 요청의 Prompt 재예약과 시간상 겹쳤다.
그 요청들은 이미 Decode Token을 출력한 상태였다.
일반적인 약300ms Mixed Iteration과 이 드문 대기를 구분해야 한다.

QPS10, Seed2026111622의 Req74가 한 예다.
22번째 Token과23번째 Token 사이의 Client ITL은4882.645ms였다.
Worker 방출 간격은4882.024ms였다.
두 Delivery Lag의 차이는0.620ms였다.
따라서 이 예의 수 초 지연을 Coordinator 전달 지연으로 설명하지 않는다.
가까운 Queue Sample의 KV Usage는100%이며 Free Block은0개였다.
같은 요청은 Prompt8000 Row를 Start0부터 다시 예약했다.
이 재예약 시각은 Output 공백 시작4558.639ms 이후였다.
그 시각부터 다음 Worker 출력까지는323.385ms였다.
두 구간은 Host Timestamp 차이며 배타적인 GPU 계산시간이 아니다.

이 기록은 KV 압력에 따른 Preemption 및 재계산 경로와 일치한다.
설치된 Scheduler의 `_preempt_request`는 Block을 해제하고 Computed Token 수를0으로 되돌린다.
인접 Queue Sample에서도 `PREEMPTED` 상태를 확인했다.
다만 정확한 Request별 Preemption 전이 시각은 `NOT_MEASURED`다.
GPU 재계산의 배타적 실행시간도 `NOT_MEASURED`다.
따라서 약4.88초를 하나의 Prefill Kernel 실행시간으로 해석하지 않는다.
이 드문 Tail과 Mean TBT 증가의 구성 변화는 다른 결과다.
전체 기록은 `analysis/fresh_primary_rare_ITL_prompt_reschedule.json`에 있다.
이 절의 증거 수준은 `TRACE_ANALYSIS`다.

## KV 압력과 중복 Scheduled Work

원래 Prompt의 고유 Token 구간을 Request별로 합집합 처리했다.
중복 구간을 다시 예약한 양은 별도로 계산했다.
분모는 각 실행의 Request 수 ×8000이다.

| Run | 고유 Prompt Rows 분모 | Scheduled Prompt Rows | 중복 Scheduled Rows | 분모 대비 중복 비율 (%) |
| --- | --- | --- | --- | --- |
| confirm_DP2_B8192_S1024_FULL_N128_O200_q10_s2026111622_v1 | 1024000 | 1040000 | 16000 | 1.562 |
| confirm_DP2_B8192_S1024_FULL_N128_O200_q10_s2026111623_v1 | 1024000 | 1040000 | 16000 | 1.562 |
| confirm_DP2_B8192_S1024_FULL_N128_O200_q10_s2026111624_v1 | 1024000 | 1040000 | 16000 | 1.562 |
| confirm_DP2_B8192_S1024_FULL_N128_O200_q2_s2026111622_v1 | 1024000 | 1024000 | 0 | 0.000 |
| confirm_DP2_B8192_S1024_FULL_N128_O200_q2_s2026111623_v1 | 1024000 | 1024000 | 0 | 0.000 |
| confirm_DP2_B8192_S1024_FULL_N128_O200_q2_s2026111624_v1 | 1024000 | 1024000 | 0 | 0.000 |
| confirm_DP2_B8192_S1024_FULL_N128_O200_q6_s2026111622_v1 | 1024000 | 1024000 | 0 | 0.000 |
| confirm_DP2_B8192_S1024_FULL_N128_O200_q6_s2026111623_v1 | 1024000 | 1024000 | 0 | 0.000 |
| confirm_DP2_B8192_S1024_FULL_N128_O200_q6_s2026111624_v1 | 1024000 | 1024000 | 0 | 0.000 |
| confirm_Native_N128_O200_q10_s2026111622_v1 | 1024000 | 1024000 | 0 | 0.000 |
| confirm_Native_N128_O200_q10_s2026111623_v1 | 1024000 | 1024000 | 0 | 0.000 |
| confirm_Native_N128_O200_q10_s2026111624_v1 | 1024000 | 1024000 | 0 | 0.000 |
| confirm_Native_N128_O200_q2_s2026111622_v1 | 1024000 | 1024000 | 0 | 0.000 |
| confirm_Native_N128_O200_q2_s2026111623_v1 | 1024000 | 1024000 | 0 | 0.000 |
| confirm_Native_N128_O200_q2_s2026111624_v1 | 1024000 | 1024000 | 0 | 0.000 |
| confirm_Native_N128_O200_q6_s2026111622_v1 | 1024000 | 1024000 | 0 | 0.000 |
| confirm_Native_N128_O200_q6_s2026111623_v1 | 1024000 | 1024000 | 0 | 0.000 |
| confirm_Native_N128_O200_q6_s2026111624_v1 | 1024000 | 1024000 | 0 | 0.000 |
| long_DP2_B8192_S1024_O200_q2.0_s2026111631_W600_v1 | 12048000 | 12048000 | 0 | 0.000 |
| long_DP2_B8192_S1024_O200_q2.0_s2026111632_W600_v1 | 12992000 | 12992000 | 0 | 0.000 |
| long_DP2_B8192_S1024_O200_q2.0_s2026111633_W600_v1 | 12360000 | 12360000 | 0 | 0.000 |
| long_DP2_B8192_S1024_O200_q6.0_s2026111631_W600_v1 | 36288000 | 37312293 | 1024293 | 2.823 |
| long_DP2_B8192_S1024_O200_q6.0_s2026111632_W600_v1 | 37184000 | 38184144 | 1000144 | 2.690 |
| long_DP2_B8192_S1024_O200_q6.0_s2026111633_W600_v1 | 36272000 | 36272000 | 0 | 0.000 |
| long_Native_O200_q2.0_s2026111631_W600_v1 | 12048000 | 12048000 | 0 | 0.000 |
| long_Native_O200_q6.0_s2026111631_W600_v1 | 36288000 | 36288000 | 0 | 0.000 |

이 값은 `TRACE_ANALYSIS`다.
Host Scheduler의 예약 Row 수이며 실제 GPU FLOPs가 아니다.
생성된 Prefix의 추가 재계산 비용 전체를 이 표가 포함하는 것은 아니다.
중복 예약 비율만으로 ITL 증가의 배타적 기여도를 계산하지 않았다.

## Finite Cohort와 장기 관측의 차이

N128은 마지막 새 Prompt가 끝난 뒤의 Decode Tail도 평균에 포함한다.
장기 관측은 새 Prompt가 계속 유입되는 구간의 비중이 다르다.
아래 경계는 마지막 First-token 방출의 Host Timestamp다.
GPU Kernel이나 재계산 종료를 뜻하지 않는다.

| QPS6 실행 | 새 Prompt 종료 후 Interval 수 | 전체 Interval 대비 (%) | 이 구간의 Mean TBT (ms) |
| --- | --- | --- | --- |
| confirm_DP2_B8192_S1024_FULL_N128_O200_q6_s2026111622_v1 | 6805 | 26.716 | 28.541 |
| confirm_DP2_B8192_S1024_FULL_N128_O200_q6_s2026111623_v1 | 9335 | 36.648 | 29.974 |
| confirm_DP2_B8192_S1024_FULL_N128_O200_q6_s2026111624_v1 | 8601 | 33.766 | 29.709 |
| long_DP2_B8192_S1024_O200_q6.0_s2026111631_W600_v1 | 2295 | 0.254 | 25.506 |
| long_DP2_B8192_S1024_O200_q6.0_s2026111632_W600_v1 | 6282 | 0.679 | 28.476 |
| long_DP2_B8192_S1024_O200_q6.0_s2026111633_W600_v1 | 4859 | 0.539 | 27.133 |

이는 `TRACE_ANALYSIS`다.
새 Prompt가 없는 구간에도 KV 작업과 Prefix 재계산이 있을 수 있다.
Batch 구성, KV 압력과 Prefill 빈도도 함께 달라질 수 있다.
따라서 두 실행의 Mean TBT 차이 전체를 Tail 비중만으로 설명하지 않는다.

## Profiler 간섭 검사

| Run family | 계측 | 5% Gate | Mean TBT 변화 (%) | TTFT p95 변화 (%) | TPOT p95 변화 (%) |
| --- | --- | --- | --- | --- | --- |
| profile_DP2_B512_S16_FULL_N128_O200_q6_s2026111616 | HOST_ONLY | True | -0.021 | 0.166 | -0.067 |
| profile_DP2_B512_S16_FULL_N128_O200_q6_s2026111616 | HOST_EVENT | True | 0.136 | 0.507 | -0.055 |
| profile_DP2_B512_S16_FULL_N128_O200_q6_s2026111616 | NSYS | False | 14.911 | 151.197 | 139.995 |
| profile_DP2_B8192_S1024_FULL_N128_O200_q2_s2026111616 | HOST_ONLY | False | 0.077 | -1.089 | -0.311 |
| profile_DP2_B8192_S1024_FULL_N128_O200_q2_s2026111616 | HOST_EVENT | False | NOT_RUN_OR_FAILED | NOT_RUN_OR_FAILED | NOT_RUN_OR_FAILED |
| profile_DP2_B8192_S1024_FULL_N128_O200_q2_s2026111616 | NSYS | False | NOT_RUN_OR_FAILED | NOT_RUN_OR_FAILED | NOT_RUN_OR_FAILED |
| profile_DP2_B8192_S1024_FULL_N128_O200_q6_s2026111616 | HOST_ONLY | False | 0.066 | 0.543 | 0.119 |
| profile_DP2_B8192_S1024_FULL_N128_O200_q6_s2026111616 | HOST_EVENT | False | 0.012 | 0.571 | 0.079 |
| profile_DP2_B8192_S1024_FULL_N128_O200_q6_s2026111616 | NSYS | False | 13.362 | 165.199 | 45.177 |
| profile_Native_N128_O200_q6_s2026111616 | HOST_ONLY | False | NOT_RUN_OR_FAILED | NOT_RUN_OR_FAILED | NOT_RUN_OR_FAILED |
| profile_Native_N128_O200_q6_s2026111616 | HOST_EVENT | False | 0.084 | -0.006 | 0.045 |
| profile_Native_N128_O200_q6_s2026111616 | NSYS | False | -0.571 | 21.849 | 0.044 |

Whole-run과 같은 Request/Token의 짧은 ROI를 모두 검사했다.
A/A 자체 변동도 5% 이하를 요구했다.
짧은 ROI의 Tail 변동으로 신뢰성 Gate가 실패한 경우도 구분했다.
계측을 통과시키기 위해 Threshold를 변경하지 않았다.

| Run | Role/GPU | 구간 (s) | 기록 Kernel Union Busy (%) | Clean 판정 |
| --- | --- | --- | --- | --- |
| profile_DP2_B512_S16_FULL_N128_O200_q6_s2026111616_NSYS_v1 | replica0/GPU0 | 2.008 | 98.565 | NOT_MEASURED |
| profile_DP2_B8192_S1024_FULL_N128_O200_q6_s2026111616_NSYS_v1 | replica0/GPU0 | 2.186 | 97.588 | NOT_MEASURED |
| profile_DP2_B8192_S1024_FULL_N128_O200_q6_s2026111616_NSYS_v2 | replica0/GPU0 | 2.164 | 98.682 | NOT_MEASURED |
| profile_Native_N128_O200_q6_s2026111616_NSYS_v1 | prefill/GPU0 | 2.361 | 94.180 | NOT_MEASURED |

위 Kernel Union은 실제 Nsight 기록의 Stream 합집합이다.
겹치는 Kernel Duration을 단순 합산하지 않았다.
하지만 Clean Gate에 실패한 값은 계측된 Diagnostic일 뿐이다.
그 값을 Clean Native GPU Busy나 사용 가능한 Idle Capacity로 주장하지 않는다.
Memcpy와 NCCL Kernel은 Compute Kernel과 분리했다.
P-only Capture는 D가 실행한 NIXL Transfer의 Wire Time을 측정하지 않는다.
기록이 없는 통신 비용을 0으로 해석하지 않는다.
CUDA Graph Node는 실제 Kernel로 계산하고 Replay API를 다시 더하지 않았다.
GPU0의 실제 Timeline과 다른 실행이나 GPU의 시간을 합치지 않았다.

## CUDA12.9 CUPTI API28의 별도 검증

기존 고정 Host Buffer Collector를 CUDA12.9 CUPTI에 연결했다.
초기 두 Build는 CUDA Header의 버전 및 누락된 CRT Header 때문에 실패했다.
이 실패에서 Serving GPU 실행은 시작하지 않았다.
일치하는 CUDA Driver Header와 명시한 CRT Header 경로로 다시 Build했다.
단위 검증은 Eager100개와 Graph100개의 Kernel을 기록했다.
그 검증의 Dropped와 Invalid 기록은 모두0이었다.
이는 `CORRECTNESS_ONLY`이며 Serving 간섭 검증과 다르다.
이후 Native에서 제한적인 A/계측/A를 시도했다.
Background Activation 실행은 Warmup Timeout으로 실패했다.
단위 검증처럼 CUDA Context 생성 전에 활성화하는 별도 실행도 보존했다.
후자의 Capture는 전체 기록이며 분석 구간만 짧게 제한했다.
예정 분석 구간은10초 이후20초다.
양 끝의0.1초는 Union 계산에서 제외했다.
Serving Thread에는 Collector용 CUDA Fence를 추가하지 않았다.
기존 A/A 및5% 간섭 기준을 그대로 적용했다.
Native 계측 문제 이후 Aggregated의 추가 API28 실행은 하지 않았다.

| API28 실행 | Role | Clean Gate | 측정 초 | 기록 Busy (%) | 기록 Idle (%) | Idle 구간 수 | Idle p95 (ms) | Idle Max (ms) | Invalid 기록 수 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Native | prefill | NOT_MEASURED_CLEAN_PHYSICAL_CAPACITY | 19.800 | 97.814 | 2.186 | 9203 | 0.003 | 73.974 | 0 |
| Native | decode | NOT_MEASURED_CLEAN_PHYSICAL_CAPACITY | 19.800 | 93.188 | 6.812 | 183196 | 0.001 | 343.708 | 0 |

Native A/A의 Output Token/s 차이는 8.981%다.
계측 실행의 처리량 변화는 두 Control 평균 대비 3.350%다.
Control 안정성과 계측 실행의 오차를 별도로 판정했다.

위 표의 Busy와 Idle은 실제 기록의 수치다.
Clean Gate가 실패한 행은 Diagnostic이며 Clean Idle 수치가 아니다.
적격 행도 해당 짧은 관측 구간에만 유효하다.
Kernel-only Capture는 Memcpy나 NIXL Wire 비용을 측정하지 않는다.
Kernel Inactivity와 Decode-safe Layer 실행 예산은 다르다.
원시 Interval, Clock Anchor, Drop Counter와 간섭 오차는 `analysis/api28_*_serving_gate.json`에 있다.

## 종합 해석

큰 Token Budget은 긴 Model Iteration을 허용한다.
기존 Decode가 그 실행에 결합되면 Token 방출도 함께 늦어질 수 있다.
높은 동시성은 처리량과 Decode 지연의 절충점을 바꾼다.
이를 구조적인 D Compute 부족으로 단정하지 않는다.
범위 변경 전에 측정한 작은-Budget 결과도 보충 자료로 확인했다.
Clean GPU Idle과 안전한 추가 Layer Capacity는 여전히 별도 검증 대상이다.

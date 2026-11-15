# Dense AR Layer-wise PD: SLO-constrained Throughput 최종 검증

## 최종 판정

**`NO_GO_ON_TESTED_GRID`**다.
동결된 Dynamic은 Strong Colocated보다 높은 SLO-qualified Throughput을 제공하지 못했다.
같은 Decode-priority SLO의 최대 확인점은 Colocated QPS5.0과 Dynamic QPS4.4다.
평균 유효 처리량은 각각 **80.071111과 71.182222 Output tokens/s**였다.
Dynamic/Colocated 비율은 **0.888988**, 독립 Start Bootstrap 95% CI는 **[0.872953, 0.906359]**다.
Dynamic은 이 비교에서 11.10% 낮았다.

이는 가장 높은 적격 Tested Point의 비교다.
정확한 전역 최대나 모든 Workload의 구조적 불가능성은 아니다.
Dynamic QPS4.8은 3회 모두 Queue·Joint SLO 기준을 실패했다.
그 점의 Raw Throughput을 지속 가능한 Capacity로 집계하지 않았다.

P→D Partial Layer 계산과 State 정합성은 실제로 재현했다.
그러나 Strong Static 대비 독립적인 Dynamic 기여도 확정하지 못했다.
Native-relative p99/TPOT 기준 통과와 Maximum ITL 보호도 구분해야 한다.
일부 Dynamic 실행에는 큰 Host Delivery Tail이 남았다.

## 1. 연구 질문과 공정한 비교

Primary Question은 같은 2GPU와 SLO에서 Strong Continuous Colocated보다 높은 지속 가능한 유효 Output 처리량을 제공하는지다.
Native보다 TTFT를 낮추는 것만으로 성공을 선언하지 않는다.
Dynamic의 독립적 기여는 Strong Static과 실제 이전량 및 Batch Geometry도 비교해야 한다.

기존의 Dynamic/Colocated Goodput 1.735×는 이번 결론에 사용하지 않는다.
그 당시 Colocated는 Balanced 목적의 설정이었다.
이번에는 Balanced, Decode-priority와 Raw 목적을 각각 탐색했다.
Dynamic 결과를 보고 SLO Threshold를 변경하지 않았다.

## 2. 고정 환경과 Dataset

| 항목 | 값 |
|---|---|
| Model | Qwen/Qwen3-8B |
| Model Revision | `b968826d9c46dd6066d109eabc6255188de91218` |
| 계산 | Original BF16, Exact Attention, 36 Layers, Full36 Decode |
| Sampling | Greedy, Thinking Disabled, 고정 Output의 `ignore_eos=True` |
| GPU | H100 80GB × 2, GPU0/1, NV18 연결 |
| GPU0 UUID | `GPU-f217c8a0-1142-20f4-d84b-af29f3a47a0d` |
| GPU1 UUID | `GPU-a77f3471-67d4-20b0-9fab-e502d4de5adb` |
| Runtime | vLLM 0.26.0+cu129, PyTorch 2.11.0+cu129, NIXL 1.4.1 |
| KV | GPU당 42 GiB, Prefix Caching Disabled |
| Primary Workload | Input8000/Output16, Open-loop Poisson |
| Source Base | `92e5275f1175bc08680de77180b8f5c652f2d08f` |
| 새 Source | `poc_flashvep/dense_ar_slo_layer_pd/` |
| Evidence | `dense_ar_slo_throughput_dynamic_layer_pd_20261114/` |

새 Confirmation Pool은 LongBench의 실제 원문 128개다.
Dataset Revision은 `5e628be450b7e67fb7ae6e201bd6d8f7056f7672`다.
이전 Prompt ID와 7900-token Prefix의 중복은 각각 0개다.
Prompt Token, 원문 ID와 Truncation 규칙은 [Manifest](dense_ar_slo_throughput_dynamic_layer_pd_20261114/data/unseen_manifest.json)에 보존한다.
Confirmation Pool SHA256은 `c9613811b1093a2abd8a021939a8fab0e9794b0bf817fd2d26260962a777b8d7`다.

장기 실행은 해당 Pool을 순환한다.
순환한 요청을 새로운 독립 원문으로 집계하지 않는다.
새 Calibration 32개는 Confirmation 128개와 분리했다.
독립성의 통계 단위는 Request나 Token이 아니라 Server Start다.

## 3. SLO와 처리량 정의

Input8000/Output16의 Frozen SLO는 다음과 같다.

| Regime | TTFT | 개별 ITL | Request TPOT |
|---|---:|---:|---:|
| TTFT-priority | 2000 ms | 54 ms | 52 ms |
| Balanced | 5000 ms | 34 ms | 33 ms |
| Decode-priority | 10000 ms | 30 ms | 29 ms |

Joint Pass는 TTFT, TPOT와 모든 개별 ITL의 통과를 요구한다.
첫 ITL도 포함한다.
ITL p99 통과만으로 Request-level Joint Pass를 판정하지 않는다.
Joint Attainment 95%는 사전에 정한 연구용 목표다.
99%는 별도의 민감도 분석이다.

SLO SHA256은 `c8aa94213e4f4ce54821d01ba985a8e9e88a20566a3e8d1f3cc8060d1b3926d9`다.
다른 Output Cap의 고정값은 [Frozen SLO 파일](dense_ar_slo_throughput_dynamic_layer_pd_20261114/configs/frozen_SLO_regimes.json)에 있다.

Raw Throughput은 Completion Window에서 완료된 요청의 모든 출력 Token을 고정 시간으로 나눈다.
SLO-qualified Throughput은 그중 Joint Pass 요청의 출력 Token만 사용한다.
입력 Token은 포함하지 않는다.
구간 안의 실제 Token Event Rate와 Finite-cohort Drain Rate는 별도로 보고한다.

Warm-up 60초, Measurement 600초와 후속 Arrival 90초를 구분한다.
측정 중 도착한 모든 요청을 이후 완료까지 추적한다.
늦은 요청을 Latency나 SLO 분모에서 제외하지 않는다.
Queue 안정성, 실제 Arrival/Completion Rate와 Lifetime을 함께 검사한다.
상세 Gate는 [지속 가능 처리량 프로토콜](dense_ar_sustainable_generation_throughput.md)에 있다.

가장 높은 적격 Tested Point와 실패한 이웃 QPS를 보고한다.
희소한 Grid로 정확한 전역 최대를 주장하지 않는다.
600초 안정성 검사를 무한 시간의 안정성 증명으로 표현하지 않는다.

## 4. Strong Colocated의 복구와 목적별 Frontier

표준 vLLM Continuous Batching, Async Scheduling과 Full Mixed CUDA Graph를 사용한다.
Colocated를 Custom Cohort Scheduler로 약화하지 않는다.
GPU0/1 각각에 전체 Model Replica를 배치한다.
Chunk Budget, Sequence Limit, Routing과 합법적인 Graph 옵션을 Discovery에서 비교했다.

GPU Phase Mask는 원래 Prefill/Decode RMSNorm Reduction을 유지한다.
작은 GPU 비교 396개는 Byte Equality를 통과했다.
N32 전체 Prompt KV, Final Hidden과 Output도 통과했다.
이 시험의 실제 Mixed Graph Replay는 472회였다.
기본 Compile3의 Output 13/32 불일치는 적격 성능에서 제외했다.
Opaque Original Norm의 Compile3는 별도 정합성을 통과했다.
Compiler 내부의 정확한 수치 변화 원인은 `NOT_IDENTIFIED`다.

아래는 600초 단일 Discovery다.
독립 확인의 평균이나 확정 최대가 아니다.

| 목적 및 설정 | Offered / Actual QPS | Raw tokens/s | 목적별 Qualified tokens/s | 판정 |
|---|---:|---:|---:|---|
| C-Balanced, FULL/B512/S16 | 5.0 / 4.946667 | 79.333333 | 79.066667 | PASS |
| C-Decode, 같은 설정 | 5.0 / 4.946667 | 79.333333 | 79.040000 | PASS |
| C-Decode, FULL/B512/S16, RR | 5.2 / 5.366667 | 84.960000 | 83.973333 | Queue 기준 FAIL |
| C-Raw, B8192/S32, Decode-only Graph | 6.4 / 6.590000 | 105.920000 | Decode 기준 0.826667 | Raw 안정성 PASS, Decode SLO FAIL |

C-Raw의 ITL p99는 299.659880 ms였다.
TPOT p95는 296.611123 ms였다.
이를 Decode-safe Capacity로 표현하지 않는다.
Token Budget, Sequence Limit과 Graph Mode가 함께 다르다.
따라서 간섭을 Chunk Size 하나의 인과 효과로 분해하지 않는다.

QPS5.2의 Least-outstanding Routing도 안정성 Gate를 통과하지 못했다.
Routing만 변경해 Capacity 한계를 해결하지 못했다.
자세한 설정과 실패 근거는 [Colocated Frontier](dense_ar_slo_specific_colocated_frontier.md)에 있다.

## 5. 실제로 수정한 병목

### 5.1 Generic KV Layout과 Handoff

선택적 Canonical Block 옵션은 새로 할당한 Block의 반환 순서만 정렬한다.
원래 선택된 Block 집합, Reference Count와 기존 요청의 Block Table은 유지한다.
Source/Destination의 반대 Physical Stride가 NIXL Coalescing을 막는 문제를 줄인다.
이것은 Generic PD 최적화다.
Dynamic만의 기여로 집계하지 않는다.

동일한 Async Batch8 A/B/A의 Handoff p95는 다음과 같다.

| 실행 | 평균 Descriptor 수 | Handoff p95 |
|---|---:|---:|
| 기존 A1 | 5572.466667 | 490.177947 ms |
| 정렬 B | 251.566667 | 166.044161 ms |
| 기존 A2 | 7647.850000 | 615.882327 ms |

B의 Handoff p95는 두 A의 평균 대비 0.300244배였다.
한 A/B/A 기전 검증이다.
독립 Serving Speedup이나 Pure Wire Time의 신뢰구간이 아니다.
근거는 [Transport A/B/A](dense_ar_slo_throughput_dynamic_layer_pd_20261114/analysis/canonical_transport_ABA.json)다.

Colocated에도 같은 옵션을 시험했다.
전체 State 검증과 A/B/A 이후 기본 Layout을 선택했다.
Colocated에 불리한 옵션을 강제로 적용하지 않았다.
기본 UCX Probe는 Zero-copy `cuda_ipc/cuda`를 기록했다.
이 Probe로 실측 Bandwidth를 주장하지 않는다.

### 5.2 직접 Deadline Predicate와 Completion Model

Last2/4/8/12를 각각 평가한다.
실제로 영향을 받는 Work의 Deadline을 Candidate Reject Predicate에 사용한다.
Suffix 시작 전에 종료되는 Cohort의 과거 ITL Deadline은 제외한다.
D의 실제 실행은 기존 Decode Cohort가 끝날 때까지 기다린다.
GPU Kernel 선점을 가정하지 않는다.

Source Frontier, Handoff, Destination Cohort와 정상 Decode의 비용을 구분한다.
목적함수는 SLO 제약하의 경험적 Pipeline-rate Surrogate다.
정확한 미래 Goodput이나 미래 Arrival을 알고 있다고 가정하지 않는다.

지난 TTFT Deadline을 계속 거절하던 Strict 정책은 Native Fallback을 누적시켰다.
별도 `QUEUE_RECOVERY`는 이미 지난 TTFT Deadline만 구분한다.
해당 요청은 기존 SLO에서 계속 실패로 집계한다.
ITL과 미래 TTFT Deadline은 그대로 검사한다.
요청을 버리거나 Prefix를 재실행하지 않는다.

새 Canonical Runtime의 Batch-aware Decode Table과 Completion Residual을 다시 Calibration했다.
N128 Calibration Start 3회의 최대 양의 명목 Group Residual은 125.296004 ms였다.
사전 규칙에 따른 Margin은 150 ms다.
작은 종속 표본의 상한을 WCET로 주장하지 않는다.
Runtime과 모델을 함께 바꿨으므로 이전 오차와의 차이를 Layout 하나의 효과로 표현하지 않는다.
상세 근거는 [Admission 계약](dense_ar_deadline_aware_small_cut_admission.md)에 있다.

### 5.3 여전히 남은 Control-plane 비용

QPS5.0 Static Discovery의 Credit RPC 633건에서 p95는 152.313 ms였다.
측정 구간의 RPC Enclosure Union은 17.586638초였다.
600초의 2.931106%다.
이를 모두 제거하는 낙관적 Component Model은 79.12를 81.509119 tokens/s로 바꾼다.
이는 모델 추정이다.
실제 Speedup, Kernel Idle 또는 구현된 개선이 아니다.
Main-loop Credit Dispatch의 별도 Fast Path는 구현하지 않았다.

## 6. 새 Runtime의 장기 Discovery

모두 같은 Seed와 600초 측정 구간을 사용했다.
Output/Lifetime은 통과했다.

| Method / Geometry | Offered QPS | Raw tokens/s | TTFT p95 | ITL p99 | TPOT p95 | Decode Joint | 적격 |
|---|---:|---:|---:|---:|---:|---:|---|
| Native | 3.4 | 53.786667 | 11.329237 s | 24.507439 ms | 22.922553 ms | 87.872861% | FAIL |
| Static Last12 / D8 Whole | 5.0 | 79.120000 | 11.452546 s | 25.773306 ms | 24.164237 ms | 83.423363% | FAIL |
| Dynamic / D8 Whole | 5.0 | 77.893333 | 20.203261 s | 25.537322 ms | 24.156384 ms | 48.075625% | FAIL |
| Static Last12 / D4 Stream | 4.8 | 76.186667 | 10.993270 s | 25.002150 ms | 23.770026 ms | 88.978873% | FAIL |
| Dynamic / D4 Stream | 4.8 | 76.133333 | 6.096550 s | 24.492574 ms | 23.734734 ms | 99.683099% | PASS |

Static D8의 Queue는 중간에 증가한 뒤 감소했다.
마지막 Outstanding은 시작의 23개보다 적은 18개였다.
따라서 모든 실패를 지속적인 무한 Queue 증가라고 설명하지 않는다.
이 실행은 사전 Queue Trend 기준과 TTFT/Joint SLO를 실패했다.

D4 Dynamic의 좋은 점 하나만 선택해 최종 GO로 표현하지 않는다.
특히 Strong Colocated의 최대 유효 처리량을 넘었는지는 별도 문제다.

Static Fraction 100%/93.75%/87.5%도 같은 N128 Trace로 검사했다.
TTFT p95는 각각 3191.095277/3104.248094/4420.484281 ms였다.
93.75%의 개선은 2.72%였다.
사전 5% Screening 기준보다 작아서 추가 장기 실행은 하지 않았다.
이는 Static 탐색의 한계다.
Static을 전역 최적이라고 주장하지 않는다.

## 7. 독립 Confirmation

Source와 설정은 Held-out 성능 평가 전에 동결했다.
동결 파일은 [heldout_runtime_freeze.json](dense_ar_slo_throughput_dynamic_layer_pd_20261114/state/heldout_runtime_freeze.json)이다.
Arrival Seed는 2026111511/2026111512/2026111513이다.
방법 순서는 Seed별로 회전하거나 뒤집는다.

선택된 Candidate는 Native3.2, Colocated5.0, Static4.4와 Dynamic4.8이다.
Static4.4는 낮은 QPS의 사전 Candidate다.
Confirmation 전에 통과한 Capacity로 표현하지 않는다.
Dynamic-relative Native는 같은 QPS4.8과 같은 Seed에서 별도로 실행한다.
과부하 Native도 모든 요청을 계속 추적한다.

새 Prompt의 Native N32, Colocated N32와 Dynamic N40은 전체 State 검증을 통과했다.
Dynamic N40은 Last0/2/4/8/12의 실제 P/D Layer 방문을 모두 확인했다.
강제 Cut 정합성 시험을 실제 Online Cut 선택 분포로 표현하지 않는다.
근거는 [Unseen State Qualification](dense_ar_slo_throughput_dynamic_layer_pd_20261114/analysis/frozen_unseen_full_state_qualification.json)이다.
장기 확인의 모든 완료 요청은 별도로 Output Parity와 Lifetime을 검사했다.
KV/Hidden 전체 Hash와 실제 Layer 방문의 상세 계측은 위 Qualification에서 수행했다.
Marker-free 성능 실행의 모든 요청에 같은 상세 State 계측을 켠 것은 아니다.

### 7.1 동결된 최초 선택점 — MEASURED

각 값은 독립 Start 3회의 평균이다.
95% CI와 Seed별 원값은 [원래 동결 확인 통계](dense_ar_slo_throughput_dynamic_layer_pd_20261114/analysis/frozen_independent_start_statistics.json)에 있다.
각 방법의 Offered QPS가 다르다.
따라서 이 표를 같은 QPS의 Latency 비교로 해석하지 않는다.

| 방법 | Offered QPS | TTFT p95 (s) | ITL p99 (ms) | TPOT p95 (ms) | Raw tokens/s | Qualified tokens/s | Decode-priority 적격 Start |
|---|---:|---:|---:|---:|---:|---:|---:|
| Native | 3.2 | 6.251805 | 24.380749 | 22.924798 | 51.662222 | 51.555556 | 1/3 |
| Strong Colocated | 5.0 | 5.393332 | 26.927546 | 24.190775 | 80.204444 | 80.071111 | 3/3 |
| Static Last12 / D4 | 4.4 | 4.016549 | 24.544892 | 23.767049 | 70.693333 | 70.417778 | 3/3 |
| Dynamic / D4 | 4.8 | 22.381416 | 25.142522 | 23.790980 | 75.511111 | 17.217778 | 0/3 |

Native의 두 실패는 Queue Trend의 보수적인 유한 구간 검사였다.
Seed11은 Completion/Arrival Ratio 1.005503이고 Queue도 11개에서 0개로 감소했다.
그러나 Slope 0.010616 requests/s가 사전 0.01 기준을 넘었다.

Seed13도 Queue가 11개에서 6개로 감소했다.
이 실행의 경험적 Slope 상한 0.025653이 사전 0.025 기준을 넘었다.
이를 무한 Queue 증가나 Native의 물리적 처리 불가능성으로 해석하지 않는다.
이 보수적 Protocol에서 Native의 독립 Capacity 확인은 `NOT_ESTABLISHED`다.

반면 Dynamic QPS4.8의 실패는 경미한 임계값 차이가 아니었다.
Seed별 TTFT p95는 30.769022/19.846835/16.528391초였다.
Arrival-cohort Joint Attainment는 16.912016/14.804661/33.862434%였다.
모든 실행에서 측정 중 도착한 요청을 완료까지 추적했다.
늦은 요청을 분모에서 제거하지 않았다.

Seed11의 Actual Arrival은 4.868333 requests/s였다.
Completion은 4.780000 requests/s였다.
600초의 차이는 Outstanding 증가 53개와 일치한다.
Discovery 통과점의 Actual Arrival 4.733333보다 높았다.
Nominal QPS4.8 하나만 보고 같은 Capacity 조건이라고 판단할 수 없다.
다른 Seed도 실패했으므로 해당 좋은 Discovery를 반복 가능한 성과로 채택하지 않았다.

### 7.2 보충 확인한 낮은 Dynamic Operating Point

정책과 Source를 그대로 둔 QPS4.4를 사전에 별도 선언했다.
새 Arrival Seed는 2026111516/2026111517/2026111518이다.
이 Rate 선택은 실패한 원래 선택점을 지우지 않는다.
새 Prompt 일반화 시험도 아니다.
원래 확인 Pool과 동결된 방법을 새 Arrival에서 검사한 보충 시험이다.
근거는 [보충 Protocol](dense_ar_slo_throughput_dynamic_layer_pd_20261114/configs/frozen_method_lower_capacity_protocol.json)이다.

| Seed | Actual QPS | Qualified tokens/s | TTFT p95 (ms) | ITL p99 (ms) | TPOT p95 (ms) | Joint Attainment | 적격 |
|---|---:|---:|---:|---:|---:|---:|---|
| 2026111516 | 4.515000 | 71.600000 | 4420.285787 | 24.910650 | 23.787727 | 99.003322% | PASS |
| 2026111517 | 4.403333 | 70.106667 | 4175.106943 | 24.627007 | 23.745611 | 99.659349% | PASS |
| 2026111518 | 4.536667 | 71.840000 | 5690.533748 | 25.030039 | 23.754641 | 99.412197% | PASS |

보충 확인의 Raw 평균은 71.644444 tokens/s였다.
Completed Rate 평균은 4.477778 requests/s였다.
Qualified 평균은 71.182222 tokens/s, 95% CI는 [70.430222, 71.788000]이다.
Strong Colocated의 Qualified 평균 CI는 [78.997778, 80.898667]이다.

서로 다른 Arrival Seed를 사용하므로 이 Capacity 비율은 Paired CI가 아니다.
각 방법의 Start를 따로 재표본화했다.
각 3 Start의 경험적 Bootstrap 조합은 총 729개다.
Dynamic/Colocated Qualified 비율 CI 상한도 1보다 낮았다.
Raw 비율은 0.893273, 95% CI는 [0.876234, 0.910734]다.
이는 각자의 적격점 비교다.
같은 Offered QPS에서의 Raw 우위나 정확한 Raw 최대를 의미하지 않는다.

모든 Latency 분위수, E2E, 실제 이전량과 CI는 [보충 포함 Capacity 통계](dense_ar_slo_throughput_dynamic_layer_pd_20261114/analysis/confirmed_capacity_with_backoff.json)에 있다.
표 형태 원값은 [Capacity CSV](dense_ar_slo_throughput_dynamic_layer_pd_20261114/analysis/confirmed_capacity_groups.csv)에 있다.

### 7.3 최대 확인점과 반복 범위

| 방법 | Decode-priority 최대 확인 Offered QPS | Qualified Output tokens/s | 독립 확인 | 범위 |
|---|---:|---:|---|---|
| Native | NOT_ESTABLISHED | NOT_ESTABLISHED | QPS3.2의 1/3 적격 | 두 실패가 보수적 Queue Trend 기준 |
| Strong Colocated | 5.0 | 80.071111 [78.997778, 80.898667] | 3/3 | QPS5.2 Discovery는 Queue 기준 실패 |
| Static Last12 / D4 | 4.4 | 70.417778 [69.157778, 71.497333] | 3/3 | QPS4.8 Discovery는 실패 |
| Dynamic / D4 | 4.4 | 71.182222 [70.430222, 71.788000] | 3/3 | QPS4.8의 독립 확인 0/3 |

QPS4.4와 4.8 사이를 세밀하게 탐색하지 않았다.
새 Method를 만들거나 Capacity가 없는 방법을 0 tokens/s로 표시하지 않았다.
Native의 미확정 Capacity도 0이 아니다.
Balanced와 TTFT-priority의 전체 Maximum Frontier는 완전히 확인하지 못했다.
이 선택점들에서 Balanced는 Static만 3/3이었다.
Colocated는 1/3, Dynamic QPS4.4는 2/3이었다.
이를 낮은 QPS의 Colocated도 실패한다는 뜻으로 확대하지 않는다.

Colocated의 별도 1200초 Persistence도 통과했다.
Raw는 80.44, Qualified는 80.12 tokens/s였다.
Arrival-cohort Joint Attainment는 99.602320%였다.
TTFT p95는 6701.222361 ms, ITL p99는 26.898496 ms였다.
해당 실행은 [별도 Persistence 근거](dense_ar_slo_throughput_dynamic_layer_pd_20261114/analysis/heldout_persistence_Colocated_O16_q5.0_s2026111515_W1200_v1_sustainable_v2.json)다.
이 긴 Start를 기존 3 Start와 동일 길이의 표본처럼 섞지 않았다.

### 7.4 같은 QPS4.8의 Native-relative Decode 검증

| Metric | Dynamic / Native | Paired Start Bootstrap 95% CI | 사전 5% 기준 |
|---|---:|---:|---|
| ITL p99 | 1.016167 | [1.013518, 1.018860] | PASS |
| TPOT p95 | 1.039578 | [1.039110, 1.039966] | PASS |
| TTFT p95 | 0.095447 | [0.077586, 0.115823] | 두 방법 모두 이 QPS에서 과부하 |

Relative Decode 두 분위수는 사전 경험적 CI 기준을 통과했다.
그러나 Dynamic의 해당 QPS는 Joint SLO·안정성 기준을 실패했다.
큰 TTFT 비율 개선을 Sustainable Serving GO로 표현하지 않는다.
QPS4.4의 같은 부하 Native 비교는 `NOT_MEASURED`다.
QPS4.8의 통과를 QPS4.4로 자동 이전하지 않는다.

Maximum ITL과 Threshold Exceedance에는 별도 위험이 남았다.
QPS4.8 Dynamic의 Seed별 최대 ITL은 876.007909/1151.496112/54.445451 ms였다.
같은 부하 Native는 25.999675/30.569280/39.609846 ms였다.
30 ms를 넘은 개별 Interval은 Dynamic 259개, Native 9개였다.
각 방법의 Measurement-arrival Cohort는 8674 requests, 총 130110 intervals다.
따라서 p99/TPOT 통과를 Maximum Tail까지 완전히 보호했다는 뜻으로 해석하면 안 된다.

Dynamic Seed12의 최악 ITL 1151.496112 ms 중 Worker Interval은 24.014782 ms였다.
Receive-lag 변화는 1127.481330 ms였다.
이 차이는 Host Delivery 지연 성분이다.
해당 Spike를 GPU Decode Kernel 실행시간으로 표시하지 않는다.
정확한 Host Blocking 경로의 인과 기여는 `NOT_IDENTIFIED`다.

낮은 QPS4.4도 최대 ITL 227.973131/445.309354/370.770438 ms가 남았다.
최악 Worker Interval은 각각 23.410175/24.536565/23.346992 ms였다.
Decode-priority의 30 ms 초과 Interval은 27/9/16개였다.
요청별 Joint Attainment는 95%를 넘지만, Maximum Latency 보장을 의미하지 않는다.
전체 First/Steady ITL과 분해값은 [Interval 근거](dense_ar_slo_throughput_dynamic_layer_pd_20261114/analysis/frozen_heldout_interval_details.json)에 있다.

### 7.5 실제 Layer 이전과 작은 Cut

QPS4.4의 Measurement-arrival Cohort는 총 8073 requests다.
6070 requests가 실제 D Prefill을 완료했다.
D로 이전한 Prefill Layer Work는 72236 layers다.
전체 290628 Prefill Layers의 약 24.85%다.
이는 577888000 Layer-token Rows다.
GPU 시간의 24.85% 절감이나 FLOPs 비율로 바꾸어 주장하지 않는다.

| 완료 Cut | QPS4.4의 실제 요청 수 |
|---|---:|
| Last0 | 2003 |
| Last2 | 0 |
| Last4 | 7 |
| Last8 | 137 |
| Last12 | 5926 |

작은 Cut은 실제 Online 선택이었다.
강제 정합성 시험과 구분한다.
그러나 작은 Cut이 Last12를 대체한 관측 사유는 `NO_AGGREGATE_COMPLETION_BENEFIT`였다.
최종 확인에서 Decode Deadline만으로 Last12를 거절하고 작은 Cut이 해결한 Serving 이득은 입증하지 못했다.
현재 Decode 보호의 핵심 실행 장치는 기존 Cohort Guard다.
Direct Deadline Predicate의 구현과 독립적인 정책 우위는 다른 주장이다.

같은 QPS4.4의 Dynamic/Static Qualified 비율은 1.010856이었다.
독립 Bootstrap 95% CI는 [0.988673, 1.035469]로 1을 포함했다.
TTFT p95 비율은 1.185589, CI는 [0.960853, 1.491237]이었다.
Arrival Seed, 실제 이전 Work와 Geometry가 다르므로 Matched Control도 아니다.
기존 Matched-budget Discovery까지 고려하면 독립 Dynamic 기여는 `NOT_ESTABLISHED`다.
세부 한계는 [동일 Budget 분석](dense_ar_dynamic_vs_static_matched_budget.md)에 있다.

### 7.6 Predictor의 실제 오차와 안전성 범위

아래 표는 각 Run의 실제로 선택된 Migration Group 전체를 사용한다.
Warm-up과 Follow-through를 포함하므로 Measurement Cohort 수와 다르다.
명목 Group First-output은 경험적 p95 Component를 조합한 예측이다.
통계적으로 불편한 평균이나 GPU Kernel 완료 예측은 아니다.

| QPS / Seed 끝자리 | Group 수 | 명목 MAE (ms) | 명목 p95 / p99 절대오차 (ms) | 명목 과소예측률 | Upper Coverage |
|---|---:|---:|---:|---:|---:|
| 4.8 / 11 | 375 | 30.598256 | 53.400986 / 476.084231 | 48.533333% | 99.200000% |
| 4.8 / 12 | 394 | 65.639601 | 169.250531 / 503.594659 | 26.395939% | 99.492386% |
| 4.8 / 13 | 340 | 85.855676 | 143.229976 / 514.787013 | 18.823529% | 100.000000% |
| 4.4 / 16 | 365 | 134.379064 | 515.839096 / 543.591178 | 13.698630% | 99.726027% |
| 4.4 / 17 | 327 | 158.300021 | 521.875252 / 536.548744 | 8.868502% | 100.000000% |
| 4.4 / 18 | 346 | 106.266954 | 473.112555 / 512.807238 | 28.034682% | 100.000000% |

낮은 부하의 큰 절대오차는 대체로 과대예측이었다.
그럼에도 QPS4.8 Seed12의 Group Upper Underprediction은 최대 775.965188 ms였다.
개별 Request의 Upper Underprediction은 최대 843.017731 ms였다.
150 ms Margin이 WCET나 모든 미래 완료의 보장은 아니다.
Policy Host p95는 여섯 Run에서 약 2.44~2.58 ms였다.
각 실행의 종속 Group을 독립 Start처럼 사용하지 않았다.
전체 오차, Known Drain과 Candidate Gate는 [Prediction Validation v2](dense_ar_slo_throughput_dynamic_layer_pd_20261114/analysis/online_prediction_validation_v2.json)에 있다.
Gate 수는 겹치므로 요청 수처럼 더하지 않는다.

## 8. Long Output과 Heterogeneous/Bursty 결과의 범위

Output64/200의 Migration0은 단순한 물리적 Slack 부족으로 확인되지 않았다.
기존 Trace에서 Known-ingress Gate는 Cut별 1152/1152개 Request를 거절했다.
Wait Limit0 Gate는 Cut별 1143/1152개를 거절했다.
이 사유들은 겹친다.
Wait Limit만 늘린 Fixed-state Counterfactual에서는 Migration이 여전히 0이었다.
두 Gate를 함께 제거한 수치는 Offline 대안일 뿐 실제 Online 이득이 아니다.

Output200에서는 Source KV Retention과 D Cohort Admission이 P Allocation을 막았다.
Coordinator ITL 198.888972 ms는 Worker Interval 28.077098 ms와 Delivery-lag 변화 170.811875 ms로 분해됐다.
Worker Interval도 Exclusive GPU Kernel Time은 아니다.
정확한 Host Delay 원인은 `NOT_IDENTIFIED`다.
세부 근거는 [Long-output Root Cause](dense_ar_long_output_zero_migration_root_cause.md)에 있다.

Bursty와 Mixed-output Discovery도 실행했다.
이 시험은 최종 Canonical v5 이전 Runtime의 단일 Start다.
Bursty에서 Static Last12와 Colocated보다 나은 Dynamic 순이익은 확인하지 못했다.
Mixed Output에서도 Dynamic의 독립적 이득은 확인하지 못했다.
Input-length Calibration과 State 검증을 Online Heterogeneous-input 성과로 표현하지 않는다.
실제 Natural EOS와 Production Trace는 `NOT_RUN`이다.
자세한 숫자는 [Heterogeneous/Bursty 결과](dense_ar_heterogeneous_bursty_crossover_results.md)에 있다.

## 9. 측정 한계와 미실행 항목

이 캠페인에서는 CUPTI/Nsight Capture를 실행하지 않았다.
따라서 실제 Kernel Union, Physical Idle, SM/HBM Counter는 `NOT_MEASURED`다.
과거의 실패한 Capture를 이번 새 실패로 집계하지 않는다.
NVML의 보수값을 물리적 Idle로 사용하지 않는다.

단독 CUDA Event는 Calibration Enclosure다.
GEMM, Attention, KV와 Host Scheduling의 Exclusive 실행시간은 완전히 분리하지 못했다.
겹칠 수 있는 p95 Enclosure를 더해 Online Critical-path Saving으로 표현하지 않는다.
실제 P/D Figure는 Host Observation으로 표시한다.

최종 v5의 Heterogeneous-input Online Serving, Long-output 독립 확인과 Spatial Co-execution은 `NOT_RUN`이다.
최종 v5의 Matched-work/동일 Geometry 독립 확인도 `NOT_RUN`이다.
Controller Zero와 과거 정책을 최종 3회 확인의 별도 Arm으로 추가하지 않았다.
이 한계를 모든 Workload의 구조적 불가능성으로 확대하지 않는다.

## 10. 최종 질문에 대한 답

| 질문 | 답과 증거 수준 |
|---|---|
| Strong Colocated의 목적별 최선 설정은 무엇인가? | `MEASURED`: Decode/Balanced는 FULL/B512/S16 Async가 선택됐다. Raw 후보는 B8192/S32 Decode-only Graph다. 전역 최적은 미증명이다. |
| Decode-priority 재튜닝 이후에도 Dynamic이 더 좋은가? | 아니다. 확인된 Qualified 비율은 0.888988 [0.872953, 0.906359]다. |
| 각 방법의 실제 지속 Capacity는 얼마인가? | 확인된 Tested Point는 C 80.071111, Static 70.417778, Dynamic 71.182222 tokens/s다. Native는 보수적 검사에서 `NOT_ESTABLISHED`다. |
| 같은 SLO에서 Dynamic의 최대 유효 Output이 더 높은가? | 검사한 Grid에서는 아니다. 세밀한 전역 최대 탐색은 수행하지 않았다. |
| Raw에서도 이겼는가? | 아니다. 적격점 Raw 비율도 0.893273이다. C-Raw 105.92는 별도 단일 Discovery다. 정확한 Raw 최대나 단일 자원 기여는 미확정이다. |
| Output64/200 Migration0의 원인은 무엇인가? | 기존 Known-ingress/Wait Gate, 긴 Cohort Residency와 Source KV Retention이 확인됐다. 물리적 GPU Slack 부재로 판정하지 않았다. |
| Wait Limit0의 제한은 얼마나 큰가? | 기존 Cut별 1143/1152 Request를 거절했다. 다른 Gate와 겹친다. Wait만 풀어도 고정 상태 대안은 0이었다. |
| 작은 Cut이 안전하고 유리한가? | Last4 7건과 Last8 137건을 QPS4.4 Cohort에서 완료했다. Deadline-rescue나 독립 Serving 우위는 입증하지 못했다. |
| Real Deadline을 Candidate 선택에 반영했는가? | 직접 Predicate와 CPU Regression을 구현했다. 실제 실행은 기존 Cohort Guard도 유지했다. |
| Deadline을 실제로 얼마나 보호했는가? | QPS4.8의 p99/TPOT Ratio CI는 통과했다. 하지만 30 ms 초과 259건과 큰 Host Delivery Tail이 남았다. 절대적 최대 지연 보장은 없다. |
| Heterogeneous Input에서 Static이 불리했는가? | 길이별 State/Cost 검증은 수행했다. 최종 Runtime의 Heterogeneous-input Online Serving은 `NOT_RUN`이다. |
| Bursty에서 Dynamic의 추가 가치가 있는가? | 단일 Discovery에서는 Strong Static/Colocated 우위를 확인하지 못했다. 최종 v5 독립 반복은 `NOT_RUN`이다. |
| 동일 Work/Geometry에서 Static보다 좋은가? | 기존 Matched 총량 시험은 Geometry Confound가 남았다. 독립 정책 기여는 `NOT_ESTABLISHED`다. |
| Cost 오차와 Tail Coverage는 얼마인가? | 위 Seed별 표와 Prediction v2에 있다. 경험적 Upper Coverage는 높지만 최악 오차와 Host Tail이 남았다. WCET가 아니다. |
| 독립 Systems Contribution을 확보했는가? | 현재 Dynamic 정책의 논문 Main Claim은 확보하지 못했다. Generic KV Layout의 기전 개선은 별도로 확인했다. |

이번 NO-GO는 원본 Layer-axis Continuation이 불가능하다는 판정이 아니다.
동일 GPU/SLO에서 Strong Colocated를 이기는 Dynamic Serving 주장을 확보하지 못했다는 판정이다.
특히 Cohort Geometry, Handoff/Control Plane과 보수적인 Completion 모델의 한계가 남았다.
Exclusive Kernel/SM/HBM 기여와 구조적 Hardware 한계는 `NOT_IDENTIFIED`다.
Source/Destination Ingress를 근본적으로 바꾸는 후속 구현은 이번 최종 방법에 조용히 합치지 않았다.

## 11. Figure, 검증 및 재현

- [실험 Spec](dense_ar_slo_throughput_dynamic_layer_pd_spec.md).
- [Strong Colocated Frontier](dense_ar_slo_specific_colocated_frontier.md).
- [Long-output Root Cause](dense_ar_long_output_zero_migration_root_cause.md).
- [Deadline-aware Admission](dense_ar_deadline_aware_small_cut_admission.md).
- [Heterogeneous/Bursty 결과](dense_ar_heterogeneous_bursty_crossover_results.md).
- [Sustainable Throughput 프로토콜](dense_ar_sustainable_generation_throughput.md).
- [Matched-budget 분석](dense_ar_dynamic_vs_static_matched_budget.md).
- [Figure와 Statistics](dense_ar_layer_pd_main_figures_and_statistics.md).

주요 Figure는 다음과 같다.

- [Figure 1: Frozen Point TTFT](dense_ar_slo_throughput_dynamic_layer_pd_20261114/figures/final_1_TTFT_p95_ms.svg).
- [Figure 2: TPOT](dense_ar_slo_throughput_dynamic_layer_pd_20261114/figures/final_2_TPOT_p95_ms.svg).
- [Figure 3: Raw Output](dense_ar_slo_throughput_dynamic_layer_pd_20261114/figures/final_3_raw_completed_output_tokens_s.svg).
- [Figure 4: Qualified Output](dense_ar_slo_throughput_dynamic_layer_pd_20261114/figures/final_4_SLO_qualified_throughput.svg).
- [Figure 5: Frozen 선택점의 적격성](dense_ar_slo_throughput_dynamic_layer_pd_20261114/figures/final_5_confirmed_capacity_points.svg).
- [Figure 6: 실제 Online Cut](dense_ar_slo_throughput_dynamic_layer_pd_20261114/figures/final_6_actual_Dynamic_cuts.svg).
- [Figure 7: Bursty/Mixed Output](dense_ar_slo_throughput_dynamic_layer_pd_20261114/figures/final_7_bursty_and_mixed_output.svg).
- [Figure 8: Matched-budget Discovery](dense_ar_slo_throughput_dynamic_layer_pd_20261114/figures/final_8_static_Dynamic_Pareto.svg).
- [Figure 9: 보충 확인 포함 최대 적격 Tested Point](dense_ar_slo_throughput_dynamic_layer_pd_20261114/figures/final_9_confirmed_capacity_with_backoff.svg).
- [Host P/D Timeline](dense_ar_slo_throughput_dynamic_layer_pd_20261114/figures/final_Host_P_D_timeline.svg).

Figure 1~6은 원래 동결 선택점이다.
실패한 Dynamic QPS4.8도 그대로 표시한다.
Figure 9는 이후 사전 선언한 낮은 Rate의 확인을 별도로 포함한다.
서로 다른 Output Length나 재튜닝된 설정을 한 Curve로 연결하지 않았다.
공통 설정의 완전한 네 방법 QPS Grid를 모두 독립 확인한 것은 아니다.

Host Timeline은 같은 QPS4.8, Arrival Seed와 Request를 비교한다.
고정 규칙으로 대표 Request와 다음 5개를 선택했다.
두 방법 모두 이 QPS에서는 과부하다.
Host Enclosure의 겹침을 실제 Kernel Overlap이나 물리적 Idle로 해석하지 않는다.

전체 Ledger는 255개 실행을 보존한다.
250개는 Runtime `COMPLETE`, 5개는 Terminal Failure다.
`COMPLETE`가 Output Exactness나 SLO 적격을 자동으로 뜻하지는 않는다.
실험 Process Wall Time 합계는 69325.458651초, 약 19.257시간이다.
Startup, Warm-up, Host Waiting과 Drain을 포함한다.
GPU Kernel Busy 시간으로 표현하지 않는다.
원래 15개 독립 확인, 별도 1200초 Persistence와 3개 보충 확인을 모두 보존했다.
시간 규칙으로 생략한 Upper Neighbor와 Dynamic Persistence는 `NOT_RUN_BUDGET`이다.

실행별 Command, Immutable Configuration, Model Revision, Dataset ID와 Source ZIP Hash는 Raw Evidence에 있다.
각 요청의 실제 완료 Layer Work와 SLO 분모도 보존했다.
큰 원본은 삭제하지 않고 SHA256을 검사한 Lossless Archive로 공개한다.
새 실행은 별도 Namespace를 사용해야 한다.
원래 Raw Evidence를 덮어쓰지 않는다.

Publication Branch는 `flashvep/dense-ar-slo-throughput-dynamic-layer-pd-20261114`다.
Runtime Freeze SHA256은 `6dbd5188fdc83e1b997e3a883c298369b2c2e0e4f07fa0f880e66cec4baa4a0f`다.
사용자 Checkout, Staged Index, 기존 Source와 외부 Runtime을 보존한다.
GPU 작업 종료 후 승인된 Burner 8개의 새 Generation을 두 차례 확인했다.
근거는 [GPU 복원 검증](dense_ar_slo_throughput_dynamic_layer_pd_20261114/state/resource_restoration_verified.json)이다.
최종 CPU Unit Test 85개와 Source AST 검사는 모두 통과했다.
문서의 Local Link 누락은 0개였다.
원본 Safetensors 5개의 실제 SHA256은 각각 HF LFS Content Hash와 일치했다.
기존 Source, 사용자 Checkout/Index와 설치된 외부 Runtime의 보존 검사도 통과했다.
검증은 [final_verification.json](dense_ar_slo_throughput_dynamic_layer_pd_20261114/state/final_verification.json)에 있다.
원본 Weight 검사는 [별도 Checksum 근거](dense_ar_slo_throughput_dynamic_layer_pd_20261114/state/model_checkpoint_integrity_verified.json)에 있다.

Commit 범위의 Whitespace 검사에는 Source 끝의 빈 줄 경고 27개가 있다.
실측 Source와 동결 Hash를 유지하기 위해 해당 줄을 제거하지 않았다.
이 검사를 경고 없는 통과로 표현하지 않는다.
기능·수치 정합성 검사와는 별개다.
근거는 [Whitespace Audit](dense_ar_slo_throughput_dynamic_layer_pd_20261114/state/frozen_source_whitespace_audit.json)에 있다.

큰 Evidence 25478913961 Bytes는 208478356 Bytes의 Lossless Archive로 보존했다.
공개용 파일은 5개 Part로 나눴다.
모든 원본의 SHA256을 압축 해제 Stream과 대조했다.
로컬 원본도 그대로 남겼다.
목록과 복구 규칙은 [Archive Manifest](dense_ar_slo_throughput_dynamic_layer_pd_20261114/archives/large_evidence_manifest.json)에 있다.
최종 Commit/Remote SHA는 Push 후 응답과 로컬 Publication Receipt에 기록한다.

# Dense AR Last2·Last4 실제 Continuation 빠른 비교

## 결과와 범위

두 Pair 모두 Batch1의 실제 D Forward에서 Layer가 더 빨랐다.
Last4의 Token/Layer 중앙값 비율은1.26418이다.
Last2의 해당 비율은3.51761이다.
이 값은 대표 Layer의 시간을 곱한 추정치가 아니다.
P에서 계산한 실제 State를 D로 넘긴 뒤 실행한 Forward를 측정했다.

각 방식은 독립 Process Start1회다.
Start마다18개 실제 원문 요청을 실행했다.
처음9개 요청으로 두 인접 Suffix Shape를 예열했다.
마지막9개 요청의 측정값을 비교했다.
요청9개를 독립 Start9회로 취급하지 않는다.
독립 Start 기반95% CI는 없다.

이 결과는 `MEASURED`인 실행 구간 비교다.
Online Serving의 TTFT·Goodput 우위나 Decode5% 비열등성 판정은 아니다.

## 실행 환경

| 항목 | 값 |
|---|---|
| Model | Qwen/Qwen3-8B |
| Revision | `b968826d9c46dd6066d109eabc6255188de91218` |
| 계산 | Original BF16, Exact Causal Attention, Full36 Decode |
| Sampling | Greedy, Thinking Disabled, `ignore_eos=True` |
| GPU | H100 SXM80GB, P0/D1 |
| GPU0 UUID | `GPU-f217c8a0-1142-20f4-d84b-af29f3a47a0d` |
| GPU1 UUID | `GPU-a77f3471-67d4-20b0-9fab-e502d4de5adb` |
| Runtime | vLLM0.26.0+cu129, PyTorch2.11.0+cu129, NIXL1.4.1 |
| Worker Scheduler | 양쪽 모두 B8192/S1024 |
| KV | GPU당42GiB, Prefix Cache Disabled |
| Workload | Input8000/Output16 |
| 도착 | 요청 간격2초의 직렬 진단, 설정 Rate0.5 requests/s |
| D Admission | 공통 상한1, 실제 준비 그룹1 |
| Dataset | 기존 `heldout128_8000.json`의 첫18개 원문 |
| 측정 요청 | `q9`–`q17` |
| Source Parent | `228a041f35cedcee9f45beba6c462b453fb3a2c5` |

이는 QPS2–10의 Primary Sweep과 별도인 진단 실험이다.
직렬 진단의 Token Rate를 Capacity로 해석하지 않는다.
모든 요청의 원문 ID·Input IDs·Output IDs와 실행 Source ZIP을 보존했다.
새 Namespace의 기존 Runtime·SLO·Primary Partition은 변경하지 않았다.

## 정확한 작업량

현재 KV 전송은16-Token 경계를 사용한다.
따라서889·444를 모든 요청에 그대로 적용하지 않았다.
인접한 두 Suffix를 사용해9개 요청의 총 작업량을 정확히 맞췄다.

| Pair | Layer의 D 작업/request | Token의9-request Suffix 분포 | 각 방식의 D Token-layer Rows/9 requests |
|---|---|---|---:|
| Last4 | 8000×4 Layers | 880×4 requests, 896×5 requests | 288,000 |
| Last2 | 8000×2 Layers | 432×2 requests, 448×7 requests | 144,000 |

평균 Suffix는 각각888.888889와444.444444 Tokens다.
전체18개 요청은 같은9-request 작업량 주기를 두 번 사용한다.
P/D Query Coverage와 실제 이전 작업량 검사도 통과했다.
같은 Token-layer Rows는 같은 Attention Work나 전체 Operator 비용을 의미하지 않는다.
Layer에는 Hidden/Residual 전송도 추가된다.

## 정합성 — CORRECTNESS_ONLY

새 Native N18 Reference를 먼저 실행했다.
Last2 Layer와 Token432/448을 각각 N18로 검사했다.
두 방식 모두 다음 검사를 통과했다.

- Original BF16 Reference의 Greedy Output Token.
- 완성된 Full Prompt KV.
- Final Hidden과 Token Prefix/Suffix Hidden 구간.
- 실제 P Prefix·D Suffix Layer 방문과 Query 위치.
- Full36 Decode Graph 정의.
- NIXL ACK와 정상 KV·Hidden·Reservation Retirement.

Last4와 Token880/896은 동일 Revision3 Runtime의 기존 N128 State Gate를 재사용했다.
새 Timing 실행에서는18개 요청 모두의 Output·Query Coverage·작업량·정상 Lifetime을 다시 검사했다.
Timing 실행에 Full KV Hash의 CPU 복사를 넣지는 않았다.

새 State Gate 이름은 다음과 같다.

- `priority_suffix_qual_Native_Last0_N18_O16_v1`
- `priority_suffix_qual_Layer_Last2_N18_O16_v1`
- `priority_suffix_qual_Token_Last2_N18_O16_v1`

기존 Last4 State Gate는 `qual_Layer_Last4_N128_v10`과 `qual_Token_Last4_N128_v10`이다.

## 실제 D Forward — MEASURED

CUDA Event는 실제 Qwen Forward 호출을 둘러싼다.
Layer는 전송받은 Hidden/Residual에서 마지막2개 또는4개 Layer를 실행한다.
Token은 전송받은 Prefix KV를 이용해 Suffix를 Full36으로 실행한다.
해당 구간에는 Attention·Norm·Activation과 Final Norm이 포함된다.
별도 Handoff·Admission·LM-head/Sampling 비용까지 모두 포함한 구간은 아니다.
Event 구간에는 Stream 대기와 Host Launch Gap이 포함될 수 있다.
이를 Exclusive GPU Kernel Time으로 표시하지 않는다.

| Pair | Layer p50, ms | Token p50, ms | Token/Layer p50 | Layer p95, ms | Token p95, ms |
|---|---:|---:|---:|---:|---:|
| Last4 / Token≈889 | 30.3130 | 38.3212 | 1.26418 | 30.6531 | 38.4401 |
| Last2 / Token≈444 | 15.2960 | 53.8052 | 3.51761 | 15.5699 | 56.3665 |

Layer의 Forward p50은 Token 대비 각각20.8975%와71.5716% 낮았다.
Last2 Token의9개 Event Sample은46.6625–56.4752ms 범위였다.
단일 Start에서 얻은 이 변동을 독립 반복의 CI로 바꾸지 않는다.

앞서 제시한16.29/24.31ms는 실제 고립 Projection 측정값에 Layer 수를 곱한 합산 추정이었다.
이번30.3130/38.3212ms는 전체 D Forward의 실제 실행 구간이다.
두 표는 비용의 범위가 다르다.
기존 추정값을 이번 Forward 실측값으로 바꾸어 표시하지 않는다.

## 직렬 Client TTFT — MEASURED, 별도 지표

아래 값도 마지막9개 요청에서 계산했다.
P Forward·Handoff·D 준비와 Client 전달을 포함한다.
동시 고부하 Serving의 대기열 성능과는 다르다.

| Pair | Layer TTFT p50, ms | Token TTFT p50, ms | Layer TTFT p95, ms | Token TTFT p95, ms |
|---|---:|---:|---:|---:|
| Last4 / Token≈889 | 316.2744 | 328.3598 | 318.0027 | 329.9976 |
| Last2 / Token≈444 | 310.4877 | 369.0893 | 323.9960 | 381.2160 |

TTFT p95는 각각3.6348%와15.0099% 낮았다.
이 비율에는 독립 Start 기반 CI가 없다.
Forward의1.26418×·3.51761× 차이를 같은 비율의 TTFT 개선으로 해석하지 않는다.

## Projection 상세 계측의 실패 — FAILED / NOT_MEASURED

실제 모든 Layer의 Projection을 개별 Event로 측정하는 별도 Observer도 준비했다.
첫 Last4 Observer는 Model Startup에서 실패했다.
래핑한 Constructor의 Signature가 `vllm_config` 인자를 노출하지 못했다.
오류는 `Qwen3ForCausalLM.__init__() missing 1 required keyword-only argument: 'vllm_config'`다.
이 실패는 측정 Window 이전에 발생했다.
정상 완료한 위 네 Forward 실행에는 이 Projection Observer를 활성화하지 않았다.

실패 실행은 `priority_suffix_projection_probe_Layer_Last4_N18_O16_v1`이다.
Raw Log와 Dispatcher 실패 Receipt를 그대로 보존한다.
이 실행에서 실제 GEMM별 시간을 얻었다고 표시하지 않는다.
나머지 세 Projection Probe는 원래 Sweep으로 복귀하라는 사용자 요청에 따라 `NOT_RUN`이다.
따라서3.51761× 전체 Forward 차이가 순수 GEMM만으로 발생했다고 주장하지 않는다.

## 해석과 복귀

이 두 작은 Partition에서는 실제 Batch1 Continuation 이득을 확인했다.
그러나4 Layers가 보편적인 성능 경계라고 판단하지 않는다.
다른 Token Batch와 Scheduler에서는 결과가 달라질 수 있다.
이번 Admission1 진단을 Strong Token Serving Baseline의 대체 결과로 사용하지 않는다.
Primary QPS Sweep의 설정은 유지했고 실행도 재개됐다.
추가 GPU Projection 계측은 실행하지 않는다.

## 근거

- [실측 Sample과 비교 비율](dense_ar_matched_axis_crossover_20261117/analysis/priority_actual_suffix_results.json)
- [실행 전 측정 계획](dense_ar_matched_axis_crossover_20261117/state/priority_actual_suffix_design.json)
- [Last2 Layer State Gate](dense_ar_matched_axis_crossover_20261117/analysis/priority_suffix_qual_Layer_Last2_N18_O16_v1_state_gate.json)
- [Last2 Token State Gate](dense_ar_matched_axis_crossover_20261117/analysis/priority_suffix_qual_Token_Last2_N18_O16_v1_state_gate.json)
- [전체 Forward Raw, Last4 Layer](dense_ar_matched_axis_crossover_20261117/raw/priority_suffix_actual_forward_Layer_Last4_N18_O16_v1/)
- [전체 Forward Raw, Last4 Token](dense_ar_matched_axis_crossover_20261117/raw/priority_suffix_actual_forward_Token_Last4_N18_O16_v1/)
- [전체 Forward Raw, Last2 Layer](dense_ar_matched_axis_crossover_20261117/raw/priority_suffix_actual_forward_Layer_Last2_N18_O16_v1/)
- [전체 Forward Raw, Last2 Token](dense_ar_matched_axis_crossover_20261117/raw/priority_suffix_actual_forward_Token_Last2_N18_O16_v1/)

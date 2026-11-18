# Dense AR B8192/S1024 Runtime 감사

## 판정과 증거 범위

`PASS_CONFIG_AND_STATE`다.
네 방식의 실제 Worker가 B8192/S1024를 사용했다.
Revision3의 N8·N32·N128 검사가 모두 통과했다.
Output200은 별도 N8 검사도 통과했다.
이 판정은 성능 우위나 Crossover를 의미하지 않는다.

증거 Namespace는 [dense_ar_matched_axis_crossover_20261117](dense_ar_matched_axis_crossover_20261117/)이다.
구성 검사는 `MEASURED`다.
출력·KV·Hidden 검사는 `CORRECTNESS_ONLY`다.
실제 Batch와 예약 이력의 해석은 `TRACE_ANALYSIS`다.

## 공통 환경

| 항목 | 값 |
|---|---|
| Model | Qwen/Qwen3-8B |
| Model Revision | `b968826d9c46dd6066d109eabc6255188de91218` |
| 계산 | Original BF16, Exact Causal Attention, Full36 Decode |
| Sampling | Greedy, Thinking Disabled, `ignore_eos=True` |
| Workload | Input8000, Output16/32/64/128/200, Open-loop Poisson |
| GPU | H100 SXM 80GB × 2, GPU0/1 |
| GPU0 UUID | `GPU-f217c8a0-1142-20f4-d84b-af29f3a47a0d` |
| GPU1 UUID | `GPU-a77f3471-67d4-20b0-9fab-e502d4de5adb` |
| Runtime | vLLM0.26.0+cu129, PyTorch2.11.0+cu129, NIXL1.4.1 |
| KV | GPU당42GiB, Prefix Cache Disabled |
| Source Parent | `228a041f35cedcee9f45beba6c462b453fb3a2c5` |
| Branch | `flashvep/dense-ar-matched-axis-crossover-20261117` |

Model Context Limit은40960이다.
Input8000에는 기존 Chat Template의 특수 Token이 포함된다.
실행 시 Model Length 상한은 Input과 Output Cap에서 파생한다.
따라서8016/8032/8064/8128/8200을 사용한다.
같은 Output에서는 모든 방식의 상한이 같다.
Output 간 비교에서는 이 파생 차이도 공개한다.

## 1. 실제 Worker B/S

B는 Iteration당 Query Token Budget이다.
S는 Worker의 최대 Sequence 수다.
둘 다 실제 GPU Batch 크기와는 다르다.

| 방식 | GPU0 | GPU1 | 구성 검사 |
|---|---|---|---|
| Aggregated DP2 | Replica0:8192/1024 | Replica1:8192/1024 | PASS |
| Native PD | P:8192/1024 | D:8192/1024 | PASS |
| Static Layer PD | P:8192/1024 | D:8192/1024 | PASS |
| Static Token PD | P:8192/1024 | D:8192/1024 | PASS |

`effective_runtime_<role>.json`은 생성된 Model Runner의 실제 Scheduler Config를 읽는다.
Config Dictionary만 비교한 결과가 아니다.
각 실행의 `unified_worker_gate.json`에 두 Worker의 원본 관측을 저장한다.
각 실행의 `command.json`에는 실행 명령과 Source ZIP의 SHA256을 저장한다.

Agg는 표준 Continuous Batching과 Full Mixed CUDA Graph를 사용한다.
PD의 P는 검증된 Eager Partial Forward를 사용한다.
PD의 D는 Full36 Decode Graph를 사용한다.
Layer/Token의 D Continuation은 검증된 Eager 경로다.
따라서 B/S 통일을 Graph 실행 방식 통일로 해석하지 않는다.

## 2. 공통 PD Admission 계약

Native/Layer/Token에 다음 계약을 적용한다.

- D Admission 상한8.
- FIFO와 기존 Whole-cohort Guard.
- 활성 D Cohort가 끝난 뒤 다음 Cohort 수용.
- P Source Macro 상한8.
- P Active·D Pending·D Active를 포함한 공통 Outstanding Window32.
- P 제출 전에 전체 D Prompt·Output의 KV Credit 예약.
- 기존 Commit·할당·Retirement 계약 유지.
- 예약 거절 시 같은 고정 Partition을 기다림.

예약량은 `ceil((8000 + OutputCap) / 16)` Blocks/request다.
기존 Ledger의 Cut 검사에36을 추가했다.
Native와 Token도 이 Ledger를 사용한다.
새 KV Manager는 구현하지 않았다.
예약 거절을 이유로 Layer를 Native로 바꾸지 않는다.

실제 Grant와 Retirement를 Request별로 검사한다.
Native N8의 첫 검사에서8개 Grant와8개 Retirement를 확인했다.
Generation의 Cut은36이었다.
출력·Query Coverage·KV 반환 검사도 통과했다.
이는 정상 완료 경로의 검증이다.
모든 장애·취소 상황을 검증했다는 의미는 아니다.

## 3. 같은 상한과 다른 실제 GPU Geometry

현재 D Continuation은 한 Iteration에서 해당 Prompt Query를 끝내야 한다.
Coordinator는 Cohort의 Query 합을8192 이하로 제한한다.
이 Predicate를 세 PD 방식에 동일하게 적용한다.

| 방식 | Request당 D 준비 Query | 가능한 Cohort 상한 |
|---|---:|---:|
| Native | 기존 Prompt KV를 이용하는 새 Token Query1 | 8 |
| Layer Last4/8/12 | 8000 | 1 |
| Token880/896 | 880 또는896 | 8 |
| Token1776/1792 | 1776 또는1792 | 4 |
| Token2656/2672 | 2656 또는2672 | 3 |

이는 `CAP_MATCHED / ACTUAL_GEOMETRY_DIFFERENT`다.
같은 B/S와 Admission 상한이 같은 Decode Batch를 보장하지 않는다.
이 제약은 현재 Atomic-frontier Runtime의 계약이다.
Dense AR Layer Continuation 자체의 구조적 불가능성으로 해석하지 않는다.
여러 Iteration으로 D 준비를 나눠 Cohort를 모으는 대안은 Primary에 구현하지 않았다.

다음 값은 N72 Pilot의 실제 Cohort 개수다.
QPS6, Arrival Seed2026111781, Last12 Pair를 사용했다.
Pilot Pool은64개 원문과 명시적으로 반복한8개 요청이다.
이를 독립된72개 원문으로 표시하지 않는다.

| Output | 방식 | Cohort 크기:개수 |
|---|---|---|
| 16 | Native | 1:44, 2:14 |
| 16 | Layer Last12 | 1:72 |
| 16 | Token2656/2672 | 1:1, 2:4, 3:21 |
| 200 | Native | 1:1, 8:8, 7:1 |
| 200 | Layer Last12 | 1:72 |
| 200 | Token2656/2672 | 1:1, 3:23, 2:1 |

공통 Admission1 보조 실험은 이 Geometry 효과를 검사한다.
이를 Admission8 Primary의 대체 결과로 사용하지 않는다.

## 4. 실제 P Chunk Packing

P는 표준 Chunked Prefill을 유지한다.
여러 요청의 Query를 B8192 안에 채울 수 있다.
Layer의 Hidden/Residual Fragment는 원래 Token 위치의 GPU Buffer에 모은다.
전체8000행이 완성되기 전에는 D에 Descriptor를 공개하지 않는다.
기존 NIXL Read와 Source ACK 계약을 유지한다.

초기 Chunked N8 관측에서 Native와 Layer 모두15개 Source Fragment를 실행했다.
8192 Query·2 Request인 실제 Input Batch가7개였다.
마지막 Batch는6656 Query·1 Request였다.
전체 Prompt Query와 State가 일치했다.
이 관측은 Kernel Busy 측정이 아니다.

Primary에서는 Request별 Query 시작점과 길이를 다시 검사한다.
누락·중복 Query가 없어야 한다.
입력·Padding·Source Macro·D Cohort를 각각 기록한다.

## 5. 정합성 및 동일 작업량

| 검사 | N8 | N32 | N128 | 별도 Output200 N8 |
|---|---|---|---|---|
| Native | PASS | PASS | PASS | PASS |
| Agg DP2 | PASS | PASS | PASS | PASS |
| Layer Last4/8/12 | PASS | PASS | PASS | PASS |
| Token의 세 인접 Suffix Pair | PASS | PASS | PASS | PASS |

검사 항목은 다음과 같다.

- Original BF16 Reference의 Greedy Output Token.
- 유효한 Full Prompt KV와 Final Hidden.
- Token Prefix/Suffix의 Hidden 구간과 Query 경계.
- 실제 P Prefix·D Suffix Layer 방문.
- Full36 Decode Graph 정의와 실행.
- Hidden Hash·NIXL ACK·정상 KV/Reservation Retirement.

정합성 실행의 CPU Tensor Hash는 Latency를 바꾼다.
따라서 그 실행의 TTFT/ITL을 성능 결과로 사용하지 않는다.
Clean Serving에서는 출력과 작업량·정상 Lifetime을 검사한다.
Full KV Hash를 모든 Clean 요청에서 다시 계산하지 않는다.

성능의 정확한 작업량 비교는 N144를 사용한다.
N128에서는 세 Pair의 Integer Suffix 합을 동시에 정확히 맞출 수 없다.
N8/N32/N128 State 검사는 인접 Suffix 경계를 모두 검사하기 위한 것이다.

| Pair | N144 Token Suffix 분포 | D GEMM Token-layer Rows |
|---|---|---:|
| Last4 | 880×64, 896×80 | 4,608,000 |
| Last8 | 1776×128, 1792×16 | 9,216,000 |
| Last12 | 2656×48, 2672×96 | 13,824,000 |

이 값은 `STRUCTURAL_ACCOUNTING`이다.
실제 완료 Query와 Layer 경로를 추가로 검사한다.
같은 Rows는 같은 Attention Work나 GPU 시간을 의미하지 않는다.
Source/Destination 모두의 Rows와 유효 KV Payload를 검사한다.
LM Head·Sampling·Padding·Metadata 비용도 별도 한계로 남긴다.

## 6. 기존 PD 설정에서 바뀐 점

기존 Native의 Worker 설정은 P32000/4, D128000/16이었다.
이번에는 두 Worker를8192/1024로 맞췄다.
기존 방식별 D Admission 차이도 공통 상한8로 바꿨다.
Source Macro와 공통 Outstanding Window도 명시했다.
Native와 Token에는 사전 KV 예약이 추가됐다.

따라서 과거·현재 차이를 B/S 하나의 인과 효과로 표시하지 않는다.
같은 과거 Prompt/Arrival을 쓰는 Native 보조 비교도 이 제한을 유지한다.
기존 결과는 `HISTORICAL`로 보존한다.
동일 Trace가 아닌 과거 Agg 곡선과 새 PD 수치의 Paired CI를 만들지 않는다.

## 7. 보존한 개발 이력

초기 Macro1 Adapter는 요청 간 Packing을 제한했다.
성능 측정 전에 이를 폐기했다.
Revision2는 Packing과 State 검사를 통과했다.
그러나 Layer만 사전 KV 예약을 사용했다.
Revision3는 이 계약을 통일했다.
세 Revision의 코드·설정·State Evidence를 보존한다.
폐기한232개 v1 성능 설정은 `NOT_RUN`이다.
이를 실행 실패나0% 개선으로 표시하지 않는다.

기존523개 Source 파일은 Parent Commit과 Byte 단위로 같았다.
기존 Checkout·Index·외부 Runtime 변경을 보존한다.
새 Inference Adapter의 Hash는 Revision3 Manifest로 동결했다.
실험 종료 시 승인된 Burner의0–7 GPU 복원을 별도로 검사한다.

## 증거 경로

- [Revision3 실행 계약](dense_ar_matched_axis_crossover_20261117/state/design_freeze_v3_before_performance.json)
- [새 Inference Source SHA256](dense_ar_matched_axis_crossover_20261117/state/frozen_new_inference_sources_v3.json)
- [N8 State Gate](dense_ar_matched_axis_crossover_20261117/analysis/qualification_reserved_N8.json)
- [N32 State Gate](dense_ar_matched_axis_crossover_20261117/analysis/qualification_reserved_N32.json)
- [N128 State Gate](dense_ar_matched_axis_crossover_20261117/analysis/qualification_reserved_N128.json)
- [Output200 N8 State Gate](dense_ar_matched_axis_crossover_20261117/analysis/qualification_reserved_Output200_N8.json)
- [실제 Chunk Geometry](dense_ar_matched_axis_crossover_20261117/analysis/chunked_source_initial_actual_geometry.json)
- [동일 Rows·Attention·Payload 계산](dense_ar_matched_axis_crossover_20261117/analysis/matched_axis_structural_work.json)
- [기존 Source 보존 감사](dense_ar_matched_axis_crossover_20261117/state/predecessor_vs_228a_source_audit.json)

Raw 대용량 파일은 종료 후 검증된 Lossless Archive로 보존한다.
실행별 Source ZIP·Config·Prompt Hash·Arrival Seed·GPU UUID를 함께 저장한다.

# Dense AR Unified-budget Matched-axis Crossover — 실행 계약

## 목적과 범위

Qwen3-8B의 고정 Layer Cut과 Completed-prefix Token Split을 비교한다.
모든 Worker에 B8192/S1024를 적용한다.
Dynamic 정책과 TP2 수정은 구현하지 않는다.
기존 결과는 HISTORICAL로 보존한다.

## 공통 조건

- Model Revision: `b968826d9c46dd6066d109eabc6255188de91218`.
- Original BF16, Exact Causal Attention, Full36 Decode, Greedy, Thinking Disabled.
- Input8000. Output16/32/64/128/200. `ignore_eos=True`.
- GPU0/1의 H100 SXM 80GB. GPU당 KV42GiB. Prefix Cache 비활성화.
- vLLM0.26.0+cu129, PyTorch2.11.0+cu129, NIXL1.4.1.
- Publication Parent: `228a041f35cedcee9f45beba6c462b453fb3a2c5`.
- Branch: `flashvep/dense-ar-matched-axis-crossover-20261117`.

## Admission과 작업량

모든 PD 방식의 D Admission 상한은8이다.
FIFO와 기존 Decode Cohort 종료 Guard를 공통으로 사용한다.
한 Cohort의 실제 Prompt Query 합은8192 이하여야 한다.
Layer의 Query는8000개다. Token의 Query는 지정한 Suffix 길이다.
Native의 Cached Query는1개다.
따라서 같은 Admission 상한에서도 실제 Cohort 크기는 다를 수 있다.
이 차이를 GEOMETRY_MATCHED로 표시하지 않는다.
공통 Admission 규칙과 자연적인 Shape 차이를 구분한다.

P Source Macro는1이다. Worker의 S는1024다.
이번 Prefix 길이는 모두4096보다 크므로 B8192에는 완전한 Prefix 한 개만 들어간다.
Partial Layer 경로에 검증되지 않은 Token Tiling을 추가하지 않는다.
측정 전에 모든 실제 Worker B/S를 검증한다.

정합성은 N8→N32→N128 순서로 검사한다.
성능의 정확한 작업량 비교는 N144를 사용한다.
N128에서는 세 Cut의 총 GEMM Rows를 정수 Suffix로 정확히 맞출 수 없다.
Last4/8/12의 인접16-token 경계를 FIFO 순서로 분배한다.
실제 P/D Query Coverage와 완성된 Migration을 다시 검사한다.
같은 GEMM Rows는 같은 Attention FLOPs 또는 GPU 시간을 뜻하지 않는다.

## 실행 순서와 판정

1. 실제 Worker 설정과 N8/N32/N128 State 정합성을 확인한다.
2. Output200 QPS2–10의 Native/Agg 및 세 Pair를 측정한다.
3. 나머지 Output의 QPS2/4/6/8/10 Discovery를 수행한다.
4. 중요 조건의 독립 Start를 최소3회 반복한다.
5. Host Critical Path와 GPU 계측의 간섭을 별도로 검증한다.
6. CI, 실패 기록, Figure와 보고서를 저장한다.
7. Commit/Push, Remote SHA 및 Burner 복원을 검증한다.

기존 Frozen SLO 파일의 Hash를 시작 전에 저장한다.
Output200 Balanced는 TTFT5000ms, 모든 ITL34ms, TPOT33ms다.
Client-visible Metric을 Primary로 사용한다.
95% CI의 재표본 단위는 독립 Server Start다.
Native-relative ITL p99와 TPOT p95 CI 상한이 모두1.05 이하여야 보호를 통과한다.
Finite-cohort 처리율을 Sustainable Capacity로 표시하지 않는다.

Profiler의 Clean 대비 간섭이5%를 넘으면 DIAGNOSTIC으로만 사용한다.
Host/CUDA Event Enclosure를 Exclusive Kernel Time으로 표시하지 않는다.
측정하지 못한 Counter는 NOT_MEASURED로 남긴다.
원인 분리가 불충분하면 ROOT_CAUSE_NOT_IDENTIFIED로 판정한다.

GPU Quiet Lane에서는 승인된 Burner만 중단한다.
CPU 작업 구간과 종료 시에는 승인 Burner를 복원한다.
기존 Checkout, Index, 외부 Runtime과 이전 Evidence를 보존한다.

## Revision2 — 성능 측정 전 Source Packing 검증

초기 Macro1 경로는 B/S와 State 정합성을 통과했다.
그러나 이 경로는 Native와 Token의 요청 간 Chunk Packing도 제한했다.
이를 Primary 비교에 사용하지 않는다. 초기 검증과 실패 기록은 보존한다.

Revision2는 모든 PD의 Source Macro 상한을8로 맞춘다.
동시에 Retained/Pending/Active Request Window32를 공통으로 적용한다.
Worker의 B8192/S1024와 표준 Chunked Prefill은 유지한다.
동일 Static Cut 안에서 Source Query는 원래 위치 순서로 계산한다.
Layer의 Hidden/Residual Fragment는 원래 Token 위치의 GPU Buffer에 모은다.
전체8000행이 완성되기 전에는 해당 State를 D에 공개하지 않는다.
기존 NIXL Read와 ACK 이후의 Source Lifetime 계약은 유지한다.
새 KV Manager나 Dynamic Cut 정책은 구현하지 않는다.

Chunked 경로는 별도의 N8→N32→N128 Gate를 통과해야 한다.
실제 Chunk 수, P/D Query Coverage와 완성된 전체 Prompt State를 검사한다.
초기 Atomic Gate와 새 Chunked Gate를 같은 결과로 표시하지 않는다.

기존 Output별 Frozen SLO를 변경하지 않는다.
Output Sweep의 보조 분석에는 공통5000/34/33ms Threshold도 사용한다.
이 보조 Threshold는 Clean Candidate 성능 측정 전에 동결했다.
기존 Output64 Balanced의 ITL35ms와 공통34ms를 구분한다.

## Revision3 — 성능 측정 전 KV 사전 예약 통일

Revision2의 N8/N32/N128 State 검사는 네 방식에서 모두 통과했다.
그러나 기존 Coordinator는 Layer에만 사전 KV 예약을 적용했다.
Native와 Token에는 같은 예약이 없었다.
이 차이가 있는 경로로 Clean 성능 Sweep을 실행하지 않는다.

Revision3는 Native/Layer/Token 모두 P 제출 전에 D KV Credit을 예약한다.
기존 Credit Ledger의 Cut 검사에36을 추가한다.
예약량은 전체 D Prompt와 Output의 Block 수다.
Commit, 할당 시 자기 Credit 제외, Request Mapping과 Retirement는 기존 코드를 유지한다.
예약 거절 또는 Commit 전 취소가 발생하면 같은 고정 Partition을 다시 기다린다.
이를 이유로 Native Partition으로 바꾸지 않는다.
새 KV Manager나 Dynamic Cut 정책은 추가하지 않는다.

Revision2의 Source와 State Evidence는 보존한다.
이미 생성한232개 미실행 v1 설정도 보존한다.
Primary Sweep은 별도의 v3 설정을 사용한다.
새 예약 계약의 N8→N32→N128 검사와 Output200 검사를 다시 수행한다.
SLO와 Partition 크기는 변경하지 않는다.

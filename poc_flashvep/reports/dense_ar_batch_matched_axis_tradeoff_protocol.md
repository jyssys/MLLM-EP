# Dense AR 배치 일치 Layer–Token 비교 계획

## 범위 변경

사용자는 2026-11-18에 실제 배치가 일치하는 Online 비교를 우선 요청했다.
작은 Cut의 Layer 이득과 큰 Cut의 Token 이득이 함께 존재하는지 검사한다.
역전은 검증할 가설이다. 역전이 나오는 결과만 채택하지 않는다.

기존 cap8 캠페인은 현재 GPU 작업이 정상 종료한 뒤 보류했다.
진행 중인 GPU 작업을 종료한 수는 0개이다.
기존 Discovery는 198/232개를 완료했다.
Admission 대조는 52/52개를 완료했다.
기존 120개 Confirmation은 새 실험으로 대체하지 않고 `NOT_RUN`으로 보존한다.
기존 Manifest와 완료·실패 기록은 변경하지 않는다.
보류된 조건을 완료로 표시하지 않는다.

전환 기록은 `dense_ar_matched_axis_crossover_20261117/state/user_batch_matched_scope_revision_20261118.json`에 보존한다.

## 고정 환경

- Model: `Qwen/Qwen3-8B`.
- Revision: `b968826d9c46dd6066d109eabc6255188de91218`.
- Original BF16, Exact Causal Attention, Greedy, Thinking Disabled.
- Decode: Full36 Layers.
- GPU: GPU0/1, H100 SXM 80GB 2개.
- vLLM: 0.26.0+cu129. PyTorch: 2.11.0+cu129. NIXL: 1.4.1.
- KV Budget: GPU당 42GiB. Prefix Cache: Disabled.
- 모든 Worker: B8192/S1024.
- Input: 8000 Tokens. Output: 16/64 Tokens. `ignore_eos=True`.
- Arrival: Open-loop Poisson. 비교 방법에 같은 계획 Arrival Trace를 적용한다.
- 기존 Numerical Correction과 8개 Frozen Inference Adapter를 유지한다.

GPU UUID는 기존 Environment 기록과 각 실행의 Provenance에서 검사한다.
Config, Dataset, SLO, Source ZIP의 SHA256을 보존한다.

## 실제 배치 계약

새 Protocol의 P Source Macro 상한은 1이다.
D Admission 상한도 1이다.
Config 상한만 검사하지 않는다.
실제 P Forward, D Prefill Forward, D Decode Forward의 요청 수가 모두 1인지 검사한다.
Source Group과 Destination Group도 요청 1개씩이어야 한다.
Decode Graph의 실제·Padding Query 수는 모두 1이어야 한다.

Source Outstanding Window는 기존과 같은 32이다.
FIFO, KV 사전 예약, ACK, Retirement와 Cohort Guard를 유지한다.
배치 수집·Queue 대기·Decode Residency를 Client-visible Metric에서 제외하지 않는다.
이전의 P Group8 결과와 새 P Group1 결과를 같은 조건으로 합치지 않는다.

이 Protocol은 배치 일치 원인 분리용 대조이다.
최적화된 자유 배치 Token 또는 Layer Baseline을 뜻하지 않는다.
배치1 제한으로 D Decode가 병목이 될 가능성도 검사한다.
이를 확인하지 않고 P Compute Saturation을 주장하지 않는다.

배치2/3은 이 Protocol에 구현하지 않는다.
B8192 아래에서 준비와 Decode 묶음을 분리하려면 별도의 실행·정합성 계약이 필요하다.

## 고정 Partition과 작업량

| Pair | Layer | Token Suffix | 요청별 D GEMM Rows 목표 |
|---|---:|---|---:|
| A | Last4 | 880/896 조합 | 평균 32000 |
| B | Last8 | 1776/1792 조합 | 평균 64000 |
| C | Last12 | 2656/2672 조합 | 평균 96000 |

N72와 N144에서 전체 GEMM Token-Layer Rows를 정확하게 맞춘다.
각 요청의 Suffix와 실제 Query Coverage를 보존한다.
GEMM Rows 일치는 Attention FLOPs 또는 실행시간 일치를 뜻하지 않는다.
Last2 전체 Sweep은 이전 취소 상태를 유지한다.

## 실행 Matrix

비교 Arm은 Native, Layer4, Token4, Layer8, Token8, Layer12, Token12이다.

| Stage | 요청 수 | Output | Offered QPS | 독립 Start |
|---|---:|---|---|---:|
| Discovery | 72 | 16 | 2/4/6/8/10 | 조건별 1회 |
| Discovery | 72 | 64 | 2/6/10 | 조건별 1회 |
| Confirmation | 144 | 16/64 | 6 | 조건별 3회 |

Discovery는 56개 Start이다.
Confirmation은 42개 Start이다.
성능 Matrix는 총 98개 Start이다.
별도로 N18 Full-state Gate를 7개 Arm에서 먼저 실행한다.

Discovery Seed는 2026111810이다.
Confirmation Seed는 2026111820/2026111821/2026111822이다.
Discovery와 Confirmation을 CI 계산에서 합치지 않는다.
실제 문서의 기존 `matched144.json` Pool을 사용한다.
새 Arrival Seed 검증이며, 완전히 새로운 Prompt Pool 검증은 아니다.

모든 설정을 새 성능 결과 전에 동결한다.
동결 파일은 `state/batch_matched_design_20261118.json`이다.
새 Raw 실행 이름은 `batch1_`로 시작한다.

## 정합성과 Metric

N18 Gate는 Greedy Output, Full Prompt KV, Hidden State, 실제 Layer 방문을 검사한다.
Partial State, Position, NIXL ACK와 Lifetime 계약도 유지한다.
성능 실행은 Output, 요청별 Query Coverage, 실제 작업량과 Credit Retirement를 검사한다.

주요 Metric은 Client-visible TTFT p95와 SLO-qualified Output Tokens/s이다.
Mean TTFT, Mean TBT, ITL p99, Maximum ITL, TPOT p95도 기록한다.
Raw Output Tokens/s와 Joint SLO Attainment를 별도로 보고한다.
Finite-cohort Drain Throughput을 Sustainable Capacity로 표시하지 않는다.

기존 Frozen SLO를 변경하지 않는다.
공통 보조 SLO는 TTFT 5000ms, 모든 ITL 34ms, TPOT 33ms이다.
Output별 원래 SLO와 공통 보조 SLO를 구분한다.
Native-relative Decode 기준은 ITL p99와 TPOT p95의 비율 상한 1.05이다.

## 성공 판정

작은 Cut에서 Layer/Token TTFT p95 비율이 1보다 작아야 한다.
큰 Cut에서 같은 비율이 1보다 커야 한다.
각 방향의 우위는 독립 Start 기반 CI로 검사한다.
기존 Type7 Bootstrap CI와 역CDF 민감도 CI를 모두 보고한다.
정의에 따라 경계 판정이 달라지면 강한 역전으로 채택하지 않는다.
3개 Start의 CI가 모집단 Coverage나 WCET 보장을 뜻하지 않는다.
여러 조건을 탐색하므로 Pointwise CI를 전역 유의성으로 해석하지 않는다.

TTFT 역전과 SLO-goodput 역전은 별도로 판정한다.
TTFT 우위를 Throughput 우위로 바꾸어 표현하지 않는다.
최종 Serving 우위에는 Native-relative Decode 보호 결과도 함께 제시한다.
SLO Threshold를 결과에 맞춰 바꾸지 않는다.

## 원인 분석

다음 증거를 구분한다.

- `MEASURED`: 실제 Online Latency, Output Tokens/s와 완료된 Query Geometry.
- `TRACE_ANALYSIS`: Host 경계별 TTFT, Admission 대기와 Decode Residency.
- `STRUCTURAL_ACCOUNTING`: GEMM Rows, Attention Pair와 Projection 호출 수.
- `DIAGNOSTIC`: 계측 간섭 Gate를 통과하지 못한 Event 또는 Profiler 결과.
- `NOT_MEASURED`: Clean-qualified Exclusive GEMM/Attention 시간, SM/HBM Counter가 없는 항목.

기존 고립 GEMM 측정은 실제 Shape에 대한 비용 모델 입력으로만 사용한다.
대표 Layer의 시간을 Layer 수만큼 곱한 값을 Online GEMM 실측으로 표시하지 않는다.
Host 경과시간을 Exclusive GPU Compute Time으로 표시하지 않는다.
성분별 p95를 더하거나 빼서 TTFT p95의 원인 비율로 표시하지 않는다.

배치 일치 뒤에도 차이가 작으면 해당 결과를 그대로 보고한다.
역전이 없거나 원인이 분리되지 않으면 그 한계를 남긴다.

## 자원과 보존

Primary Timing은 기존 Quiet-lane Guard를 사용한다.
실험이 없을 때와 최종 종료 뒤에는 승인된 GPU0~7 Burner를 복원한다.
사용자 Checkout, Staged Index와 외부 Runtime을 보존한다.
실패한 실행은 삭제하지 않는다.
코드·Raw Evidence·Figure·보고서를 Commit/Push한 뒤 Remote SHA를 검사한다.

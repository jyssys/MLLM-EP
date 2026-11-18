# 고정 Cohort2의 실행 계약과 조건부 Online 계획

## 목적과 범위

배치1에서는 앞선 요청의 Decode Residency가 다음 Prefill의 TTFT에 들어간다.
배치1 결과만으로 큰 행렬의 계산 이득을 일반화하지 않는다.
준비 배치와 Decode 배치를 분리한 고정 Cohort2를 별도 대조로 검사한다.

두 분할 축에서 P 준비 배치는 1이다.
D Suffix 준비 배치도 1이다.
그 뒤의 Full36 Decode 배치는 2이다.
Token만 여러 요청의 Suffix GEMM을 합치지 않는다.
모든 실제 Scheduler Iteration의 Token Budget은 8192 이하이다.
모든 Worker 설정은 B8192/S1024이다.

새 Dynamic Cut Scheduler는 없다.
요청의 Cut과 Token Suffix는 실행 전에 고정한다.
기존 배치1 Matrix와 새 Cohort2 결과는 다른 Stage로 보존한다.

## Model과 환경

Model은 `Qwen/Qwen3-8B`이다.
Revision은 `b968826d9c46dd6066d109eabc6255188de91218`이다.
Original BF16, Exact Causal Attention, Greedy와 Full36 Decode를 유지한다.
GPU0/1의 H100 SXM 80GB 2개를 사용한다.
vLLM 0.26.0+cu129, PyTorch 2.11.0+cu129, NIXL 1.4.1을 사용한다.
KV Budget은 GPU당 42GiB이다. Prefix Cache는 끈다.
기존 Numerical Correction과 8개 Frozen Inference 파일을 변경하지 않는다.

## 준비와 첫 토큰의 계약

P에서 실제 준비된 요청 2개가 있어야 새 Cohort를 수락한다.
FIFO 순서를 유지한다.
Layer와 Token은 요청 1개씩 Suffix를 실행한다.
모든 Cohort Member의 로컬 Suffix가 제출된 뒤 Decode를 허용한다.
이미 제출한 Kernel을 선점한다고 가정하지 않는다.

첫 토큰 알림은 Cohort 전체의 첫 알림이 존재할 때 전달한다.
그 알림을 기다린 시간은 Client-visible TTFT에 포함한다.
원래 Generation Timestamp와 Token Count는 변경하지 않는다.
이 Holding을 무료 계산 여유 또는 GPU Idle로 표시하지 않는다.
Worker의 첫 알림부터 마지막 알림까지의 경과시간에는 남은 준비가 포함될 수 있다.
그 경과시간을 Exclusive Decode GPU Time으로 표시하지 않는다.

Native는 P의 원래 토큰을 재사용하는 기존 계약을 유지한다.
Cached Query의 초기 Input Flag가 prefilling이어도 전체 Prompt를 재실행한 것은 아니다.
Native의 초기 Query와 후속 Full36 Query는 실제 요청 배치2로 실행한다.
Native 첫 알림 2개는 같은 초기 Forward에서 나온다.
Native 토큰을 인위적으로 교체하거나 전달 간격을 조작하지 않는다.

## 보존한 실패

### Version1

Native N18의 State와 실제 Decode2 Geometry는 통과했다.
Layer Last4는 다음 입력을 제출했다.

- q0: Decode Query 1개.
- q1: Suffix Query 8000개.

기존 Partial Prefill 안전 검사가 혼합 입력을 거절했다.
실행은 `FAILED`이다.
Greedy Output 또는 Full-state 성능 Gate의 통과로 사용하지 않는다.
Computed Hint와 Request Metadata에 의존한 준비 판정은 이 혼합을 막지 못했다.
실패 시점의 모든 내부 Counter 값을 계측한 것은 아니다.

### Version2

Version2는 실제 로컬 Suffix의 Scheduled-token Receipt를 추적한다.
Native와 Layer Last4의 State Gate는 통과했다.
실제 P 준비1, Layer Suffix 준비1, Decode2 Geometry도 통과했다.

Token Last4에서는 Client 등록이 사라졌다.
기존 Token `PrefixChannel`은 `ar_worker.Client`에서 새 Client를 만든다.
Version2는 다른 `worker.Client`만 감쌌다.
새 Frontier는 등록되지 않은 q0을 거절했다.
이 실행도 `FAILED`이며 성능으로 사용하지 않는다.

두 실패의 Receipt와 Config SHA256을 별도로 보존한다.
정확히 일치하는 실패만 배치1과 분리한다.
새로운 실패나 배치1 오류를 일괄적으로 무시하지 않는다.
실패 후 GPU Process와 Resource Guard도 검사한다.

## Version3 검사

Version3는 Token Installer가 사용하는 Client Alias를 보존한다.
Source Prefix Channel의 동작은 변경하지 않는다.
Version2의 로컬 제출 Frontier와 Model Forward도 변경하지 않는다.
Version1과 Version2 파일은 보존한다.
Version3는 새 실행 이름과 Source ZIP을 사용한다.

Token Client Alias 회귀 검사는 실패 상태에서 실행한 뒤 수정했다.
수정한 CPU 검사는 같은 등록 Client가 PrefixChannel 안에 남는지 검사한다.
CPU 검사만으로 GPU 정합성을 통과했다고 표시하지 않는다.

N18 State 검사는 Native와 Last4/8/12의 Layer/Token, 총 7개 Arm이다.
Output, Full Prompt KV, Hidden, 실제 Layer 방문과 Lifetime을 검사한다.
실제 준비·Decode 배치, Padding Query와 First-token Barrier도 별도로 검사한다.
모든 검사 전에 Source Adapter Hash를 동결한다.
7개 State/Geometry Gate가 모두 통과해야 성능을 시작한다.

## 조건부 성능 Matrix

Input은 8000 Tokens이고 Output은 16 Tokens이다.
`ignore_eos=True`를 유지한다.
Arm은 Native, Layer4, Token4, Layer8, Token8, Layer12, Token12이다.

| Stage | N | QPS | Seed | Start 수 |
|---|---:|---|---|---:|
| Discovery | 72 | 2/6/10 | 2026111810 | 21 |
| Confirmation | 144 | 6 | 2026111820/2026111821/2026111822 | 21 |

총 42개 성능 Start를 조건부로 선언한다.
정합성 실패 시 해당 성능은 `NOT_RUN`으로 남긴다.
배치1과 같은 Prompt Pool 및 계획 Arrival Trace를 사용한다.
이는 새 Prompt Pool의 독립 검증이 아니다.
배치1과 배치2의 Start를 같은 CI에 합치지 않는다.
N72와 N144도 같은 Curve나 CI에 합치지 않는다.

기존 Frozen SLO를 유지한다.
Output16 Balanced SLO는 TTFT 5000ms, 모든 ITL 34ms, TPOT 33ms이다.
Native-relative Decode 기준은 ITL p99와 TPOT p95의 비율 상한 1.05이다.
독립 Start 기반 CI 상한으로 검사한다.
Type7과 역CDF 민감도 CI를 함께 보고한다.

TTFT, SLO-goodput과 Raw Output Tokens/s를 별도로 판정한다.
작은 Cut의 Layer 우위나 큰 Cut의 Token 우위를 가정하지 않는다.
Finite-cohort Throughput을 Sustainable Capacity로 표시하지 않는다.

## Evidence 경로

- 실패 원본: `raw/fixed_cohort2_state_*_v1/`, `raw/fixed_cohort2_state_*_v2/`.
- 정확한 실패 기록: `state/preserved_fixed_cohort_prototype_failures.json`.
- Version3 계획: `state/fixed_cohort2_v3_qualification_design.json`.
- 실제 Geometry 검사: `fixed_cohort_audit --version 3`.
- 조건부 성능 계획: `state/cohort2_online_design.json`.
- 실제 성능 완료 여부: `state/cohort2_online_progress.json`, `state/cohort2_online_complete.json`.

모든 상대 경로의 기준은 `dense_ar_matched_axis_crossover_20261117/`이다.
계획한 조건과 완료한 조건은 별도로 보고한다.

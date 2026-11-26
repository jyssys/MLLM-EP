# H200 Dynamic Split Active C3 Online Serving

실제 Online 조건은 E/P/D H200 3장, B8192/S1024, KV42GiB, D Cap8과 QPS5다. Main100 Natural EOS와 동결 5,269-token Replay를 따로 평가한다. 실용 비교에 NativeStrong 및 NativeCommon의 Free를 유지하고, 수치 모드와 inline 경로를 맞춘 Native와도 비교한다.

## 실제 C3의 판정 기준

완료된 Main100, Client와 Coordinator의 유효 Timestamp 및 동일 Arrival Digest를 확인한다. 실제 D를 사용한 Request는 선택 Callback, 원자적 예약, KV/EC Handoff, D Suffix Query 및 결과가 연결되어야 한다. Shadow의 fit 수와 정합성 전용 N8의 offload는 C3 실행 수와 별도로 집계한다.

| 검증 단계 | 보고 항목 |
|---|---|
| 정합성 | 명시한 Numerical Regime과 Transport의 Exact Greedy, 원본 상태와 Full28, 실제 Mixed Decode |
| 실제 실행 | 요청 및 Token 기준 Offload, P Continuation, Prefix-ready D 상태와 예약·대기 |
| Batch | Cap, Scheduler와 Prepared Query Geometry, 한 요청 Forward 비율, Token-weighted Batch |
| 성능 | TTFT p50/p95, ITL p99, TPOT p95, absolute Decode SLO, Joint SLO, Goodput |
| 비교 | 동일 시작 수의 독립 짝, 고정 Arrival과 Workload, BI1 matched 및 BI0 Native Free |

TPOT는 요청별 Output 간격 평균을 사용한다. Absolute Decode SLO는 각 요청의 모든 Client ITL ≤34 ms와 TPOT ≤33 ms를 함께 확인한다. TTFT와 Decode가 함께 통과한 요청만 Joint SLO에 포함한다. C3 성능의 독립 반복은 정합성 Probe와 별도로 계산한다. [정합성 검증](mllm_epd_h200_dynamic_split_correctness_recovery.md) · [연구 계약](mllm_epd_h200_dynamic_correctness_slice_aware_poc_spec.md)

## 초기 BI1 성능 진단

QPS5 Natural Main100에서 P/D BI1 NativeCommon-B8은 100건을 완료했다. 첫 Cohort의 TTFT p50/p95는 6.400/17.732 s, ITL p99는 31.881 ms, TPOT p95는 22.677 ms였다. 26건에 34 ms를 넘는 Client ITL이 있어 absolute Decode SLO는 74%였다. 이 실행에는 Stock Scheduler Observer가 있었고, 별도 Batch Counter를 켠 두 번째 Cohort의 Decode SLO는 77%였다. 따라서 첫 Exact Correctness 복구 경로의 성능도 독립적으로 문제가 있음을 확인했다. 두 Cohort의 새 Stock Trace는 flush 결함으로 유실되었다.

## Method A의 첫 Main100 실행

BI1 Slice-aware Late10은 QPS5에서 예약을 실제로 허용한 상태로 Main100을 완료했다. 첫 Cohort에서는 12개 Callback이 모두 `D_ADMISSION_BUSY`였고 D Suffix가 0개였다. 별도 Batch Counter Cohort에서는 같은 이유로 18개 Callback이 모두 거절됐다. 이 판정은 Coordinator의 Active HTTP 수명을 관측한다. 두 Cohort에서 Callback마다 한 번 retirement hold를 썼고 재계산은 0건이었다. Worker 측 Callback RPC 중앙값은 각각 1.962 ms와 1.617 ms였다.

첫 Cohort의 TTFT p50/p95는 7.331/19.448 s, ITL p99는 33.839 ms, TPOT p95는 22.564 ms였다. Absolute Decode SLO는 67%였다. 이 성능은 동작 중인 후보 경로의 0-offload 결과다. Suffix 연산을 실제로 이전해 얻는 TTFT 또는 Decode 효과는 이 Cohort에서 산출할 수 없다.

## 실제 Suffix를 이전한 Eager Slice Pilot

Eager 5%와 10% 경로가 각각 N1 및 실제 Mixed N8의 기존 Exact 기준을 통과했다. Same-request continuation은 같은 Eager Native EPD와 요청·P/D Query Geometry·원본 E·Full28·Greedy가 모두 일치했다. 별도로 E+P 단독 Teacher와의 이전 비교는 Greedy 60→56으로 미달 상태를 유지했다. 아래 실행의 범위는 정합성을 확인한 EPD 경로의 Pilot이다. [Eager 정합성의 범위](mllm_epd_h200_dynamic_split_correctness_recovery.md)

| Main100 Natural, B8 | Actual suffixes / tokens | TTFT p50 / p95 (s) | ITL p99 (ms) | TPOT p95 (ms) | Decode SLO | Joint SLO | Goodput (tok/s) |
|---|---:|---:|---:|---:|---:|---:|---:|
| Eager NativeCommon Stock Observer, QPS5 | 0 / 0 | 4.282 / 13.144 | 26.062 | 13.532 | 78% | 40% | 40.949 |
| Eager Slice Late10, QPS5 | 1 / 2,742 | 3.625 / 12.116 | 26.341 | 14.665 | 87% | 50% | 49.802 |
| Eager Slice Late10, QPS4 | 2 / 4,742 | 3.341 / 11.248 | 25.954 | 14.382 | 90% | 56% | 75.420 |

위 Pilot은 독립 Server Start 한 번으로 각 Load의 100건을 완료했다. 별도 Batch Counter Cohort는 QPS5에서 3개 Suffix를 실행했고 QPS4에서 0개였다. Main100 Natural 출력은 QPS5 Native 및 Late10에서 서로 달랐으며, 표의 차이에는 독립 Paired CI가 없다.

실제 Suffix를 실행한 두 Load의 여섯 Event에서 예약 당시 D Engine의 Active/Pending은 모두 0, KV free blocks는 6,143이었다. D Coordinator Wait는 모두 1 ms 미만이었다. 각 Suffix의 이미지 한 장은 P가 원본 inline Descriptor를 이미 사용한 crossing 이미지였다. D가 해당 한 장의 새 Descriptor를 E Engine에서 RPC로 받았고, 나머지는 P의 미사용 inline Descriptor로 진행했다. QPS5 주요 실행의 D submit-to-first는 1,993.381 ms였다. E Descriptor Export RPC가 1,731.800 ms, D import/clone/sync Host Enclosure가 55.596 ms였다. 측정한 최장 Export RPC는 2,470.676 ms였다. [실제 Suffix의 Host Timing](../mllm_epd_h200_dynamic_correctness_slice_aware/results/active_crossing_export.csv)

## 사전 발급과 Inline Readiness를 적용한 BI1 Pilot

BI1에서 사전 발급을 적용한 10/20%의 N1, Mixed N8와 P Continuation이 기존 정합성 기준을 통과했다. Active Late10의 허용 비율 구간은 7.5–15%였다. QPS3은 BI1의 D 용량 변화에 따른 추가 민감도다. [사전 발급 정합성 및 회수](mllm_epd_h200_slice_aware_embedding_transfer.md)

| Main100 Natural, B8, 첫 Cohort | Actual suffixes / tokens | TTFT p50 / p95 (s) | ITL p99 (ms) | TPOT p95 (ms) | Decode SLO | Joint SLO |
|---|---:|---:|---:|---:|---:|---:|
| QPS5 | 0 / 0 | 7.185 / 18.564 | 30.755 | 22.557 | 80% | 22% |
| QPS4 | 0 / 0 | 3.553 / 13.582 | 30.191 | 22.594 | 82% | 46% |
| QPS3 | 1 / 4,032 | 2.947 / 10.534 | 28.569 | 22.414 | 88% | 56% |

별도 Queue/Batch Counter에서 QPS4와 QPS3 각각 1건을 추가 이전해, 실제 세 Suffix의 D submit-to-first는 221–271 ms였다. 세 건 모두 crossing Descriptor 사전 발급을 사용했고 Export RPC는 0 ms였다. 모두 중앙 Arrival 관측 구간 밖에서 발생했다. QPS3의 두 이전은 이후 준비된 Native 요청을 각각 최대 약 79.26 ms, 24.48 ms 기다리게 했다. 세 이전 요청 자체는 Decode SLO를 통과했다.

별도 Counter의 논리 Prepared Query를 요청별로 역추적했다. QPS4의 실제 D Suffix Query는 정확히 3,068개였고 해당 요청의 Decode 39 step 중 8개에서 Native와 함께 실행했다. QPS3은 2,842개와 25 step 중 19개였다. 두 Cohort에서 실제 Decode Batch 최대는 6이었다. 이 Counter는 독립 Qualification이 없는 `DIAGNOSTIC`이다. [실제 D Query 및 Co-batch](../mllm_epd_h200_dynamic_correctness_slice_aware/results/active_decode_geometry.json) · [예약당 Host Timing](../mllm_epd_h200_dynamic_correctness_slice_aware/results/active_crossing_export.csv)

같은 BI1, Preissue 및 Arrival Seed의 Native-Matched Stock Observer에서는 QPS3/4/5의 TTFT p95가 각각 13.065/14.809/17.215 s, absolute Decode SLO가 93/76/75%였다. QPS3의 Late10 점추정은 TTFT p95 −19.4%였고 Decode SLO는 5 pp 낮았다. QPS4의 두 첫 Cohort가 모두 0-offload였는데도 TTFT p95는 8.3% 달랐다. 관측 설정과 시작 간 변동을 포함한 N1 비교에서 QPS3 차이를 한 번의 이전 효과로 귀속할 근거가 부족하다.

## B-new의 실제 Dynamic Suffix

사전 발급 BI1의 정합성을 통과한 10/20% 범위, QPS4 Discovery로 학습한 Causal P-ready, 실제 같은 BI1의 isolated Cost, Adaptive Budget, Point Gain과 10 ms 안전 Margin을 동결했다. Inline Descriptor가 준비된 경우에만 D 예약을 허용했다. 이 런타임의 E/P/D는 실제 C3 Serving을 수행했다.

| Main100 Natural, B8, 첫 Cohort | Actual suffixes / tokens | TTFT p50 / p95 (s) | ITL p99 (ms) | TPOT p95 (ms) | Decode SLO | Joint SLO | Goodput (tok/s) |
|---|---:|---:|---:|---:|---:|---:|---:|
| B-new, QPS5 | 1 / 1,482 | 7.845 / 20.317 | 31.941 | 22.655 | 75% | 12% | 8.060 |
| B-new, QPS4 | 2 / 3,234 | 3.673 / 14.135 | 31.178 | 23.044 | 80% | 47% | 45.095 |
| B-new, QPS3 | 2 / 3,234 | 3.103 / 13.037 | 28.694 | 22.618 | 88% | 59% | 49.874 |

별도 Counter Cohort를 포함한 실제 Suffix는 12건이었다. 중앙 Arrival 구간의 Suffix는 QPS4 Counter 1건이었다. 12건 모두 crossing Descriptor를 사전에 받아 Export RPC가 없었다. QPS3 Counter 1건은 실제 Cut-to-first 311.16 ms로 예측 상한 228.51 ms와 Budget 244.56 ms를 넘었다. 해당 요청의 Clone/Sync Host Enclosure는 188.62 ms였다. 12건 중 이전 요청 자체의 Decode SLO를 통과한 수는 11건이다. QPS5 첫 Cohort에서 이전한 1,482-token 요청은 Client ITL 최대 38.32 ms로 실패했다. [Cut별 예측과 결과](../mllm_epd_h200_dynamic_correctness_slice_aware/results/slack/bnew_bi1_preissue_natural/) · [실제 D 전송 및 Client ITL](../mllm_epd_h200_dynamic_correctness_slice_aware/results/active_crossing_export.csv)

Counter에서 QPS5 Native-Matched와 B-new의 Step-weighted Prepared Decode Batch는 2.648과 2.640이었다. QPS4/3 B-new의 실제 Split Decode step 52/52개 중 Native와 공동 실행한 수는 29/31개였다. QPS5 Counter의 Split Decode 10 step은 단독 실행이었다. 모든 Counter에서 실제 D에 준비한 Prompt Row 수는 각 이전 Suffix와 일치했다. [실제 D Batch와 Query Coverage](../mllm_epd_h200_dynamic_correctness_slice_aware/results/active_decode_geometry.json)

동일 설정의 Native-Matched Stock Observer 대비 첫 Cohort TTFT p95 차이는 QPS5 +18.0%, QPS4 −4.6%, QPS3 −0.2%였다. 전 구간에서 B-new의 absolute Decode SLO는 요구된 95%보다 낮았다. 이 비교는 Seed와 Transport를 맞춘 N1이며 Stock Observer와 시작 차이가 남는다.

## Equal-output Replay와 로컬 Task Quality

별도 시작의 QPS5 Equal-output Replay에서 B-new는 Main100 100건과 동결 출력 5,269개를 완료했다. 첫 Cohort의 실제 이전은 2건, 4,237 tokens였으며 TTFT p50/p95는 6.185/18.035 s, ITL p99는 32.998 ms, TPOT p95는 22.867 ms였다. Absolute Decode와 Joint SLO는 73%와 28%, Goodput은 16.952 tok/s였다. Counter Cohort에서는 실제 이전이 1건이었다. Natural EOS의 Cohort와 Replay의 독립 시작은 서로 다른 결과로 집계한다.

같은 QPS5 Seed, Main100 및 5,269-token 계약의 NativeStrong-Free BI0 Clean Before/After는 TTFT p95 15.849/12.332 s, ITL p99 11.829/12.178 ms, TPOT p95 5.122/5.224 ms, absolute Decode SLO 98/95%였다. B-new는 두 Clean Cohort 모두보다 TTFT가 높았고 Decode 목표를 충족하지 못했다. 이 두 Native Cohort는 한 서버 시작의 Observer bracket이다.

NativeStrong-Free의 Collector 수집은 통과했다. 5% Clean/Profile/Clean Qualification은 Clean TTFT p95 spread 24.96%와 Profile ITL p99 deviation 13.24% 등의 항목에서 실패했다. 같은 Capture의 E/P/D Kernel Busy는 90.79/94.21/60.26%였으며 Evidence는 `DIAGNOSTIC`이다. [NativeStrong Replay 및 Observer 검증](../mllm_epd_h200_dynamic_correctness_slice_aware/results/native_strong_replay_profile.json)

동일 QPS5 Natural Seed의 B-new와 Native-Matched를 기존 로컬 Scorer로 비교했다. 전체 Greedy Token ID는 99/100이 일치했고, Mantis-Eval, DocVQA, MP-DocVQA의 점수 차이는 각각 0 pp였다. Hard와 B-new의 해당 Token ID는 100/100이 일치했다. MMLongBench-Doc의 로컬 RAW 점수는 별도 진단이다. 공식 Answer Extraction, 새 Context-disjoint Heldout 및 독립 시작의 품질 비열등 CI는 미실행이다. [요청별 품질](../mllm_epd_h200_dynamic_correctness_slice_aware/results/quality/dynamic_quality_requests.csv) · [Scorer와 품질 범위](../mllm_epd_h200_dynamic_correctness_slice_aware/results/quality/dynamic_quality_receipt.json)

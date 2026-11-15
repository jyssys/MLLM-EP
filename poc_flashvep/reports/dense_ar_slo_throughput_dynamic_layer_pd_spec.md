# SLO-constrained throughput and deadline-aware Layer PD

## Objective and immutable contracts

Compare sustainable SLO-qualified output throughput with SLO-specific Continuous Colocated DP2 frontiers.
Do not treat a finite-cohort drain rate as sustainable capacity.
Separate a Native-relative benefit from an independent Dynamic contribution.

Use Qwen3-8B revision `b968826d9c46dd6066d109eabc6255188de91218`.
Keep original BF16, exact Attention, Full36 Decode, greedy outputs, and disabled thinking.
Retain the qualified phase-specific RMSNorm correction in every arm.
Keep GPU0/1 and 42 GiB KV per GPU for primary comparisons.
Disable prefix caching.
Preserve the old source, evidence, checkout, staged index, and external runtime.

Publication parent: `92e5275f1175bc08680de77180b8f5c652f2d08f`.
Branch: `flashvep/dense-ar-slo-throughput-dynamic-layer-pd-20261114`.
Source namespace: `poc_flashvep/dense_ar_slo_layer_pd/`.
Evidence namespace: `dense_ar_slo_throughput_dynamic_layer_pd_20261114/`.

## Primary metric and fixed SLO

Primary research target: 95% request-level joint SLO attainment.
The 99% target is a separately reported sensitivity analysis.
Neither target is a production standard.
Retain the existing TTFT-priority, Balanced, and Decode-priority thresholds.
Frozen SLO SHA256: `c8aa94213e4f4ce54821d01ba985a8e9e88a20566a3e8d1f3cc8060d1b3926d9`.
Do not change thresholds after candidate results.

Evaluate raw completed-output tokens/s and completed requests/s in a fixed completion window.
Count every completion in that window, including requests that arrived during the preceding warm-up.
Also report tokens actually emitted inside the window as a boundary diagnostic.
Do not conflate these two definitions.

Calculate SLO-qualified tokens/s from completed requests that satisfy all request-level SLO conditions.
Independently follow every request that arrives during the measurement window to completion.
Include late and unfinished measurement arrivals in cohort SLO attainment.
An unresolved request cannot silently disappear from the denominator.
Keep first ITL, steady ITL, maximum ITL, and threshold counts separate.
Service TTFT remains NOT_MEASURED unless its actual boundary exists.

## Sustainable operating-point screen

Separate warm-up, fixed measurement, follow-through, and final drain.
Initial duration pilots use 60 s warm-up and 240 s measurement.
Confirmation uses at least 600 s measurement when a capacity boundary is promising.
Continue arrivals for 60 s after the measurement window.
This guard interval prevents an artificial empty-server drain from improving the last measurement cohort.
Then stop arrivals and retain all remaining completions or bounded failures.

실행 프로토콜 보완: 현재 600초 탐색과 독립 확인은 후속 도착 구간을 90초로 설정한다.
이는 최초 240초 Pilot의 60초 후속 구간과 다르다.
실험별 Immutable Configuration의 `follow_seconds`를 최종 근거로 사용한다.
독립 확인에서는 모든 방법에 같은 90초를 적용한다.

Require at least 95% joint attainment in both completion-window and measurement-arrival cohorts.
Require completed rate at least 98% of realized arrival rate inside the measurement window.
Record both configured QPS and realized Poisson rate.
Check P-ready, D-ready, all outstanding requests, and KV/Hidden credits over time.
Use 30 s blocks for queue-trend uncertainty.
Require outstanding-queue slope at most 0.01 requests/s.
Require its empirical upper estimate at most 0.025 requests/s.
Require end outstanding at most start outstanding plus max(4,1% of measurement arrivals).
Extend a near-threshold window rather than silently relax these tests.

These finite tolerances define an operational screen, not a mathematical proof of infinite-horizon stability.
Report the largest qualified tested point and its rejected or unconfirmed upper neighbor.
State QPS bracket resolution.
Do not claim an exact global maximum from a sparse search.
Use independent server starts for final uncertainty, not token or request resampling.
Retain both Native-relative Decode ratio upper bounds at most 1.05.

## Ordered research stages

1. Reproduce the exact Native and Continuous paths in the new namespace.
2. Tune Continuous separately for Balanced, Decode-priority, and raw throughput on viewed prompts.
3. Replay long-output zero-migration traces and attribute overlapping candidate gates.
4. Diagnose Source KV retention and the captured delivery-lag spike with bounded diagnostics.
5. Implement a direct affected-Decode deadline predicate and a separately tested reservation path.
6. Calibrate length, batch, transfer, control, and completion uncertainty before fresh evaluation.
7. Qualify Last0/2/4/8/12 and mixed request lengths before heterogeneous serving.
8. Search mixed-input, mixed-output, and bursty conditions on discovery traces.
9. Freeze settings and capacity candidates before held-out confirmation.
10. Compare Native, SLO-specific Continuous, Strong Static, and Dynamic under the same service objective.

Every proposed deadline must belong to work that the candidate can actually delay.
Do not reject a future suffix solely because an already completed cohort had an earlier deadline.
Do not log a deadline and then claim that the predicate enforces it.
Use an assertion-capable CPU regression test for the direct deadline rule.
Record affected request identity, ready estimate, deadline, and predicted delay for every candidate.

Keep own-request benefit and aggregate queue benefit as distinct policy modes.
An aggregate-benefit mode must demonstrate actual service goodput rather than assumed future improvement.
Do not force Last2/4 to produce a favorable histogram.
Do not weaken Decode guards without a separate correctness and interference qualification.

## Fair selection and controls

Tune Continuous chunk budget, sequence limit, routing, and legal execution options on discovery data.
Select C-Balanced, C-Decode, and C-Raw independently.
Qualify every selected numerical path.
Compiled or backend output failures are not eligible performance results.

Tune Static cuts and fractions on the same workload and objective.
Freeze future confirmation prompts, arrival seeds, method settings, and predictor bounds.
An exact-work control must audit realized request counts, layer work, and batch geometry separately.
If geometry does not match, retain that confound in the verdict.
Keep Controller Zero and predecessor policies as diagnostics.

## Measurement and limits

Roofline supplies a structural lower bound only.
Admission uses measured cost bounds and residual uncertainty from calibration.
Separate GEMM, Attention, KV, pointwise work, graph/eager mode, host submission, handoff, and output delivery where observable.
Mark unavailable exclusive component times NOT_MEASURED.
Do not add overlapping enclosures into a claimed critical-path saving.
Use marker-free latency for primary serving.
Keep detailed instrumentation in matched diagnostics with interference checks.
NVML utilization is not kernel busy, physical idle, or a safe budget.

Use real LongBench/ShareGPT documents with recorded IDs, token hashes, and truncation rules.
Do not repeat filler tokens.
A long duration run may cycle a declared prompt pool with prefix caching disabled.
Record distinct IDs and do not claim independent prompts from repeated requests.
Natural EOS requires separate reference and metric qualification.

## Operation and stopping

Use bounded per-job startup, execution, and process-group watchdogs.
Use the shared quiet-lane lease for primary timing.
Stop only verified approved burners.
Restore all eight approved burners during idle periods and after GPU work.
Check actual ownership and fresh completed generation at closeout.
Keep the missing-GPU-only restoration retry.

The initial planning ceiling is 18.5 h before a closeout reserve of 1.5 h.
This is not a required duration or a reason to repeat an exhausted hypothesis.
Review progress at major phase boundaries.
Stop failed expansions safely and preserve every failure.
Do not spend the campaign on a profiler that cannot meet its interference gate.

## Deliverables

Write the requested final report and seven auxiliary reports under `poc_flashvep/reports/`.
Include QPS/latency, raw throughput, qualified throughput, sustainable capacity, cut decisions, heterogeneity, and Pareto figures.
Distinguish MEASURED, TRACE_ANALYSIS, CORRECTNESS_ONLY, OFFLINE_ORACLE, STRUCTURAL_ACCOUNTING, and unexecuted or failed work.
Publish only this campaign through a private Git index.
Check Remote SHA and unchanged user state after push.
Keep approved burners active after the final response.

## 실행 중 검토 기록 — 2026-11-15 KST

SLO Threshold와 95% Joint Target은 변경하지 않는다.
실행 중 발견한 구현 문제 때문에 초기 시간 계획을 다시 검토한다.
총 계획 상한을 시작 후 21.5시간으로 변경한다.
새 GPU 측정은 시작 후 20시간 이전에 종료한다.
마지막 1.5시간은 분석, 보존, Push 및 GPU 복원에 배정한다.
이 시간은 사용 의무가 아니다.

다음 세 문제가 추가 검증의 근거다.

1. Strong Colocated의 정확한 Full Mixed Graph 경로를 확보하는 데 별도 수치 정합성 검증이 필요했다.
2. 지난 TTFT Deadline을 계속 Admission Gate로 사용하는 정책에서 Native Fallback이 누적됐다.
3. 큰 Batch에서 실제 NIXL Descriptor 수가 다시 12,000~18,000개로 증가했다.

`QUEUE_RECOVERY`는 의사결정 시점에 이미 지난 TTFT Deadline만 구분한다.
해당 요청은 기존 SLO 평가에서 계속 실패로 집계한다.
ITL과 아직 지나지 않은 TTFT Deadline은 기존 Predicate를 유지한다.
요청을 버리거나 처음부터 다시 계산하지 않는다.

선택적 KV 배치 변경은 새로 할당한 Block의 반환 순서만 정렬한다.
원래 Allocator가 선택한 Block 집합과 Reference Count는 유지한다.
기존 요청의 Logical Block Table과 Free Queue 연결은 변경하지 않는다.
Prefix Caching은 비활성 상태를 유지한다.
N32 전체 State 검증과 N8 D8 검증을 먼저 수행한다.
그다음 A/B/A에서 Descriptor 감소와 Handoff 변화를 확인한다.
이 변경은 Generic PD 최적화다.
Dynamic 정책의 독립적인 이득으로 집계하지 않는다.

기본 UCX Probe는 `cuda_ipc` Zero-copy 경로를 기록했다.
GPU0/1의 NV18 연결만으로 경로를 추정하지 않는다.
아직 새로운 Transport 설정은 적용하지 않는다.
Probe Logging 결과를 Clean Serving 성능으로 사용하지 않는다.

새 Runtime으로 대체한 이전 탐색은 `NOT_RUN_SUPERSEDED`로 구분한다.
이미 실행한 결과와 실패 로그는 보존한다.
종료 대상은 대기 중인 소유 Controller로 제한한다.
실행 중인 GPU Worker는 해당 정리 과정에서 종료하지 않는다.

## 후속 판정 기준 — 2026-11-15 05:15 KST

Streaming `QUEUE_RECOVERY`의 QPS 4.8 실행은 완료됐다.
600초 구간의 실제 도착률은 4.805 requests/s였다.
완료율은 4.4533 requests/s였다.
Outstanding Request는 29개에서 240개로 증가했다.
이 실행은 지속 가능한 처리량 기준을 통과하지 못했다.
지난 TTFT Deadline을 구분한 것만으로 처리 용량 문제가 해결되지는 않았다.

새 KV Block 순서 변경은 전체 State 정합성 검증을 통과했다.
아직 A/B/A와 Serving 성능 검증은 완료되지 않았다.
검증 전에는 Descriptor 감소나 성능 향상을 주장하지 않는다.
이후 확인은 Whole Macro와 Generic Transport 변경의 효과에 집중한다.
추가 정책 변형보다 동결된 설정의 독립 확인을 우선한다.

## 독립 확인 설계 보완 — Held-out 성능 평가 이전

각 방법의 선택된 Operating Point에서 독립 Start 3회를 유지한다.
순서는 Seed별로 뒤집거나 회전한다.
Native-relative Decode 비교는 선택된 Dynamic의 실제 QPS에서 수행한다.
두 방법에 같은 Prompt와 Arrival Seed를 사용한다.
Native가 그 QPS에서 과부하라면 모든 요청을 계속 추적한다.
해당 Native 실행을 지속 가능한 Capacity로 표현하지 않는다.
낮은 QPS에서 통과한 비율을 높은 QPS의 Decode 보호 근거로 사용하지 않는다.

1200초 Persistence와 추가 Upper Neighbor는 필수 3회 확인 이후에 수행한다.
Selection 파일에 새 GPU 실험의 종료 시각을 동결한다.
남은 시간이 1500초보다 짧으면 새 Persistence를 시작하지 않는다.
남은 시간이 1000초보다 짧으면 새 Upper Neighbor를 시작하지 않는다.
생략한 실행은 `NOT_RUN_BUDGET`으로 기록한다.
이미 측정한 Discovery Upper Neighbor는 별도로 유지한다.
희소한 QPS 탐색으로 정확한 전역 최대를 주장하지 않는다.

Generic KV 배치를 채택할 경우 Colocated도 같은 옵션을 시험한다.
N32 전체 State 검증을 먼저 수행한다.
이후 동일한 N128 Trace에서 기본/정렬/기본 순서의 A/B/A를 실행한다.
고정된 Decode-priority Goodput으로 해당 옵션을 선택한다.
이 짧은 선정 실험을 지속 가능한 Capacity나 독립 신뢰구간으로 해석하지 않는다.

해당 배치의 PD Capacity Discovery는 Native QPS 3.4와 Static/Dynamic QPS 5.0으로 시작한다.
각 실행은 Warm-up 60초, 측정 600초, 후속 Arrival 90초를 사용한다.
Static과 Dynamic은 같은 Generic Runtime을 사용한다.
확인되지 않은 더 높은 QPS를 최대 처리량으로 집계하지 않는다.

새 Generic Transport의 Completion Margin은 별도로 Calibration한다.
8000/16, N128, QPS 5.0의 독립 Start 3회를 사용한다.
Arrival Seed는 2026111464/2026111465/2026111466이다.
각 Start에서 실제 Migration Group이 8개 이상이어야 한다.
명목 Group Completion의 최대 양의 Residual에 20 ms를 더한다.
10 ms 단위로 올림하고 110 ms 하한을 유지한다.
Delivery Outlier를 제외하지 않는다.

## 동결된 방법의 보충 Capacity 확인 — 2026-11-15 12:14 KST

Dynamic QPS4.8은 독립 확인 3회 모두 적격 기준을 실패했다.
이 실패점을 최대 지속 가능한 처리량으로 집계하지 않는다.
기존 동결 확인과 선택 결과는 그대로 보존한다.
Runtime, 정책, Cost Model, SLO와 Dataset을 변경하지 않는다.

보충 확인은 기존 낮은 후보 QPS4.4 하나로 제한한다.
새 Arrival Seed는 2026111516/2026111517/2026111518이다.
각 실행은 Warm-up 60초, 측정 600초와 후속 Arrival 90초를 사용한다.
앞선 보충 실행이 모두 적격인 경우에만 다음 Start를 시작한다.
실패가 하나라도 발생하면 더 낮은 QPS를 추가 탐색하지 않는다.
성공한 실행만 선택해 평균하지 않는다.

보충 확인은 기존 Controller가 정상 종료한 이후에 시작한다.
추가 실행도 같은 GPU0/1 Quiet-lane 계약을 따른다.
새 Runtime 구현이나 재튜닝 캠페인은 시작하지 않는다.
GPU 측정 종료 계획을 시작 후 21시간으로 보완한다.
이는 보충 Capacity 확인에 최대 약 1시간을 배정한 것이다.
종료 후 분석과 보존을 위한 1.5시간 계획은 유지한다.

이 Pool의 Prompt는 기존 확인에서 이미 관측했다.
따라서 이 시험을 새로운 Prompt에 대한 일반화 검증으로 표현하지 않는다.
새 Arrival에서 동결된 방법의 낮은 Rate를 검사하는 보충 시험이다.
QPS4.4의 Native-relative 5% 비열등성은 같은 Rate의 Native 없이는 주장하지 않는다.
원래 QPS4.8의 상대 Decode 검증과 별도로 보고한다.
이 규칙은 해당 Calibration 실행 전에 고정한다.
후속 Goodput을 보고 Margin을 선택하지 않는다.
Group들은 같은 Start 안에서 독립적이지 않다.
이 상한은 WCET나 미래 p99 보장이 아니다.
Held-out 확인에서 Coverage와 실제 SLO 실패를 별도로 보고한다.

Destination Cohort 8의 ITL이 Native 대비 큰지 별도로 검사한다.
기존 짧은 실행에서는 약 27 ms와 약 24.7 ms의 차이가 관측됐다.
따라서 새 Transport에서는 D4 Streaming도 QPS 4.8에서 비교한다.
Static Last12와 Dynamic에 같은 Source Macro 8과 Destination 상한 4를 적용한다.
Source Physical Batch 상한은 4다.
추가되는 Inter-cohort Decode Residence를 Completion 모델에 포함한다.
실제 실행 전에는 해당 Geometry의 더 높은 Capacity를 가정하지 않는다.

최종 Forced Qualification은 Cut 수와 Source Macro 크기로 요청 수를 정한다.
Source Macro 8에서는 최소 40개 요청으로 Last0/2/4/8/12를 모두 실행한다.
실제 Layer Visit에서 Cut 집합을 다시 검사한다.
Native Cut의 D Imported KV도 검증한다.
이 Forced 분포를 Online 선택 분포로 사용하지 않는다.

별도 후보로 D8 Suffix와 D4 Decode 두 Cohort를 분리하는 실행 구조도 검토했다.
이를 현재 정책에 통합하지 않는다.
먼저 단독 Calibration과 관측한 D4 Decode 비용으로 Component Screen을 계산한다.
새 Paused-request Lifecycle과 First-output Admission의 구현 비용은 포함하지 않는다.
이 낙관적인 계산도 Strong Colocated 대비 충분한 이득을 보이지 않으면 GPU Prototype을 생략한다.
계산 결과를 실제 Serving이나 물리적 Upper Bound로 보고하지 않는다.

새 Lookup의 Decode Cost도 기존 Capture에서 다시 계산한다.
Batch 1/2/3/4/8/16의 cap2 Calibration에는 실제 Full36 Decode Graph가 포함된다.
Batch마다 15개 Group을 사용한다.
각 Group의 최대 Worker Emission Interval을 구하고 그 p95를 Lookup에 저장한다.
기존 온라인 Decode Margin 2 ms는 별도로 유지한다.
CUDA Stream Event는 비교용 Enclosure로만 기록한다.
v4의 Prefix/Suffix/Handoff/Commit Table은 변경하지 않는다.
새 Decode Table은 v5라는 별도 파일로 저장한다.
작은 종속 표본의 p95를 p99 보장이나 WCET로 표현하지 않는다.
변경한 Predictor의 Completion Residual은 이후의 독립 Calibration에서 다시 검사한다.

## 최종 Confirmation Point 선정 규칙 — 평가 전 고정

PD 방법별로 적격한 높은 Discovery Point를 먼저 선택한다.
우선순위는 D8 Whole Macro QPS 5.0, D4 Streaming QPS 4.8이다.
두 점이 모두 실패하면 같은 D4 경로의 QPS 4.4를 Lower Candidate로 시험한다.
그 Lower Point는 확인 전에는 적격 Capacity로 표현하지 않는다.
Native는 새 QPS 3.4 실행이 실패했으므로 QPS 3.2를 확인한다.
Colocated는 SLO별 Graph/Router/Layout 탐색의 QPS 5.0 경로를 확인한다.
QPS 5.2의 RR과 Load-aware 실행은 모두 Queue 판정에 실패했다.

이 규칙은 마지막 D4 Discovery 결과와 Held-out 성능을 보기 전에 기록한다.
Hidden Confirmation 결과를 보고 Cut, Margin, Geometry 또는 QPS를 다시 튜닝하지 않는다.
선정 파일에는 적격 Discovery와 미실행 Lower Candidate를 구분한다.
실제로 실행하는 Core Source가 선정 근거의 Source와 같은지도 검사한다.

Strong Static의 새 Runtime Fraction 대조도 Confirmation 전에 완료한다.
D4, Last12, N128, QPS 4.8에서 100%/93.75%/87.5%를 비교한다.
Source Request Block은 8개다.
짧은 TTFT p95가 100% 기준보다 5% 이상 개선돼야 장기 실험으로 확대한다.
그때 ITL p99 Ratio는 1.05 이하여야 한다.
Joint Attainment도 95% 이상이어야 한다.
적격한 Fraction 중 짧은 TTFT p95가 가장 낮은 하나만 600초로 검사한다.
장기 Gate도 통과하면 Strong Static 선정에 반영한다.
실패하거나 미실행한 경우 이를 적격 Static Capacity로 사용하지 않는다.
Held-out 결과를 보고 이 규칙을 바꾸지 않는다.

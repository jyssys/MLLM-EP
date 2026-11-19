# Dense AR Layer-vs-Token Sweet Spot 및 Minimal Knob 최종 요약

## 실행 계약과 증거 범위

상태: `CLOSED`. Base Evidence Commit은 `4cf3f9b80a2a8618da3bdddb1140fab7f403c3c7`다.
새 Branch는 `flashvep/dense-ar-axis-sweetspot-knob-discovery-20261119`다. 기존 Checkout과 Index는 변경하지 않는다.

- Model: `Qwen/Qwen3-8B`, Revision `b968826d9c46dd6066d109eabc6255188de91218`.
- Runtime: vLLM `0.26.0+cu129`, PyTorch `2.11.0+cu129`, NIXL `1.4.1`.
- GPU: H100 SXM 80GB 두 대. P=GPU0, D=GPU1.
- GPU0 UUID: `GPU-f217c8a0-1142-20f4-d84b-af29f3a47a0d`.
- GPU1 UUID: `GPU-a77f3471-67d4-20b0-9fab-e502d4de5adb`.
- Original BF16, Exact Causal Attention, Full36 Decode, Greedy, Thinking Disabled.
- Input8000. Output은 `ignore_eos=True`로 제어한다. GPU당 KV42GiB와 Prefix Cache Disabled를 유지한다.
- 모든 Worker의 Scheduler 상한은 B8192/S1024다. 이 값은 실제 GPU Batch가 아니다.
- Primary 실제 Batch는 P Prefill1, D Suffix1, D Decode1이다. Decode2/4 통제 실험은 별도로 표시한다.
- 성능 N36/N72는 9의 배수다. Layer2/4/8/12와 Token 인접16-Token 길이를 조합해 전체 GEMM Rows를 맞춘다.
- Prompt 원문 ID와 Input Token Hash를 보존한다. 새 Arrival Seed는 사용하지만 Prompt Distribution은 새로 분리하지 않았다.
- Common Balanced SLO는 TTFT≤5000ms, 모든 ITL≤34ms, Request TPOT≤33ms다. 기존 SLO 파일은 변경하지 않았다.
- 기존 실행 계약은 Runtime max_model_len을8000+max(Output cap,16)으로 설정한다. Output16/64/200에서8016/8064/8200이다.
- Client-visible Coordinator 수신 시각을 Primary로 사용한다. First ITL을 포함한다. Worker 시각은 별도 진단값이다.
- CI는 Paired Independent Server Start를 재표본한다. Request나 Token을 독립 Start로 세지 않는다.
- 신규 CI는 Bootstrap Draw의 Empirical CDF 분위수를 사용한다. n≤6은 전수 재표본한다. n3의27개 Draw를 선형 보간하지 않는다.

예산 시작은 2026-11-19 01:14:22 KST다. 예산 종료 목표는 11:14:22 KST다.
이 보고서 생성 시 경과시간은 8.43시간이다. 10:14:22 이후에는 새 GPU 실험을 시작하지 않는다.

[GPU·환경 기록](dense_ar_axis_sweetspot_knob_discovery_20261119/state/environment.json),
[Base Source 감사](dense_ar_axis_sweetspot_knob_discovery_20261119/state/base_source_audit.json),
[상속 Python742개 Source 감사](dense_ar_axis_sweetspot_knob_discovery_20261119/analysis/inherited_snapshot_integrity.json),
[설치 Runtime 원문 Snapshot](dense_ar_axis_sweetspot_knob_discovery_20261119/references/runtime_source_snapshot_manifest.json),
[Dataset 계약](dense_ar_axis_sweetspot_knob_discovery_20261119/data/pool_contract.json),
[전체 Start별 Metric CSV](dense_ar_axis_sweetspot_knob_discovery_20261119/analysis/per_start_client_metrics.csv)를 보존한다.
실험별 Config, Command, Source ZIP, Prompt Hash, Arrival Trace와 Worker Log는 별도 Raw Directory에 있다.
GitHub에서는 큰 원시파일을 검증된 Lossless 분할 Archive로 제공한다. 원본은 로컬에 유지한다.
[Archive 복원·SHA256 Manifest](dense_ar_axis_sweetspot_knob_discovery_20261119/archives/large_evidence_manifest.json)에 각 파일의 Hash와 복원 방법을 기록한다.

## 핵심 판정

Stage Service Balance와 Whole-cohort Guard는 기존 Sweet Spot을 설명하는 유력한 기전이다.
작은 Token Shape의 실제 Serving GEMM 비효율도 존재한다.
그러나 Forward 이득과 TTFT 이득은 다르다. Queue와 Admission이 전체 결과를 바꿀 수 있다.
물리적 GPU 성분을 완전히 분리하지 못했으므로 Root Cause 판정은 `PARTIAL_ROOT_CAUSE_IDENTIFICATION`이다.
1~2 Knob과 Offline Headroom의 검증 범위는 아래 결과 Table과 개별 보고서로 제한한다.
전체 Workload에 적용되는 Online Dynamic Axis 우위는 `NOT_ESTABLISHED`다.

## 새 Output 통제 실험에서 확인한 역전

Last4/Token≈889, Input8000, QPS6, N72, 실제 P1/D Suffix1/Decode2다.
Partition, 전체 GEMM Rows, Source Window와 Arrival Seed를 일치시켰다.
Output cap만16→64로 바꿨다. Runtime Context 상한도8016→8064로 파생된다.
따라서 이는 Output cap의 전체 효과다. 순수 GPU Decode 시간만의 효과라고 주장하지 않는다.

| Output | Paired Starts | Layer TTFT p95(s) | Token TTFT p95(s) | Layer/Token TTFT | 95% CI | Layer/Token Qualified token/s | 95% CI | Native5% Layer/Token |
|---|---|---|---|---|---|---|---|---|
| 16 | 3 | 8.359 | 8.087 | 1.03354 | [1.00213, 1.08012] | 0.96204 | [0.89697, 0.99920] | PASS/PASS |
| 64 | 3 | 42.511 | 43.302 | 0.98175 | [0.96787, 0.99267] | 1.10680 | [1.02558, 1.15709] | PASS/PASS |


작은 Cut이라는 사실만으로 Layer의 TTFT 우위를 예측할 수 없다.
짧은 Output에서는 P 앞 대기가 D 준비 비용 절감을 상쇄할 수 있다.
긴 Output에서는 D Cohort Residency와 D 앞 대기가 커진다.
이때 같은 D 준비 비용 차이가 후속 요청의 대기에 누적될 수 있다.
정확한 Mean TTFT 구간 분해와 Host 비용은 원인 분석 보고서에 있다.
이 결과는 Finite-cohort 역전이다. Joint SLO95%의 장기 안정 Capacity나 Online Dynamic 이득을 뜻하지 않는다.

## 최소 변수와 Dynamic 필요성

검증 Cell은 15개다. TTFT 승자 CI가 확정된 Cell은 8개다.
Pressure 규칙은 7/8, 두 변수 규칙은 8/8개를 맞췄다.
더 단순한 Static Metadata Cycle도 8/8개를 맞췄다.
이 기준선은 `ΔC=max(p_L,u_L+d)−max(p_T,u_T+d)`의 부호를 사용한다.
P/D 비용은 동결된 Lookup이다. d는 Output cap과 실제 Decode Batch에서 추정한 공통 Decode Demand다.
이는 하나의 파생 Scalar다. 센서 하나만 측정하거나 실제 GPU Slack을 직접 측정한다는 뜻은 아니다.
미확정 Cell을 포함한 Fixed-axis TTFT Regret 점 추정치는 Cycle 0.900%, 두 변수 0.889%다.
이 차이의 통계적 우위나 Online Switching 이득은 입증하지 않았다.
이 Grid에서는 비용 균형 Scalar에 제한적인 Held-out 지지가 있다. 두 번째 상태 변수의 필수성은 입증되지 않았다.
새 Batch4의 승자는 미확정이다. Timing Model 검증 실패도 남아 있다.
따라서 이를 범용 Deadline-safe Admission Rule로 사용하지 않는다.

모든 세 Start의 Fixed Timing/SLO Gate를 통과한 Fine Search Cell은 3개다.
그 부분집합의 모델상 Qualified token/s Headroom 점 추정치는 최대 0.04601%다.
실제 Online 개선이나 전역 Oracle 상한이 아니다. 나머지 Cell의 모델 검증 실패는 별도로 남겼다.
요구한 검증된5% Dynamic Headroom은 확보하지 못했다. 따라서 현재는 Online Axis Scheduler 개발을 보류한다.


동결한 Pressure Scalar, 두 변수 규칙과 Static Metadata Cycle을 같은 새 Cell에서 검사했다.
Static Metadata Cycle도 비용 균형을 하나의 파생 Scalar로 표현할 수 있다.
하지만 Metadata 기준선만큼의 예측 성공은 State-aware Dynamic의 독립 기여가 아니다.
미확정 승자, Calibration 순위 불확실성과 새로운 Batch의 예측 실패를 함께 보존한다.
검증된 Fixed Trace에서5% 이상 Headroom이 확인되지 않으면 복잡한 Axis Scheduler를 보류한다.
상세 Accuracy, Regret와 조건부 Oracle 수치는 각 보고서에 있다.

## 신규 실험 범위

| Evidence Status | 실험 수 |
|---|---|
| CORRECTNESS_ONLY | 24 |
| DIAGNOSTIC | 6 |
| FAILED | 1 |
| ISOLATED_KERNEL_MEASUREMENT | 1 |
| MEASURED_ONLINE | 210 |
| NOT_RUN | 14 |

## 정합성

신규 State Gate는 24개다. 통과 수는 24개다.
N8→N32에서 Original Greedy Output, Full Prompt KV, Final Hidden, Layer/Query 방문과 Lifetime을 검사한다.
Token Prefix/Suffix Hidden은 별도로 검사한다. ACK와 Credit Retirement도 확인한다.
State 복사 실행은 `CORRECTNESS_ONLY`다. 해당 실행의 Latency를 성능 Table에 넣지 않는다.
신규 성능 실행마다 Output, Actual Query Coverage, Matched Rows, Worker B/S와 Reservation Retirement를 다시 검사한다.
Decode2는 Base의 동결된 v3 Adapter State 계약도 재사용한다. Decode4는 신규 State Gate를 먼저 통과해야 한다.
N8/N32는 정확한 Aggregate GEMM Rows 비교가 아니다. 성능 N36/N72에서 정확한 Rows를 검사한다.

## 독립 Start3회 이상의 Layer / Token 결과

| Stage | Cut | Output | QPS | Decode Batch | WIP | Starts | TTFT Ratio | 95% CI | Goodput Ratio | 95% CI |
|---|---|---|---|---|---|---|---|---|---|---|
| batch4_knob_confirmation | 8 | 16 | 3.0 | 4 | 32 | 3 | 1.061 | [0.963, 1.210] | 1.002 | [0.999, 1.003] |
| batch4_knob_confirmation | 8 | 16 | 6.0 | 4 | 32 | 3 | 1.042 | [0.980, 1.135] | 0.909 | [0.812, 1.009] |
| decode_batch_counterfactual | 8 | 16 | 6.0 | 2 | 32 | 3 | 0.888 | [0.809, 0.978] | 1.158 | [1.042, 1.287] |
| decode_batch_counterfactual | 8 | 16 | 6.0 | 4 | 32 | 3 | 1.031 | [0.957, 1.117] | 0.922 | [0.847, 1.025] |
| heldout_confirmation | 2 | 16 | 2.0 | 1 | 32 | 3 | 0.793 | [0.546, 0.997] | 1.005 | [1.000, 1.009] |
| heldout_confirmation | 4 | 16 | 2.0 | 1 | 32 | 3 | 0.900 | [0.801, 1.141] | 1.007 | [0.998, 1.019] |
| heldout_confirmation | 8 | 16 | 2.0 | 1 | 32 | 3 | 0.902 | [0.787, 0.961] | 1.031 | [1.007, 1.084] |
| heldout_confirmation | 12 | 16 | 2.0 | 1 | 32 | 3 | 0.902 | [0.782, 1.016] | 1.071 | [1.007, 1.240] |
| heldout_confirmation | 2 | 16 | 6.0 | 1 | 32 | 3 | 0.895 | [0.830, 0.934] | 1.177 | [1.104, 1.345] |
| heldout_confirmation | 4 | 16 | 6.0 | 1 | 32 | 3 | 0.979 | [0.942, 1.034] | 1.013 | [0.917, 1.103] |
| heldout_confirmation | 8 | 16 | 6.0 | 1 | 32 | 3 | 0.956 | [0.925, 0.972] | 1.078 | [1.019, 1.114] |
| heldout_confirmation | 12 | 16 | 6.0 | 1 | 32 | 3 | 0.935 | [0.915, 0.965] | 1.156 | [1.110, 1.211] |
| new_QPS_confirmation | 4 | 16 | 3.0 | 2 | 32 | 3 | 0.998 | [0.968, 1.065] | 1.006 | [1.000, 1.015] |
| new_batch_cost_calibration | 8 | 16 | 6.0 | 4 | 32 | 3 | 1.029 | [0.946, 1.082] | 0.956 | [0.854, 1.119] |
| output_residency_counterfactual | 4 | 16 | 6.0 | 2 | 32 | 3 | 1.034 | [1.002, 1.080] | 0.962 | [0.897, 0.999] |
| output_residency_counterfactual | 4 | 64 | 6.0 | 2 | 32 | 3 | 0.982 | [0.968, 0.993] | 1.107 | [1.026, 1.157] |
| source_window_control | 12 | 16 | 6.0 | 2 | 4 | 3 | 0.915 | [0.847, 0.973] | 1.125 | [1.075, 1.196] |
| source_window_control | 12 | 16 | 6.0 | 2 | 32 | 3 | 0.863 | [0.784, 0.923] | 1.563 | [1.080, 2.171] |


## 15개 연구 질문에 대한 답

1. Last4의 과거 Token 우위는 해당 Decode2 실행의 P 단계 부담과 관련됐다. 신규 Batch1 결과에 그대로 일반화하지 않는다.
2. Last8 Token의 긴 Ready→D 대기는 대부분 이전 Cohort Guard와 겹쳤다. Guard에는 Suffix와 Decode/Delivery가 포함된다.
3. Last12에서는 두 방식의 D Guard/request가 P Source Service보다 커졌다. 이 Pipeline 불균형과 반복 대기가 주요 설명 후보다.
4. Batch를 맞춰도 Query 길이, Layer 수, Attention 분배와 Hidden Payload가 다르다. 동일 GEMM Rows가 동일 전체 FLOPs를 뜻하지 않는다.
5. 과거 Last8의 낮은 TTFT는 해당 Decode2의 P/D 균형과 일치했다. 신규 Batch1에서는 더 작은 Cut이나 Native가 강할 수 있다.
6. QPS0.5의 O16 Control에서 Layer의 짧은 First Path가 관측됐다. 각 행동1회이므로 확정적인 범용 우위는 아니다.
7. QPS 변화의 승자와 CI는 Online Table에 있다. 점 추정치 부호가 바뀐 사실과 확정 Crossover를 구분한다.
8. Output만 바꾼 인과 비교는 Decode2/N72에서 검사했다. Batch1의 N72/O16과 N36/Long Output은 Count도 달라 단독 인과 근거가 아니다.
9. 단순 Cut 크기보다 P/D Effective Service Demand와 Decode Cohort Residency가 중요한 후보였다. 실제 GPU Idle과 동일한 변수는 아니다.
10. 단일 P/D Pressure Rule은 사전 동결한 κ4를 사용했다. 더 단순한 비용 균형 Scalar의 검증 결과도 위에 제시했다.
11. 두 변수 Rule은 Decode Demand와 이미 도착한 Total Backlog를 사용했다. Static Metadata 기준선보다 높은 확정 Cell Accuracy는 얻지 못했다.
12. 새 Seed/QPS/Output/Batch를 검사했다. 새 Prompt Distribution과 Input Length 일반화는 실행하지 않았다.
13. 사용하는 Counter는 P 제출 전에 관측 가능하다. 비용 Lookup은 Calibration이다. 미래 Arrival/Completion을 읽지 않는다.
14. Offline Headroom은 상태를 갱신하는 제한적 Block Oracle로 계산했다. 검증 Gate 실패 Cell은 Diagnostic이며 Online Speedup이 아니다.
15. 현재는 비용·Queue 모델을 더 검증할 근거가 있다. 강한 Fixed Axis 대비 검증된5% Headroom과 Decode 보호가 없으면 Dynamic 구현을 진행하지 않는다.

## 실패와 미실행

| 실행 | 상태 |
|---|---|
| state_Native_Last0_N8_B1_v1 | FAILED |


초기 Native v1은 Extension 생성자의 Cut 계약에서 측정 전에 실패했다.
유효한 Placeholder Cut을 유지하는 새 v2로 복구했다. 실제 Native는 Full36을 수행한다.
실패한 Config와 Log는 삭제하지 않았다.
온라인 Observer4개 Cell과 Serial Token Cell은5% Observer Gate를 실패했다.
이를 Clean Kernel Time으로 사용하지 않았다. GPU SM/HBM Counter는 `NOT_MEASURED`다.
Main QPS2–10 중 신규로 실행하지 않은 점은 `NOT_RUN`이다.
Inventory의 NOT_RUN14개는 초기 v1에서 v2로 교체한 미실행 Config다. 삭제하거나 실제 실패로 세지 않았다.
장기 Sustainable Throughput과 Online Dynamic Axis Scheduler는 이번 캠페인에서 검증하지 않았다.

## 보고서와 재현 자료

- [dense_ar_axis_existing_evidence_root_cause_audit.md](dense_ar_axis_existing_evidence_root_cause_audit.md)
- [dense_ar_axis_batch_matched_online_results.md](dense_ar_axis_batch_matched_online_results.md)
- [dense_ar_axis_service_queue_causal_decomposition.md](dense_ar_axis_service_queue_causal_decomposition.md)
- [dense_ar_axis_minimal_knob_discovery.md](dense_ar_axis_minimal_knob_discovery.md)
- [dense_ar_axis_offline_oracle_headroom.md](dense_ar_axis_offline_oracle_headroom.md)
- [dense_ar_axis_sweetspot_main_figures.md](dense_ar_axis_sweetspot_main_figures.md)
- [dense_ar_axis_research_decision_log.md](dense_ar_axis_research_decision_log.md)


[전체 Matrix](dense_ar_axis_sweetspot_knob_discovery_20261119/analysis/experiment_inventory.json),
[실측 CSV](dense_ar_axis_sweetspot_knob_discovery_20261119/analysis/per_start_client_metrics.csv),
[Resume Manifest](dense_ar_axis_sweetspot_knob_discovery_20261119/state/resume_progress_manifest.json),
[확인한 원문과 공식 코드](dense_ar_axis_sweetspot_knob_discovery_20261119/references/verified_primary_sources.md)를 보존한다.
최종 Commit과 Remote SHA는 별도 Publication Receipt에서 검증한다.
Burner 복원은 Resource Closeout Receipt에서 확인한다. 검증 전에는 완료했다고 표현하지 않는다.

## 해석 제한

Host Service Enclosure에는 Launch, Scheduling, Transfer와 통지 비용이 들어간다.
이를 Exclusive GPU Kernel Time이나 물리적 GPU Idle로 해석하지 않는다.
Finite-cohort Drain Token/s는 Sustainable Capacity가 아니다.
과부하 실행의 미완료 요청은 제거하지 않는다. 모든 요청을 Drain까지 포함한다.
평균 구간의 합은 Mean TTFT의 정확한 항등식이다. 구간별 p95를 더하지 않는다.
단일 Start의 우위에는 CI를 붙이지 않는다. 여러 Cell의 CI는 Pointwise이며 Multiplicity 보정은 없다.
기존 결과와 신규 결과는 별도로 표시한다. Oracle은 실제 Online Speedup이 아니다.
현재 Token 기준선은 Completed-prefix KV 이후 Suffix를 실행한다. 모든 Streaming Token Pipeline과 비교한 결과는 아니다.
Whole-cohort Guard와 Reservation ACK 경로는 현재 Runtime의 계약이다. Dense AR의 구조적 필수 제약으로 일반화하지 않는다.

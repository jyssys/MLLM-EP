# Figure 목록과 비교 계약

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

## 네 Panel Motivation Figure

SLO Attainment, Mean TTFT, Mean TBT, Generated Output Token/s를 표시한다.
X축은 Offered QPS2–10이다. 신규 측정이 없는 점은 `NOT_RUN`이다.
Output200 Last4/8/12를 각각 표시한다. B512/S16은 표시하지 않는다.
Agg DP2는 기존 N144 Historical 곡선이다. 신규 PD는 별도 Count/Seed의 Batch1 Control이다.
따라서 이를 동일 Arrival Seed의 Paired Four-method 성능 비교라고 주장하지 않는다.
Agg Continuous Batching과 PD의 Fixed Cohort는 실제 Batch 의미도 다르다.
동일 Resource/B/S, PD 내부 Batch Matching, 별도 Batch1 Mechanism Control을 구분한다.

새 PNG/PDF는 Evidence Figure Directory와 `poc_flashvep/` 바로 아래에 저장한다.
기존 Figure는 덮어쓰지 않는다. 빠진 QPS 점을 보간한 측정값으로 표시하지 않는다.

## 전체 Figure

| Stem | Evidence | 파일 | 범위 |
|---|---|---|---|
| historical_pipeline_balance | TRACE_ANALYSIS | [historical_pipeline_balance.png](dense_ar_axis_sweetspot_knob_discovery_20261119/figures/historical_pipeline_balance.png) [historical_pipeline_balance.pdf](dense_ar_axis_sweetspot_knob_discovery_20261119/figures/historical_pipeline_balance.pdf) | Historical measured host stage demand and a derived balance ratio. |
| historical_wait_decomposition | TRACE_ANALYSIS | [historical_wait_decomposition.png](dense_ar_axis_sweetspot_knob_discovery_20261119/figures/historical_wait_decomposition.png) [historical_wait_decomposition.pdf](dense_ar_axis_sweetspot_knob_discovery_20261119/figures/historical_wait_decomposition.pdf) | Exact host-wait identity. No physical GPU activity claim. |
| axis_ratio_O16_q0.5_B1 | MEASURED_ONLINE | [axis_ratio_O16_q0.5_B1.png](dense_ar_axis_sweetspot_knob_discovery_20261119/figures/axis_ratio_O16_q0.5_B1.png) [axis_ratio_O16_q0.5_B1.pdf](dense_ar_axis_sweetspot_knob_discovery_20261119/figures/axis_ratio_O16_q0.5_B1.pdf) | Per-cut matched-axis comparison. No claim for missing points. |
| axis_ratio_O16_q2_B1 | MEASURED_ONLINE | [axis_ratio_O16_q2_B1.png](dense_ar_axis_sweetspot_knob_discovery_20261119/figures/axis_ratio_O16_q2_B1.png) [axis_ratio_O16_q2_B1.pdf](dense_ar_axis_sweetspot_knob_discovery_20261119/figures/axis_ratio_O16_q2_B1.pdf) | Per-cut matched-axis comparison. No claim for missing points. |
| axis_ratio_O16_q6_B1 | MEASURED_ONLINE | [axis_ratio_O16_q6_B1.png](dense_ar_axis_sweetspot_knob_discovery_20261119/figures/axis_ratio_O16_q6_B1.png) [axis_ratio_O16_q6_B1.pdf](dense_ar_axis_sweetspot_knob_discovery_20261119/figures/axis_ratio_O16_q6_B1.pdf) | Per-cut matched-axis comparison. No claim for missing points. |
| axis_ratio_O64_q2_B1 | MEASURED_ONLINE | [axis_ratio_O64_q2_B1.png](dense_ar_axis_sweetspot_knob_discovery_20261119/figures/axis_ratio_O64_q2_B1.png) [axis_ratio_O64_q2_B1.pdf](dense_ar_axis_sweetspot_knob_discovery_20261119/figures/axis_ratio_O64_q2_B1.pdf) | Per-cut matched-axis comparison. No claim for missing points. |
| axis_ratio_O64_q6_B1 | MEASURED_ONLINE | [axis_ratio_O64_q6_B1.png](dense_ar_axis_sweetspot_knob_discovery_20261119/figures/axis_ratio_O64_q6_B1.png) [axis_ratio_O64_q6_B1.pdf](dense_ar_axis_sweetspot_knob_discovery_20261119/figures/axis_ratio_O64_q6_B1.pdf) | Per-cut matched-axis comparison. No claim for missing points. |
| axis_ratio_O200_q2_B1 | MEASURED_ONLINE | [axis_ratio_O200_q2_B1.png](dense_ar_axis_sweetspot_knob_discovery_20261119/figures/axis_ratio_O200_q2_B1.png) [axis_ratio_O200_q2_B1.pdf](dense_ar_axis_sweetspot_knob_discovery_20261119/figures/axis_ratio_O200_q2_B1.pdf) | Per-cut matched-axis comparison. No claim for missing points. |
| axis_ratio_O200_q6_B1 | MEASURED_ONLINE | [axis_ratio_O200_q6_B1.png](dense_ar_axis_sweetspot_knob_discovery_20261119/figures/axis_ratio_O200_q6_B1.png) [axis_ratio_O200_q6_B1.pdf](dense_ar_axis_sweetspot_knob_discovery_20261119/figures/axis_ratio_O200_q6_B1.pdf) | Per-cut matched-axis comparison. No claim for missing points. |
| axis_ratio_O16_q3_B2 | MEASURED_ONLINE | [axis_ratio_O16_q3_B2.png](dense_ar_axis_sweetspot_knob_discovery_20261119/figures/axis_ratio_O16_q3_B2.png) [axis_ratio_O16_q3_B2.pdf](dense_ar_axis_sweetspot_knob_discovery_20261119/figures/axis_ratio_O16_q3_B2.pdf) | Per-cut matched-axis comparison. No claim for missing points. |
| axis_ratio_O16_q6_B2 | MEASURED_ONLINE | [axis_ratio_O16_q6_B2.png](dense_ar_axis_sweetspot_knob_discovery_20261119/figures/axis_ratio_O16_q6_B2.png) [axis_ratio_O16_q6_B2.pdf](dense_ar_axis_sweetspot_knob_discovery_20261119/figures/axis_ratio_O16_q6_B2.pdf) | Per-cut matched-axis comparison. No claim for missing points. |
| axis_ratio_O64_q6_B2 | MEASURED_ONLINE | [axis_ratio_O64_q6_B2.png](dense_ar_axis_sweetspot_knob_discovery_20261119/figures/axis_ratio_O64_q6_B2.png) [axis_ratio_O64_q6_B2.pdf](dense_ar_axis_sweetspot_knob_discovery_20261119/figures/axis_ratio_O64_q6_B2.pdf) | Per-cut matched-axis comparison. No claim for missing points. |
| axis_ratio_O16_q3_B4 | MEASURED_ONLINE | [axis_ratio_O16_q3_B4.png](dense_ar_axis_sweetspot_knob_discovery_20261119/figures/axis_ratio_O16_q3_B4.png) [axis_ratio_O16_q3_B4.pdf](dense_ar_axis_sweetspot_knob_discovery_20261119/figures/axis_ratio_O16_q3_B4.pdf) | Per-cut matched-axis comparison. No claim for missing points. |
| axis_ratio_O16_q6_B4 | MEASURED_ONLINE | [axis_ratio_O16_q6_B4.png](dense_ar_axis_sweetspot_knob_discovery_20261119/figures/axis_ratio_O16_q6_B4.png) [axis_ratio_O16_q6_B4.pdf](dense_ar_axis_sweetspot_knob_discovery_20261119/figures/axis_ratio_O16_q6_B4.pdf) | Per-cut matched-axis comparison. No claim for missing points. |
| four_metrics_o200_last4 | MEASURED_ONLINE_WITH_HISTORICAL_CONTEXT | [four_metrics_o200_last4.png](dense_ar_axis_sweetspot_knob_discovery_20261119/figures/four_metrics_o200_last4.png) [four_metrics_o200_last4.pdf](dense_ar_axis_sweetspot_knob_discovery_20261119/figures/four_metrics_o200_last4.pdf) | New matched PD controls. Historical Agg is explicitly unpaired and uses different actual batch semantics. |
| four_metrics_o200_last8 | MEASURED_ONLINE_WITH_HISTORICAL_CONTEXT | [four_metrics_o200_last8.png](dense_ar_axis_sweetspot_knob_discovery_20261119/figures/four_metrics_o200_last8.png) [four_metrics_o200_last8.pdf](dense_ar_axis_sweetspot_knob_discovery_20261119/figures/four_metrics_o200_last8.pdf) | New matched PD controls. Historical Agg is explicitly unpaired and uses different actual batch semantics. |
| four_metrics_o200_last12 | MEASURED_ONLINE_WITH_HISTORICAL_CONTEXT | [four_metrics_o200_last12.png](dense_ar_axis_sweetspot_knob_discovery_20261119/figures/four_metrics_o200_last12.png) [four_metrics_o200_last12.pdf](dense_ar_axis_sweetspot_knob_discovery_20261119/figures/four_metrics_o200_last12.pdf) | New matched PD controls. Historical Agg is explicitly unpaired and uses different actual batch semantics. |
| controlled_output_cap_crossover | MEASURED_ONLINE | [controlled_output_cap_crossover.png](dense_ar_axis_sweetspot_knob_discovery_20261119/figures/controlled_output_cap_crossover.png) [controlled_output_cap_crossover.pdf](dense_ar_axis_sweetspot_knob_discovery_20261119/figures/controlled_output_cap_crossover.pdf) | The same matched partition and actual batch change rank across Output caps. No exact intermediate crossover threshold or pure-GPU causal effect is inferred. |
| isolated_projection_vs_online_TTFT | ISOLATED_KERNEL_MEASUREMENT_AND_MEASURED_ONLINE | [isolated_projection_vs_online_TTFT.png](dense_ar_axis_sweetspot_knob_discovery_20261119/figures/isolated_projection_vs_online_TTFT.png) [isolated_projection_vs_online_TTFT.pdf](dense_ar_axis_sweetspot_knob_discovery_20261119/figures/isolated_projection_vs_online_TTFT.pdf) | Keep the weighted projection model and real client latency in separate panels. |
| historical_queue_amplification_proxy | TRACE_ANALYSIS | [historical_queue_amplification_proxy.png](dense_ar_axis_sweetspot_knob_discovery_20261119/figures/historical_queue_amplification_proxy.png) [historical_queue_amplification_proxy.pdf](dense_ar_axis_sweetspot_knob_discovery_20261119/figures/historical_queue_amplification_proxy.pdf) | Compare the deterministic overloaded-pipeline proxy with measured tail differences, using all three cuts and starts. |
| source_window_wait_relocation | MEASURED_ONLINE | [source_window_wait_relocation.png](dense_ar_axis_sweetspot_knob_discovery_20261119/figures/source_window_wait_relocation.png) [source_window_wait_relocation.pdf](dense_ar_axis_sweetspot_knob_discovery_20261119/figures/source_window_wait_relocation.pdf) | A source-window intervention can relocate waiting without an equal TTFT gain. |
| minimal_service_demand_boundary | STRUCTURAL_ACCOUNTING | [minimal_service_demand_boundary.png](dense_ar_axis_sweetspot_knob_discovery_20261119/figures/minimal_service_demand_boundary.png) [minimal_service_demand_boundary.pdf](dense_ar_axis_sweetspot_knob_discovery_20261119/figures/minimal_service_demand_boundary.pdf) | A one-demand model can have a crossing or a dominance region, depending on calibrated action costs. |
| decode_batch_control_and_ACK_feedback | MEASURED_ONLINE_AND_TRACE_ANALYSIS | [decode_batch_control_and_ACK_feedback.png](dense_ar_axis_sweetspot_knob_discovery_20261119/figures/decode_batch_control_and_ACK_feedback.png) [decode_batch_control_and_ACK_feedback.pdf](dense_ar_axis_sweetspot_knob_discovery_20261119/figures/decode_batch_control_and_ACK_feedback.pdf) | Changing actual Decode batch alters the critical stage. Keep the unresolved reversal and mandatory ACK delay visible. |
| heldout_one_two_knob_classification | TRACE_ANALYSIS | [heldout_one_two_knob_classification.png](dense_ar_axis_sweetspot_knob_discovery_20261119/figures/heldout_one_two_knob_classification.png) [heldout_one_two_knob_classification.pdf](dense_ar_axis_sweetspot_knob_discovery_20261119/figures/heldout_one_two_knob_classification.pdf) | Compare confident cell labels and frozen shadow-vote forecasts. Unresolved cells remain gray. |
| heldout_one_two_knob_classification_v3 | TRACE_ANALYSIS | [heldout_one_two_knob_classification_v3.png](dense_ar_axis_sweetspot_knob_discovery_20261119/figures/heldout_one_two_knob_classification_v3.png) [heldout_one_two_knob_classification_v3.pdf](dense_ar_axis_sweetspot_knob_discovery_20261119/figures/heldout_one_two_knob_classification_v3.pdf) | Compare confident cell labels and frozen shadow-vote forecasts. Unresolved cells remain gray. |
| heldout_one_two_knob_classification_v4 | TRACE_ANALYSIS | [heldout_one_two_knob_classification_v4.png](dense_ar_axis_sweetspot_knob_discovery_20261119/figures/heldout_one_two_knob_classification_v4.png) [heldout_one_two_knob_classification_v4.pdf](dense_ar_axis_sweetspot_knob_discovery_20261119/figures/heldout_one_two_knob_classification_v4.pdf) | Compare confident cell labels and frozen shadow-vote forecasts. Unresolved cells remain gray. |
| fresh_cost_model_prediction_errors | OFFLINE_REPLAY | [fresh_cost_model_prediction_errors.png](dense_ar_axis_sweetspot_knob_discovery_20261119/figures/fresh_cost_model_prediction_errors.png) [fresh_cost_model_prediction_errors.pdf](dense_ar_axis_sweetspot_knob_discovery_20261119/figures/fresh_cost_model_prediction_errors.pdf) | Separate unseen-batch bias, batch calibration and uncensored control-gap calibration. No test refitting. |
| explicit_control_boundary_replay_errors | OFFLINE_REPLAY | [explicit_control_boundary_replay_errors.png](dense_ar_axis_sweetspot_knob_discovery_20261119/figures/explicit_control_boundary_replay_errors.png) [explicit_control_boundary_replay_errors.pdf](dense_ar_axis_sweetspot_knob_discovery_20261119/figures/explicit_control_boundary_replay_errors.pdf) | Validate the mandatory ACK dependency directly. Fixed-axis validation failures remain visible. No actual dynamic gain is implied. |
| fine_oracle_headroom | OFFLINE_ORACLE | [fine_oracle_headroom.png](dense_ar_axis_sweetspot_knob_discovery_20261119/figures/fine_oracle_headroom.png) [fine_oracle_headroom.pdf](dense_ar_axis_sweetspot_knob_discovery_20261119/figures/fine_oracle_headroom.pdf) | Show bounded-search headroom only with its fixed-trace validation status. Keep Diagnostic results gray. |
| last4_representative_host_timeline | TRACE_ANALYSIS | [last4_representative_host_timeline.png](dense_ar_axis_sweetspot_knob_discovery_20261119/figures/last4_representative_host_timeline.png) [last4_representative_host_timeline.pdf](dense_ar_axis_sweetspot_knob_discovery_20261119/figures/last4_representative_host_timeline.pdf) | Separate actual-run timelines. Cohort guard explains legal admission timing, not GPU idle. |
| last8_representative_host_timeline | TRACE_ANALYSIS | [last8_representative_host_timeline.png](dense_ar_axis_sweetspot_knob_discovery_20261119/figures/last8_representative_host_timeline.png) [last8_representative_host_timeline.pdf](dense_ar_axis_sweetspot_knob_discovery_20261119/figures/last8_representative_host_timeline.pdf) | Separate actual-run timelines. Cohort guard explains legal admission timing, not GPU idle. |
| last12_representative_host_timeline | TRACE_ANALYSIS | [last12_representative_host_timeline.png](dense_ar_axis_sweetspot_knob_discovery_20261119/figures/last12_representative_host_timeline.png) [last12_representative_host_timeline.pdf](dense_ar_axis_sweetspot_knob_discovery_20261119/figures/last12_representative_host_timeline.pdf) | Separate actual-run timelines. Cohort guard explains legal admission timing, not GPU idle. |


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

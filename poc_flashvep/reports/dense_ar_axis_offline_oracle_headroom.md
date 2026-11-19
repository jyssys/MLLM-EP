# Offline Axis Oracle의 Headroom과 한계

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

## 모델 검증을 먼저 검사한다

기존 Discovery 비용으로 별도 과거 Confirmation54개 Start를 예측했다. Timing Gate는48/54개에서 통과했다.
이 결과는 신규 Held-out 결과가 아니다.
신규 비용은 Calibration/Mechanism/Source Window32의30개 Start에서만 추정했다.
Output64/200과 Decode4에는 사전 명시한 외삽이 있다. 해당 테스트 결과로 비용을 다시 맞추지 않았다.
신규 Same-cut Cell 중 Timing과 SLO Gate를 모두 통과한 수는 28/62개다.

Timing Gate는 TTFT p95 상대오차≤10%, Raw Token/s 상대오차≤5%다.
Ready→D 평균 오차는 max(10ms,실측20%) 이하여야 한다.
SLO Gate는 Joint Attainment 절대오차≤0.05와 Qualified Token/s 상대오차≤10%다.
실측 Goodput0이면 예측 Goodput도0이어야 한다.
Gate 실패 Cell의 Headroom은 `DIAGNOSTIC`이다.
단일 P/D Host Step 오차와 누적 Client TTFT 오차도 별도로 기록했다.
MAE, p95/p99 절대오차, Underprediction 비율과 명목값1.05배 Coverage를 포함한다.
이 Coverage는 Calibration된 Upper Bound나 WCET 보장이 아니다.

## Queue State를 갱신하는 Replay

P Queue, FIFO D Ready, Whole-cohort Guard와 Source Outstanding Window를 갱신한다.
KV는 실제 Fresh Native의19113 Free Block을 시작 Budget으로 사용한다.
Prompt+Output 길이에 따른 Uniform Request Reservation Slot을 계산한다.
이는 상세 NIXL Transfer Manager나 Hidden Buffer Simulator가 아니다.
실제 Host State와 Kernel 동시 실행을 완전히 복원했다고 주장하지 않는다.

Same-cut Oracle은18-request Block 안에서 Axis를 선택한다. Decode4에서는36-request Block을 사용한다.
Block마다9-period Token Plan과 Decode Cohort 정렬을 유지한다.
가능한 Sequence는 해당 유한 Grid 안에서 전수 검사한다.
이를 전역 최적 Systems 상한이라고 부르지 않는다.
각 선택 후 후속 Queue와 Guard를 갱신한다. Frozen-state Request별 최소값의 합이 아니다.

## Same-cut Oracle 전체 Cell

표의 Best Fixed는 해당 Trace의 모델상 최선이다. 배포 전에 선정한 설정과 구분한다.

| Stage | Cut | Output | QPS | Batch | Seed | Timing+SLO Gate | 모델 Best Fixed | Fixed Qualified token/s | Oracle Qualified token/s | 모델 Headroom(%) |
|---|---|---|---|---|---|---|---|---|---|---|
| batch4_knob_confirmation | 8 | 16 | 3.0 | 4 | 2026111970 | False | Layer | 52.906 | 52.906 | 0.000 |
| batch4_knob_confirmation | 8 | 16 | 3.0 | 4 | 2026111971 | True | Layer | 47.366 | 47.366 | 0.000 |
| batch4_knob_confirmation | 8 | 16 | 3.0 | 4 | 2026111972 | True | Layer | 43.184 | 43.184 | 0.000 |
| batch4_knob_confirmation | 8 | 16 | 6.0 | 4 | 2026111970 | False | Layer | 28.631 | 28.631 | 0.000 |
| batch4_knob_confirmation | 8 | 16 | 6.0 | 4 | 2026111971 | False | Layer | 52.789 | 52.789 | 0.000 |
| batch4_knob_confirmation | 8 | 16 | 6.0 | 4 | 2026111972 | False | Layer | 50.776 | 50.776 | 0.000 |
| decode_batch_counterfactual | 8 | 16 | 6.0 | 2 | 2026111930 | False | Layer | 30.386 | 30.386 | 0.000 |
| decode_batch_counterfactual | 8 | 16 | 6.0 | 2 | 2026111931 | False | Layer | 55.845 | 55.845 | 0.000 |
| decode_batch_counterfactual | 8 | 16 | 6.0 | 2 | 2026111932 | False | Layer | 26.280 | 26.280 | 0.000 |
| decode_batch_counterfactual | 8 | 16 | 6.0 | 4 | 2026111930 | False | Layer | 36.684 | 36.684 | 0.000 |
| decode_batch_counterfactual | 8 | 16 | 6.0 | 4 | 2026111931 | False | Layer | 64.420 | 64.420 | 0.000 |
| decode_batch_counterfactual | 8 | 16 | 6.0 | 4 | 2026111932 | False | Layer | 32.210 | 32.210 | 0.000 |
| heldout_confirmation | 2 | 16 | 2.0 | 1 | 2026111920 | False | Layer | 36.930 | 36.930 | 0.000 |
| heldout_confirmation | 2 | 16 | 2.0 | 1 | 2026111921 | False | Layer | 33.314 | 33.314 | 0.000 |
| heldout_confirmation | 2 | 16 | 2.0 | 1 | 2026111922 | False | Layer | 32.421 | 32.421 | 0.000 |
| heldout_confirmation | 2 | 16 | 6.0 | 1 | 2026111920 | False | Layer | 8.788 | 8.788 | 0.000 |
| heldout_confirmation | 2 | 16 | 6.0 | 1 | 2026111921 | True | Layer | 9.337 | 9.337 | 0.000 |
| heldout_confirmation | 2 | 16 | 6.0 | 1 | 2026111922 | False | Layer | 9.886 | 9.886 | 0.000 |
| heldout_confirmation | 4 | 16 | 2.0 | 1 | 2026111920 | False | Layer | 36.953 | 36.953 | 0.000 |
| heldout_confirmation | 4 | 16 | 2.0 | 1 | 2026111921 | True | Layer | 33.332 | 33.332 | 0.000 |
| heldout_confirmation | 4 | 16 | 2.0 | 1 | 2026111922 | False | Layer | 32.438 | 32.438 | 0.000 |
| heldout_confirmation | 4 | 16 | 6.0 | 1 | 2026111920 | False | Layer | 8.791 | 8.791 | 0.000 |
| heldout_confirmation | 4 | 16 | 6.0 | 1 | 2026111921 | True | Layer | 9.340 | 9.340 | 0.000 |
| heldout_confirmation | 4 | 16 | 6.0 | 1 | 2026111922 | False | Layer | 9.890 | 9.890 | 0.000 |
| heldout_confirmation | 8 | 16 | 2.0 | 1 | 2026111920 | True | Layer | 32.842 | 32.842 | 0.000 |
| heldout_confirmation | 8 | 16 | 2.0 | 1 | 2026111921 | False | Layer | 32.885 | 32.885 | 0.000 |
| heldout_confirmation | 8 | 16 | 2.0 | 1 | 2026111922 | False | Layer | 32.095 | 32.095 | 0.000 |
| heldout_confirmation | 8 | 16 | 6.0 | 1 | 2026111920 | False | Layer | 6.920 | 6.920 | 0.000 |
| heldout_confirmation | 8 | 16 | 6.0 | 1 | 2026111921 | True | Layer | 6.920 | 6.920 | 0.000 |
| heldout_confirmation | 8 | 16 | 6.0 | 1 | 2026111922 | True | Layer | 7.909 | 7.909 | 0.000 |
| heldout_confirmation | 12 | 16 | 2.0 | 1 | 2026111920 | False | Layer | 21.744 | 21.744 | 0.000 |
| heldout_confirmation | 12 | 16 | 2.0 | 1 | 2026111921 | False | Layer | 32.612 | 32.612 | 0.000 |
| heldout_confirmation | 12 | 16 | 2.0 | 1 | 2026111922 | False | Layer | 31.887 | 31.887 | 0.000 |
| heldout_confirmation | 12 | 16 | 6.0 | 1 | 2026111920 | True | Layer | 6.042 | 6.042 | 0.000 |
| heldout_confirmation | 12 | 16 | 6.0 | 1 | 2026111921 | True | Layer | 6.042 | 6.042 | 0.000 |
| heldout_confirmation | 12 | 16 | 6.0 | 1 | 2026111922 | True | Layer | 6.971 | 6.971 | 0.000 |
| new_QPS_confirmation | 4 | 16 | 3.0 | 2 | 2026111940 | True | Layer | 46.571 | 46.571 | 0.000 |
| new_QPS_confirmation | 4 | 16 | 3.0 | 2 | 2026111941 | True | Layer | 49.456 | 49.456 | 0.000 |
| new_QPS_confirmation | 4 | 16 | 3.0 | 2 | 2026111942 | True | Layer | 47.326 | 47.326 | 0.000 |
| output_residency_counterfactual | 4 | 16 | 6.0 | 2 | 2026111940 | True | Token | 34.745 | 34.745 | 0.000 |
| output_residency_counterfactual | 4 | 16 | 6.0 | 2 | 2026111941 | True | Token | 30.796 | 30.796 | 0.000 |
| output_residency_counterfactual | 4 | 16 | 6.0 | 2 | 2026111942 | True | Token | 21.321 | 21.321 | 0.000 |
| output_residency_counterfactual | 4 | 64 | 6.0 | 2 | 2026111940 | True | Layer | 8.909 | 8.909 | 0.000 |
| output_residency_counterfactual | 4 | 64 | 6.0 | 2 | 2026111941 | False | Layer | 7.795 | 7.795 | 0.000 |
| output_residency_counterfactual | 4 | 64 | 6.0 | 2 | 2026111942 | True | Layer | 8.909 | 8.909 | 0.000 |
| residency_discovery | 4 | 64 | 2.0 | 1 | 2026111903 | True | Layer | 8.100 | 8.100 | 0.000 |
| residency_discovery | 4 | 64 | 6.0 | 1 | 2026111903 | False | Layer | 4.629 | 4.629 | 0.000 |
| residency_discovery | 4 | 200 | 2.0 | 1 | 2026111903 | True | Layer | 2.348 | 2.348 | 0.000 |
| residency_discovery | 4 | 200 | 6.0 | 1 | 2026111903 | False | Layer | 2.348 | 2.348 | 0.000 |
| residency_discovery | 8 | 64 | 2.0 | 1 | 2026111903 | False | Layer | 6.736 | 6.736 | 0.000 |
| residency_discovery | 8 | 64 | 6.0 | 1 | 2026111903 | True | Layer | 4.491 | 4.491 | 0.000 |
| residency_discovery | 8 | 200 | 2.0 | 1 | 2026111903 | True | Token | 2.325 | 2.325 | 0.000 |
| residency_discovery | 8 | 200 | 6.0 | 1 | 2026111903 | True | Token | 2.325 | 2.325 | 0.000 |
| residency_discovery | 12 | 64 | 2.0 | 1 | 2026111903 | False | Layer | 6.622 | 6.622 | 0.000 |
| residency_discovery | 12 | 64 | 6.0 | 1 | 2026111903 | True | Layer | 4.415 | 4.415 | 0.000 |
| residency_discovery | 12 | 200 | 2.0 | 1 | 2026111903 | True | Layer | 2.309 | 2.309 | 0.000 |
| residency_discovery | 12 | 200 | 6.0 | 1 | 2026111903 | False | Layer | 2.309 | 2.309 | 0.000 |
| source_window_control | 4 | 16 | 6.0 | 2 | 2026111901 | False | Token | 56.855 | 56.855 | 0.000 |
| source_window_control | 8 | 16 | 6.0 | 2 | 2026111901 | True | Layer | 59.130 | 59.130 | 0.000 |
| source_window_control | 12 | 16 | 6.0 | 2 | 2026111901 | False | Layer | 50.665 | 50.665 | 0.000 |
| source_window_control | 12 | 16 | 6.0 | 2 | 2026111950 | False | Layer | 31.474 | 31.474 | 0.000 |
| source_window_control | 12 | 16 | 6.0 | 2 | 2026111951 | True | Layer | 24.565 | 24.565 | 0.000 |


## Cut까지 포함한 제한적 Oracle

Layer/Token Last2/4/8/12의8개 행동을 비교한다.
Primary Fixed 설정은 새 Discovery에서 선정한 뒤 Held-out 전에 동결했다.
Cut 변경으로 총 이전 Layer Work가 달라질 수 있다. Matched-work Axis 효과와 별도로 해석한다.
아래 CI는 독립 Arrival Start를 재표본한다. 비용 모델 불확실성은 포함하지 않는다.

| QPS | Starts | Oracle / Frozen Fixed Goodput | 95% CI | Oracle / Trace Best Fixed | 95% CI | 모든 Fixed 행동 검증 |
|---|---|---|---|---|---|---|
| 2.0 | 3 | 1.001 | [1.001, 1.001] | 1.000 | [1.000, 1.000] | False |
| 6.0 | 3 | 1.001 | [1.001, 1.001] | 1.000 | [1.000, 1.000] | False |


[configs/frozen_best_fixed_axes_v1.json](dense_ar_axis_sweetspot_knob_discovery_20261119/configs/frozen_best_fixed_axes_v1.json),
[analysis/frozen_model_validation_and_oracle.json](dense_ar_axis_sweetspot_knob_discovery_20261119/analysis/frozen_model_validation_and_oracle.json),
[analysis/cross_cut_restricted_oracle.json](dense_ar_axis_sweetspot_knob_discovery_20261119/analysis/cross_cut_restricted_oracle.json)에 전체 Sequence와 검증 오차가 있다.
Native는 외부 실측 기준선이다. Oracle의 제한적 Axis 집합에는 Native를 넣지 않았다.
Oracle이 두 Offload Axis보다 높아도 Native나 Agg보다 높다는 뜻이 아니다.

## 더 세밀한 Same-cohort Search

18/36-request Block 때문에 좋은 선택 시점을 놓칠 수 있다.
별도 Search에서는 실제 Decode Cohort마다 Axis를 선택한다.
한 Cohort 안의 Axis는 같다. P1/D Suffix1과 Decode Batch를 유지하므로 Group Fragmentation을 추가하지 않는다.
16-token Prefix Plan의 Token-layer Row 차이를 DP로 추적해 전체 N개의 작업량을 정확히 맞춘다.
Beam Width8과32를 검사한다. 큰 Search에는 작은 Search의 최선과 두 Fixed 행동을 포함한다.
이것은 유한 Beam Search다. 전역 최적 상한이라고 주장하지 않는다.
Oracle은 미래 Trace를 볼 수 있다. Rule은 이미 도착한 상태와 사전 Request-index Plan만 사용한다.
유한 실험의 Row Debt 정산은 Production Stream 일반화와 다르다.

| Cost Model | Stage | Cut | Output | QPS | Batch | Starts | Beam Oracle / Trace Best Fixed | 95% CI | Fixed Trace 검증 | One Rule Ratio | Two Rule Ratio |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 2 | batch4_knob_confirmation | 8 | 16 | 3.0 | 4 | 3 | 1.000 | [1.000, 1.000] | False | 1.000 | 1.000 |
| 2 | batch4_knob_confirmation | 8 | 16 | 6.0 | 4 | 3 | 1.000 | [1.000, 1.000] | False | 1.000 | 1.000 |
| 2 | decode_batch_counterfactual | 8 | 16 | 6.0 | 2 | 3 | 1.000 | [1.000, 1.000] | False | 1.000 | 1.000 |
| 2 | decode_batch_counterfactual | 8 | 16 | 6.0 | 4 | 3 | 1.000 | [1.000, 1.000] | False | 1.000 | 1.000 |
| 2 | heldout_confirmation | 2 | 16 | 2.0 | 1 | 3 | 1.000 | [1.000, 1.000] | False | 1.000 | 1.000 |
| 2 | heldout_confirmation | 2 | 16 | 6.0 | 1 | 3 | 1.000 | [1.000, 1.000] | False | 1.000 | 1.000 |
| 2 | heldout_confirmation | 4 | 16 | 2.0 | 1 | 3 | 1.000 | [1.000, 1.000] | False | 1.000 | 1.000 |
| 2 | heldout_confirmation | 4 | 16 | 6.0 | 1 | 3 | 1.000 | [1.000, 1.000] | False | 1.000 | 1.000 |
| 2 | heldout_confirmation | 8 | 16 | 2.0 | 1 | 3 | 1.000 | [1.000, 1.000] | False | 1.000 | 1.000 |
| 2 | heldout_confirmation | 8 | 16 | 6.0 | 1 | 3 | 1.000 | [1.000, 1.000] | False | 1.000 | 1.000 |
| 2 | heldout_confirmation | 12 | 16 | 2.0 | 1 | 3 | 1.000 | [1.000, 1.000] | False | 1.000 | 1.000 |
| 2 | heldout_confirmation | 12 | 16 | 6.0 | 1 | 3 | 1.000 | [1.000, 1.000] | True | 1.000 | 1.000 |
| 2 | new_QPS_confirmation | 4 | 16 | 3.0 | 2 | 3 | 1.000 | [1.000, 1.001] | True | 1.000 | 0.999 |
| 2 | output_residency_counterfactual | 4 | 16 | 6.0 | 2 | 3 | 1.000 | [1.000, 1.000] | True | 0.977 | 1.000 |
| 2 | output_residency_counterfactual | 4 | 64 | 6.0 | 2 | 3 | 1.000 | [1.000, 1.000] | False | 1.000 | 1.000 |
| 3 | batch4_knob_confirmation | 8 | 16 | 3.0 | 4 | 3 | 1.000 | [1.000, 1.000] | False | 1.000 | 0.999 |
| 3 | batch4_knob_confirmation | 8 | 16 | 6.0 | 4 | 3 | 1.008 | [1.001, 1.034] | False | 0.999 | 1.001 |
| 4 | batch4_knob_confirmation | 8 | 16 | 3.0 | 4 | 3 | 1.000 | [1.000, 1.000] | False | 1.000 | 1.000 |
| 4 | batch4_knob_confirmation | 8 | 16 | 6.0 | 4 | 3 | 1.000 | [1.000, 1.000] | False | 1.000 | 1.000 |


[analysis/fine_cohort_axis_oracle.json](dense_ar_axis_sweetspot_knob_discovery_20261119/analysis/fine_cohort_axis_oracle.json)에 모든 Sequence, Row Debt와 실제 Search 범위를 보존한다.
Mean 비용표는 인접16-token 길이의 개별 시간 차이를 복원하지 않는다.
Timing 또는 SLO Gate를 실패하면 이 Search의 이득도 `DIAGNOSTIC`이다.

## 모든 Fixed Gate를 통과한 세 Start Cell만 별도 표시

다음은 두 Axis와 모든 세 Start의 Fixed-trace Timing/SLO Gate가 통과한 부분집합이다.
Headroom은 같은 모델의 Trace Best Fixed 대비 Beam32의 Qualified token/s 개선율이다.
실측 Dynamic 개선이 아니다. 유한 Search이므로 전역 상한도 아니다.

| Stage | Cut | QPS | Decode Batch | Starts | 모델 Headroom(%) | 95% CI(%) |
|---|---|---|---|---|---|---|
| heldout_confirmation | 12 | 6.0 | 1 | 3 | 0.00000 | [0.00000, 0.00000] |
| new_QPS_confirmation | 4 | 3.0 | 2 | 3 | 0.04601 | [0.01309, 0.07648] |
| output_residency_counterfactual | 4 | 6.0 | 2 | 3 | 0.00000 | [0.00000, 0.00000] |


검증 실패 Cell을 이 Table에서 성공으로 간주하지 않는다. 전체 Cell은 위 Table과 JSON에 남긴다.
이 부분집합에서 작은 Headroom이 나온 사실만으로 모든 조건의 Dynamic 가능성을 배제하지 않는다.

## Control ACK 경계를 명시한 별도 Replay

고정 Mean P Gap은 Axis가 바뀔 때 D Step과 Reservation Reply의 의존성을 갱신하지 못한다.
따라서 마지막 구조적 대조군에서는 Reservation 요청과 ACK를 별도 Event로 계산한다.
P는 ACK 뒤에만 시작한다. D는 다음 모델 Host Step 경계에서 Reply한다.
Preparation은 요청당 하나의 비선점 Step으로 근사한다. Decode는 출력 간격별 경계를 사용한다.
실제 Async Front-matter와 Step 내부 경계는 계측하지 못했다. 이 근사는 검증 대상이지 확정된 Runtime 복제본이 아니다.

새 Calibration6개 Start만으로 Non-ACK P Gap과 Idle RTT를 고정했다.
규칙과 Simulator 구조는 새 Seed1970/1971/1972의 측정 Arrival 전에 동결했다.
아래 오차는 그 신규 Confirmation의 Fixed-axis 실행에서 계산한다. 검사 실패 후 재튜닝하지 않는다.

| Axis | QPS | Seed | TTFT p95 오차(%) | Raw token/s 오차(%) | Ready→D 평균 오차(ms) | ACK MAE(ms) | ACK 절대오차 p95(ms) | Timing Gate | SLO Gate |
|---|---|---|---|---|---|---|---|---|---|
| Layer | 3.0 | 2026111970 | 5.223 | 0.007 | 11.917 | 20.149 | 53.592 | True | True |
| Layer | 3.0 | 2026111971 | 0.849 | 0.007 | 12.617 | 13.769 | 53.078 | True | True |
| Layer | 3.0 | 2026111972 | 1.728 | 0.584 | 10.098 | 16.750 | 68.803 | True | True |
| Layer | 6.0 | 2026111970 | 7.252 | 3.021 | 5.593 | 18.863 | 54.033 | True | True |
| Layer | 6.0 | 2026111971 | 5.951 | 2.396 | 23.803 | 23.640 | 53.043 | True | False |
| Layer | 6.0 | 2026111972 | 1.459 | 0.543 | 20.941 | 25.215 | 68.744 | True | True |
| Token | 3.0 | 2026111970 | 30.489 | 0.018 | 19.749 | 21.911 | 81.811 | False | True |
| Token | 3.0 | 2026111971 | 1.333 | 0.846 | 21.593 | 19.445 | 82.143 | True | True |
| Token | 3.0 | 2026111972 | 0.197 | 1.006 | 13.489 | 16.561 | 72.187 | True | True |
| Token | 6.0 | 2026111970 | 14.580 | 6.263 | 3.388 | 25.102 | 81.851 | False | False |
| Token | 6.0 | 2026111971 | 19.137 | 6.565 | 3.428 | 24.126 | 81.797 | False | False |
| Token | 6.0 | 2026111972 | 12.615 | 4.148 | 4.863 | 18.940 | 67.142 | False | False |


두 Fixed Axis의 Timing/SLO Gate가 모두 통과한 Trace에만 Beam Search를 적용한다.
실패한 Trace는 삭제하지 않는다. 해당 Trace의 새 Mixed Search는 `NOT_RUN`이다.
아래 Table은 조건부 통과 집합이다. 실패한 Start를 제외한 무조건적 캠페인 성공으로 해석하지 않는다.

| QPS | 조건부 검증 Start | Beam / Trace Best Fixed Qualified token/s | 95% CI |
|---|---|---|---|
| 3.0 | 2 | 1.001 | [1.000, 1.002] |


[configs/frozen_control_feedback_replay_inputs_v1.json](dense_ar_axis_sweetspot_knob_discovery_20261119/configs/frozen_control_feedback_replay_inputs_v1.json),
[state/control_feedback_replay_inputs_frozen.json](dense_ar_axis_sweetspot_knob_discovery_20261119/state/control_feedback_replay_inputs_frozen.json),
[analysis/control_boundary_replay_validation.json](dense_ar_axis_sweetspot_knob_discovery_20261119/analysis/control_boundary_replay_validation.json)에 동결 시각, 전체 오차와 실패를 보존한다.
이 별도 모델도 Mean 비용을 사용한다. ITL p99 보장이나 실제 Dynamic Serving 이득을 뜻하지 않는다.

## 후속 Dynamic 연구 판단

검증된 Headroom5% 이상을 후속 연구의 목표로 삼는다.
작은 차이가 비용 예측 오차보다 작으면 Dynamic 구현을 보류한다.
평균 Decode Interval 모델은 ITL p99 또는 Native-relative5%를 보장하지 않는다.
Mixed Layer/Token State Routing의 실제 Online 정합성은 `NOT_RUN`이다.
이번 연구에서는 Online Dynamic Axis Scheduler를 구현하지 않았다.

## 다음 연구의 최소 검증 순서

먼저 실제 D Host Step의 Control 처리 경계를 계측한다. 계측 전후5% Observer Gate를 다시 검사한다.
그 다음 ACK 대기를 줄이는 Reservation Prefetch 대조군을 두 Axis에 공통 적용할 수 있는지 검토한다.
Credit, ACK와 Lifetime 정합성을 유지해야 한다. 이 대조군은 이번에 구현하거나 측정하지 않았다.
이 변경으로 Axis 우위가 사라지면 기존 이득을 알고리즘의 고유 이득으로 주장하지 않는다.
Whole-cohort Guard를 변경하는 실험도 별도 Runtime 실험으로 분리해야 한다.

새 Queue 모델은 모든 Fixed Axis의 Timing/SLO Gate를 독립 Start에서 먼저 통과해야 한다.
Strong Fixed 기준선에는 각 Cut의 Layer/Token과 Native Fallback을 포함한다.
검증된 Oracle에서5% 이상 여지가 재현될 때만 비용 균형 Scalar에 상태 변수를 추가한다.
실제 Mixed-axis State 정합성과 장기 Queue 안정성도 그 이후 별도 Gate다.
이 순서를 통과하기 전에는 복잡한 Online Controller 구현을 보류한다.

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

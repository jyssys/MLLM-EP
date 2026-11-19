# Dense AR 실제 Batch-matched Online 결과

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
## 정합성

신규 State Gate는 24개다. 통과 수는 24개다.
N8→N32에서 Original Greedy Output, Full Prompt KV, Final Hidden, Layer/Query 방문과 Lifetime을 검사한다.
Token Prefix/Suffix Hidden은 별도로 검사한다. ACK와 Credit Retirement도 확인한다.
State 복사 실행은 `CORRECTNESS_ONLY`다. 해당 실행의 Latency를 성능 Table에 넣지 않는다.
신규 성능 실행마다 Output, Actual Query Coverage, Matched Rows, Worker B/S와 Reservation Retirement를 다시 검사한다.
Decode2는 Base의 동결된 v3 Adapter State 계약도 재사용한다. Decode4는 신규 State Gate를 먼저 통과해야 한다.
N8/N32는 정확한 Aggregate GEMM Rows 비교가 아니다. 성능 N36/N72에서 정확한 Rows를 검사한다.

## 실행 Matrix

| Evidence Status | 실험 수 |
|---|---|
| CORRECTNESS_ONLY | 24 |
| DIAGNOSTIC | 6 |
| FAILED | 1 |
| ISOLATED_KERNEL_MEASUREMENT | 1 |
| MEASURED_ONLINE | 210 |
| NOT_RUN | 14 |


신규 Clean Start는 198개다. 모든 실행 상태는 [analysis/experiment_inventory.json](dense_ar_axis_sweetspot_knob_discovery_20261119/analysis/experiment_inventory.json)에 있다.
`NOT_RUN` Cell은 보간하지 않는다. Serial Observer Control은 Poisson Primary에서 제외한다.
`MEASURED_ONLINE`의 의미는 Client-visible 실측이다. 물리적 GPU 계산시간 측정을 뜻하지 않는다.

## 신규 실측 전체 Cell

단위는 TTFT초, ITL/TPOT밀리초, 처리량Output tokens/s다. Joint는 요청 비율이다.
Actual Batch 열은 P Prefill / D Prefill / D Decode를 뜻한다.
Native에는 D Prompt Prefill이 없으므로 가운데 값은0이다. 공통 준비 정책의 상한은1이다.

| Stage | Arm | Output | QPS | N | Batch | WIP | Starts | TTFT p95(s) | ITL p99(ms) | TPOT p95(ms) | Raw token/s | Qualified token/s | Joint |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| batch4_knob_confirmation | Layer8 | 16 | 3.0 | 72 | 1/1/4 | 32 | 3 | 2.313 | 24.952 | 23.988 | 47.793 | 47.793 | 1.000 |
| batch4_knob_confirmation | Layer8 | 16 | 6.0 | 72 | 1/1/4 | 32 | 3 | 7.334 | 24.858 | 23.972 | 59.895 | 35.169 | 0.593 |
| batch4_knob_confirmation | Native0 | 16 | 3.0 | 72 | 1/0/4 | 32 | 3 | 2.813 | 24.140 | 22.982 | 46.701 | 46.701 | 1.000 |
| batch4_knob_confirmation | Native0 | 16 | 6.0 | 72 | 1/0/4 | 32 | 3 | 10.224 | 23.771 | 22.957 | 51.748 | 22.229 | 0.431 |
| batch4_knob_confirmation | Token8 | 16 | 3.0 | 72 | 1/1/4 | 32 | 3 | 2.181 | 24.447 | 23.937 | 47.721 | 47.721 | 1.000 |
| batch4_knob_confirmation | Token8 | 16 | 6.0 | 72 | 1/1/4 | 32 | 3 | 7.039 | 24.413 | 23.771 | 60.838 | 38.685 | 0.639 |
| calibration | Layer2 | 16 | 0.5 | 36 | 1/1/1 | 32 | 1 | 0.364 | 24.127 | 23.761 | 8.982 | 8.982 | 1.000 |
| calibration | Layer4 | 16 | 0.5 | 36 | 1/1/1 | 32 | 1 | 0.338 | 24.088 | 23.800 | 8.983 | 8.983 | 1.000 |
| calibration | Layer8 | 16 | 0.5 | 36 | 1/1/1 | 32 | 1 | 0.377 | 24.412 | 23.915 | 8.979 | 8.979 | 1.000 |
| calibration | Layer12 | 16 | 0.5 | 36 | 1/1/1 | 32 | 1 | 0.368 | 24.484 | 23.898 | 8.980 | 8.980 | 1.000 |
| calibration | Native0 | 16 | 0.5 | 36 | 1/0/1 | 32 | 1 | 0.336 | 23.814 | 23.011 | 8.985 | 8.985 | 1.000 |
| calibration | Token2 | 16 | 0.5 | 36 | 1/1/1 | 32 | 1 | 0.402 | 24.146 | 23.761 | 8.976 | 8.976 | 1.000 |
| calibration | Token4 | 16 | 0.5 | 36 | 1/1/1 | 32 | 1 | 0.373 | 24.119 | 23.801 | 8.980 | 8.980 | 1.000 |
| calibration | Token8 | 16 | 0.5 | 36 | 1/1/1 | 32 | 1 | 0.388 | 24.255 | 23.899 | 8.976 | 8.976 | 1.000 |
| calibration | Token12 | 16 | 0.5 | 36 | 1/1/1 | 32 | 1 | 0.701 | 24.355 | 23.945 | 8.858 | 8.858 | 1.000 |
| decode_batch_counterfactual | Layer8 | 16 | 6.0 | 72 | 1/1/2 | 32 | 3 | 7.305 | 24.448 | 23.834 | 60.626 | 40.354 | 0.662 |
| decode_batch_counterfactual | Layer8 | 16 | 6.0 | 72 | 1/1/4 | 32 | 3 | 7.347 | 24.947 | 23.909 | 61.317 | 37.963 | 0.620 |
| decode_batch_counterfactual | Native0 | 16 | 6.0 | 72 | 1/0/2 | 32 | 3 | 10.673 | 23.637 | 22.946 | 51.134 | 23.170 | 0.454 |
| decode_batch_counterfactual | Native0 | 16 | 6.0 | 72 | 1/0/4 | 32 | 3 | 10.688 | 24.115 | 22.987 | 51.745 | 22.327 | 0.431 |
| decode_batch_counterfactual | Token8 | 16 | 6.0 | 72 | 1/1/2 | 32 | 3 | 8.223 | 24.198 | 23.700 | 57.716 | 34.841 | 0.602 |
| decode_batch_counterfactual | Token8 | 16 | 6.0 | 72 | 1/1/4 | 32 | 3 | 7.127 | 24.454 | 23.945 | 61.990 | 41.167 | 0.662 |
| heldout_confirmation | Layer2 | 16 | 2.0 | 72 | 1/1/1 | 32 | 3 | 2.029 | 24.090 | 23.795 | 34.266 | 34.266 | 1.000 |
| heldout_confirmation | Layer2 | 16 | 6.0 | 72 | 1/1/1 | 32 | 3 | 15.927 | 24.082 | 23.803 | 41.187 | 10.297 | 0.250 |
| heldout_confirmation | Layer4 | 16 | 2.0 | 72 | 1/1/1 | 32 | 3 | 2.398 | 24.230 | 23.855 | 34.202 | 34.202 | 1.000 |
| heldout_confirmation | Layer4 | 16 | 6.0 | 72 | 1/1/1 | 32 | 3 | 17.722 | 24.173 | 23.848 | 38.586 | 8.759 | 0.227 |
| heldout_confirmation | Layer8 | 16 | 2.0 | 72 | 1/1/1 | 32 | 3 | 3.018 | 24.527 | 23.912 | 33.506 | 32.363 | 0.968 |
| heldout_confirmation | Layer8 | 16 | 6.0 | 72 | 1/1/1 | 32 | 3 | 19.389 | 24.494 | 23.914 | 36.430 | 7.766 | 0.213 |
| heldout_confirmation | Layer12 | 16 | 2.0 | 72 | 1/1/1 | 32 | 3 | 3.699 | 24.698 | 23.924 | 32.977 | 30.910 | 0.940 |
| heldout_confirmation | Layer12 | 16 | 6.0 | 72 | 1/1/1 | 32 | 3 | 21.405 | 24.652 | 23.923 | 34.138 | 6.803 | 0.199 |
| heldout_confirmation | Native0 | 16 | 2.0 | 72 | 1/0/1 | 32 | 3 | 1.631 | 23.482 | 22.997 | 34.423 | 34.252 | 0.995 |
| heldout_confirmation | Native0 | 16 | 6.0 | 72 | 1/0/1 | 32 | 3 | 14.565 | 23.392 | 22.946 | 43.461 | 11.869 | 0.273 |
| heldout_confirmation | Token2 | 16 | 2.0 | 72 | 1/1/1 | 32 | 3 | 2.560 | 24.116 | 23.745 | 34.100 | 34.100 | 1.000 |
| heldout_confirmation | Token2 | 16 | 6.0 | 72 | 1/1/1 | 32 | 3 | 17.794 | 24.099 | 23.748 | 38.509 | 8.752 | 0.227 |
| heldout_confirmation | Token4 | 16 | 2.0 | 72 | 1/1/1 | 32 | 3 | 2.665 | 24.137 | 23.841 | 33.954 | 33.954 | 1.000 |
| heldout_confirmation | Token4 | 16 | 6.0 | 72 | 1/1/1 | 32 | 3 | 18.097 | 24.169 | 23.850 | 38.093 | 8.645 | 0.227 |
| heldout_confirmation | Token8 | 16 | 2.0 | 72 | 1/1/1 | 32 | 3 | 3.345 | 24.239 | 23.883 | 33.149 | 31.383 | 0.949 |
| heldout_confirmation | Token8 | 16 | 6.0 | 72 | 1/1/1 | 32 | 3 | 20.287 | 24.239 | 23.885 | 35.388 | 7.206 | 0.204 |
| heldout_confirmation | Token12 | 16 | 2.0 | 72 | 1/1/1 | 32 | 3 | 4.099 | 24.573 | 23.935 | 32.569 | 28.868 | 0.889 |
| heldout_confirmation | Token12 | 16 | 6.0 | 72 | 1/1/1 | 32 | 3 | 22.897 | 24.545 | 23.969 | 32.600 | 5.886 | 0.181 |
| mechanism_discovery | Layer2 | 16 | 2.0 | 72 | 1/1/1 | 32 | 1 | 2.151 | 24.189 | 23.798 | 35.670 | 35.670 | 1.000 |
| mechanism_discovery | Layer2 | 16 | 6.0 | 72 | 1/1/1 | 32 | 1 | 16.719 | 24.087 | 23.778 | 40.974 | 10.813 | 0.264 |
| mechanism_discovery | Layer4 | 16 | 2.0 | 72 | 1/1/1 | 32 | 1 | 2.037 | 24.197 | 23.861 | 35.700 | 35.205 | 0.986 |
| mechanism_discovery | Layer4 | 16 | 6.0 | 72 | 1/1/1 | 32 | 1 | 17.596 | 24.177 | 23.868 | 39.688 | 9.922 | 0.250 |
| mechanism_discovery | Layer8 | 16 | 2.0 | 72 | 1/1/1 | 32 | 1 | 3.788 | 24.468 | 23.904 | 34.905 | 34.905 | 1.000 |
| mechanism_discovery | Layer8 | 16 | 6.0 | 72 | 1/1/1 | 32 | 1 | 20.793 | 24.447 | 23.909 | 35.556 | 7.901 | 0.222 |
| mechanism_discovery | Layer12 | 16 | 2.0 | 72 | 1/1/1 | 32 | 1 | 4.991 | 24.619 | 23.929 | 32.757 | 30.937 | 0.944 |
| mechanism_discovery | Layer12 | 16 | 6.0 | 72 | 1/1/1 | 32 | 1 | 22.701 | 24.573 | 23.921 | 33.489 | 6.512 | 0.194 |
| mechanism_discovery | Native0 | 16 | 2.0 | 72 | 1/0/1 | 32 | 1 | 1.642 | 23.493 | 22.992 | 35.938 | 34.441 | 0.958 |
| mechanism_discovery | Native0 | 16 | 6.0 | 72 | 1/0/1 | 32 | 1 | 16.011 | 23.528 | 22.960 | 42.100 | 11.694 | 0.278 |
| mechanism_discovery | Token2 | 16 | 2.0 | 72 | 1/1/1 | 32 | 1 | 3.378 | 24.210 | 23.754 | 35.336 | 35.336 | 1.000 |
| mechanism_discovery | Token2 | 16 | 6.0 | 72 | 1/1/1 | 32 | 1 | 19.711 | 24.148 | 23.755 | 36.843 | 8.699 | 0.236 |
| mechanism_discovery | Token4 | 16 | 2.0 | 72 | 1/1/1 | 32 | 1 | 2.309 | 24.131 | 23.841 | 35.604 | 35.604 | 1.000 |
| mechanism_discovery | Token4 | 16 | 6.0 | 72 | 1/1/1 | 32 | 1 | 18.889 | 24.124 | 23.840 | 37.938 | 8.958 | 0.236 |
| mechanism_discovery | Token8 | 16 | 2.0 | 72 | 1/1/1 | 32 | 1 | 3.622 | 24.295 | 23.857 | 34.700 | 34.218 | 0.986 |
| mechanism_discovery | Token8 | 16 | 6.0 | 72 | 1/1/1 | 32 | 1 | 21.367 | 24.262 | 23.917 | 34.903 | 7.756 | 0.222 |
| mechanism_discovery | Token12 | 16 | 2.0 | 72 | 1/1/1 | 32 | 1 | 4.871 | 24.487 | 23.899 | 32.898 | 31.984 | 0.972 |
| mechanism_discovery | Token12 | 16 | 6.0 | 72 | 1/1/1 | 32 | 1 | 23.595 | 24.646 | 23.963 | 32.575 | 6.334 | 0.194 |
| new_QPS_confirmation | Layer4 | 16 | 3.0 | 72 | 1/1/2 | 32 | 3 | 1.813 | 24.351 | 23.832 | 47.895 | 47.895 | 1.000 |
| new_QPS_confirmation | Native0 | 16 | 3.0 | 72 | 1/0/2 | 32 | 3 | 2.471 | 23.569 | 22.945 | 47.509 | 47.509 | 1.000 |
| new_QPS_confirmation | Token4 | 16 | 3.0 | 72 | 1/1/2 | 32 | 3 | 1.816 | 24.183 | 23.774 | 47.832 | 47.603 | 0.995 |
| new_batch_cost_calibration | Layer8 | 16 | 6.0 | 72 | 1/1/4 | 32 | 3 | 6.164 | 24.930 | 23.991 | 61.622 | 47.118 | 0.764 |
| new_batch_cost_calibration | Token8 | 16 | 6.0 | 72 | 1/1/4 | 32 | 3 | 5.989 | 24.586 | 23.675 | 62.194 | 49.264 | 0.792 |
| output_residency_counterfactual | Layer4 | 16 | 6.0 | 72 | 1/1/2 | 32 | 3 | 8.359 | 24.325 | 23.806 | 56.050 | 27.553 | 0.491 |
| output_residency_counterfactual | Layer4 | 64 | 6.0 | 72 | 1/1/2 | 32 | 3 | 42.511 | 24.138 | 23.670 | 80.282 | 8.920 | 0.111 |
| output_residency_counterfactual | Native0 | 16 | 6.0 | 72 | 1/0/2 | 32 | 3 | 9.928 | 23.588 | 22.940 | 51.913 | 20.679 | 0.398 |
| output_residency_counterfactual | Native0 | 64 | 6.0 | 72 | 1/0/2 | 32 | 3 | 40.070 | 23.388 | 22.854 | 84.094 | 9.344 | 0.111 |
| output_residency_counterfactual | Token4 | 16 | 6.0 | 72 | 1/1/2 | 32 | 3 | 8.087 | 24.273 | 23.782 | 56.800 | 28.640 | 0.505 |
| output_residency_counterfactual | Token4 | 64 | 6.0 | 72 | 1/1/2 | 32 | 3 | 43.302 | 24.092 | 23.699 | 79.132 | 8.059 | 0.102 |
| residency_discovery | Layer4 | 64 | 2.0 | 36 | 1/1/1 | 32 | 1 | 34.604 | 24.087 | 23.748 | 41.478 | 8.065 | 0.194 |
| residency_discovery | Layer4 | 64 | 6.0 | 36 | 1/1/1 | 32 | 1 | 45.281 | 24.096 | 23.745 | 41.890 | 4.654 | 0.111 |
| residency_discovery | Layer4 | 200 | 2.0 | 36 | 1/1/1 | 32 | 1 | 140.377 | 24.037 | 23.784 | 42.336 | 2.352 | 0.056 |
| residency_discovery | Layer4 | 200 | 6.0 | 36 | 1/1/1 | 32 | 1 | 152.281 | 24.075 | 23.781 | 42.126 | 1.170 | 0.028 |
| residency_discovery | Layer8 | 64 | 2.0 | 36 | 1/1/1 | 32 | 1 | 35.923 | 24.272 | 23.766 | 40.480 | 6.747 | 0.167 |
| residency_discovery | Layer8 | 64 | 6.0 | 36 | 1/1/1 | 32 | 1 | 46.344 | 24.247 | 23.772 | 41.022 | 4.558 | 0.111 |
| residency_discovery | Layer8 | 200 | 2.0 | 36 | 1/1/1 | 32 | 1 | 141.177 | 24.048 | 23.748 | 42.123 | 2.340 | 0.056 |
| residency_discovery | Layer8 | 200 | 6.0 | 36 | 1/1/1 | 32 | 1 | 152.823 | 24.136 | 23.782 | 42.001 | 2.333 | 0.056 |
| residency_discovery | Layer12 | 64 | 2.0 | 36 | 1/1/1 | 32 | 1 | 36.873 | 24.419 | 23.789 | 39.791 | 6.632 | 0.167 |
| residency_discovery | Layer12 | 64 | 6.0 | 36 | 1/1/1 | 32 | 1 | 47.375 | 24.310 | 23.765 | 40.281 | 4.476 | 0.111 |
| residency_discovery | Layer12 | 200 | 2.0 | 36 | 1/1/1 | 32 | 1 | 143.567 | 24.141 | 23.801 | 41.496 | 2.305 | 0.056 |
| residency_discovery | Layer12 | 200 | 6.0 | 36 | 1/1/1 | 32 | 1 | 153.887 | 24.114 | 23.780 | 41.731 | 2.318 | 0.056 |
| residency_discovery | Native0 | 64 | 2.0 | 36 | 1/0/1 | 32 | 1 | 32.075 | 23.238 | 22.917 | 43.606 | 8.479 | 0.194 |
| residency_discovery | Native0 | 64 | 6.0 | 36 | 1/0/1 | 32 | 1 | 43.275 | 23.264 | 22.906 | 43.586 | 4.843 | 0.111 |
| residency_discovery | Native0 | 200 | 2.0 | 36 | 1/0/1 | 32 | 1 | 136.364 | 23.313 | 22.942 | 43.441 | 2.413 | 0.056 |
| residency_discovery | Native0 | 200 | 6.0 | 36 | 1/0/1 | 32 | 1 | 147.660 | 23.334 | 22.952 | 43.409 | 1.206 | 0.028 |
| residency_discovery | Token4 | 64 | 2.0 | 36 | 1/1/1 | 32 | 1 | 34.497 | 24.066 | 23.751 | 41.550 | 8.079 | 0.194 |
| residency_discovery | Token4 | 64 | 6.0 | 36 | 1/1/1 | 32 | 1 | 45.783 | 24.063 | 23.750 | 41.466 | 3.455 | 0.083 |
| residency_discovery | Token4 | 200 | 2.0 | 36 | 1/1/1 | 32 | 1 | 141.802 | 24.084 | 23.786 | 41.931 | 2.330 | 0.056 |
| residency_discovery | Token4 | 200 | 6.0 | 36 | 1/1/1 | 32 | 1 | 151.959 | 24.033 | 23.783 | 42.207 | 2.345 | 0.056 |
| residency_discovery | Token8 | 64 | 2.0 | 36 | 1/1/1 | 32 | 1 | 35.400 | 24.074 | 23.760 | 40.856 | 7.944 | 0.194 |
| residency_discovery | Token8 | 64 | 6.0 | 36 | 1/1/1 | 32 | 1 | 47.269 | 24.187 | 23.792 | 40.349 | 4.483 | 0.111 |
| residency_discovery | Token8 | 200 | 2.0 | 36 | 1/1/1 | 32 | 1 | 141.495 | 23.996 | 23.776 | 42.057 | 2.336 | 0.056 |
| residency_discovery | Token8 | 200 | 6.0 | 36 | 1/1/1 | 32 | 1 | 152.789 | 24.035 | 23.790 | 42.024 | 2.335 | 0.056 |
| residency_discovery | Token12 | 64 | 2.0 | 36 | 1/1/1 | 32 | 1 | 36.758 | 24.261 | 23.770 | 39.893 | 6.649 | 0.167 |
| residency_discovery | Token12 | 64 | 6.0 | 36 | 1/1/1 | 32 | 1 | 48.291 | 24.309 | 23.760 | 39.640 | 4.404 | 0.111 |
| residency_discovery | Token12 | 200 | 2.0 | 36 | 1/1/1 | 32 | 1 | 143.106 | 24.130 | 23.794 | 41.624 | 2.312 | 0.056 |
| residency_discovery | Token12 | 200 | 6.0 | 36 | 1/1/1 | 32 | 1 | 154.209 | 24.126 | 23.781 | 41.642 | 1.157 | 0.028 |
| source_window_control | Layer4 | 16 | 6.0 | 72 | 1/1/2 | 4 | 1 | 5.105 | 24.543 | 23.828 | 55.391 | 51.544 | 0.931 |
| source_window_control | Layer4 | 16 | 6.0 | 72 | 1/1/2 | 32 | 1 | 4.689 | 24.293 | 23.753 | 56.586 | 55.800 | 0.986 |
| source_window_control | Layer8 | 16 | 6.0 | 72 | 1/1/2 | 4 | 1 | 4.206 | 24.382 | 23.820 | 59.011 | 59.011 | 1.000 |
| source_window_control | Layer8 | 16 | 6.0 | 72 | 1/1/2 | 32 | 1 | 4.142 | 24.452 | 23.877 | 59.192 | 59.192 | 1.000 |
| source_window_control | Layer12 | 16 | 6.0 | 72 | 1/1/2 | 4 | 3 | 8.310 | 24.383 | 23.820 | 54.054 | 30.689 | 0.569 |
| source_window_control | Layer12 | 16 | 6.0 | 72 | 1/1/2 | 32 | 3 | 8.190 | 24.295 | 23.762 | 54.586 | 35.049 | 0.639 |
| source_window_control | Token4 | 16 | 6.0 | 72 | 1/1/2 | 4 | 1 | 4.600 | 24.319 | 23.780 | 56.973 | 56.973 | 1.000 |
| source_window_control | Token4 | 16 | 6.0 | 72 | 1/1/2 | 32 | 1 | 4.593 | 24.328 | 23.826 | 56.901 | 56.901 | 1.000 |
| source_window_control | Token8 | 16 | 6.0 | 72 | 1/1/2 | 4 | 1 | 4.564 | 24.437 | 23.784 | 56.964 | 56.964 | 1.000 |
| source_window_control | Token8 | 16 | 6.0 | 72 | 1/1/2 | 32 | 1 | 4.594 | 24.316 | 23.728 | 56.870 | 56.870 | 1.000 |
| source_window_control | Token12 | 16 | 6.0 | 72 | 1/1/2 | 4 | 3 | 9.086 | 24.349 | 23.812 | 52.048 | 27.287 | 0.523 |
| source_window_control | Token12 | 16 | 6.0 | 72 | 1/1/2 | 32 | 3 | 9.494 | 24.556 | 23.875 | 50.996 | 22.429 | 0.440 |


## Layer / Token Paired Ratio

TTFT Ratio<1이면 Layer가 빠르다. Goodput Ratio>1이면 Layer가 높다.
Output, QPS, 요청 수, Batch, Window와 Arrival Seed를 일치시킨다.
CI가1을 포함하는 Cell에는 확정 승자 Label을 붙이지 않는다.

| Stage | Cut | Output | QPS | Decode Batch | WIP | Starts | TTFT Ratio | 95% CI | Goodput Ratio | 95% CI |
|---|---|---|---|---|---|---|---|---|---|---|
| batch4_knob_confirmation | 8 | 16 | 3.0 | 4 | 32 | 3 | 1.061 | [0.963, 1.210] | 1.002 | [0.999, 1.003] |
| batch4_knob_confirmation | 8 | 16 | 6.0 | 4 | 32 | 3 | 1.042 | [0.980, 1.135] | 0.909 | [0.812, 1.009] |
| calibration | 2 | 16 | 0.5 | 1 | 32 | 1 | 0.905 | CI 없음 | 1.001 | CI 없음 |
| calibration | 4 | 16 | 0.5 | 1 | 32 | 1 | 0.909 | CI 없음 | 1.000 | CI 없음 |
| calibration | 8 | 16 | 0.5 | 1 | 32 | 1 | 0.970 | CI 없음 | 1.000 | CI 없음 |
| calibration | 12 | 16 | 0.5 | 1 | 32 | 1 | 0.526 | CI 없음 | 1.014 | CI 없음 |
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
| mechanism_discovery | 2 | 16 | 2.0 | 1 | 32 | 1 | 0.637 | CI 없음 | 1.009 | CI 없음 |
| mechanism_discovery | 4 | 16 | 2.0 | 1 | 32 | 1 | 0.882 | CI 없음 | 0.989 | CI 없음 |
| mechanism_discovery | 8 | 16 | 2.0 | 1 | 32 | 1 | 1.046 | CI 없음 | 1.020 | CI 없음 |
| mechanism_discovery | 12 | 16 | 2.0 | 1 | 32 | 1 | 1.025 | CI 없음 | 0.967 | CI 없음 |
| mechanism_discovery | 2 | 16 | 6.0 | 1 | 32 | 1 | 0.848 | CI 없음 | 1.243 | CI 없음 |
| mechanism_discovery | 4 | 16 | 6.0 | 1 | 32 | 1 | 0.932 | CI 없음 | 1.108 | CI 없음 |
| mechanism_discovery | 8 | 16 | 6.0 | 1 | 32 | 1 | 0.973 | CI 없음 | 1.019 | CI 없음 |
| mechanism_discovery | 12 | 16 | 6.0 | 1 | 32 | 1 | 0.962 | CI 없음 | 1.028 | CI 없음 |
| new_QPS_confirmation | 4 | 16 | 3.0 | 2 | 32 | 3 | 0.998 | [0.968, 1.065] | 1.006 | [1.000, 1.015] |
| new_batch_cost_calibration | 8 | 16 | 6.0 | 4 | 32 | 3 | 1.029 | [0.946, 1.082] | 0.956 | [0.854, 1.119] |
| output_residency_counterfactual | 4 | 16 | 6.0 | 2 | 32 | 3 | 1.034 | [1.002, 1.080] | 0.962 | [0.897, 0.999] |
| output_residency_counterfactual | 4 | 64 | 6.0 | 2 | 32 | 3 | 0.982 | [0.968, 0.993] | 1.107 | [1.026, 1.157] |
| residency_discovery | 4 | 64 | 2.0 | 1 | 32 | 1 | 1.003 | CI 없음 | 0.998 | CI 없음 |
| residency_discovery | 8 | 64 | 2.0 | 1 | 32 | 1 | 1.015 | CI 없음 | 0.849 | CI 없음 |
| residency_discovery | 12 | 64 | 2.0 | 1 | 32 | 1 | 1.003 | CI 없음 | 0.997 | CI 없음 |
| residency_discovery | 4 | 64 | 6.0 | 1 | 32 | 1 | 0.989 | CI 없음 | 1.347 | CI 없음 |
| residency_discovery | 8 | 64 | 6.0 | 1 | 32 | 1 | 0.980 | CI 없음 | 1.017 | CI 없음 |
| residency_discovery | 12 | 64 | 6.0 | 1 | 32 | 1 | 0.981 | CI 없음 | 1.016 | CI 없음 |
| residency_discovery | 4 | 200 | 2.0 | 1 | 32 | 1 | 0.990 | CI 없음 | 1.010 | CI 없음 |
| residency_discovery | 8 | 200 | 2.0 | 1 | 32 | 1 | 0.998 | CI 없음 | 1.002 | CI 없음 |
| residency_discovery | 12 | 200 | 2.0 | 1 | 32 | 1 | 1.003 | CI 없음 | 0.997 | CI 없음 |
| residency_discovery | 4 | 200 | 6.0 | 1 | 32 | 1 | 1.002 | CI 없음 | 0.499 | CI 없음 |
| residency_discovery | 8 | 200 | 6.0 | 1 | 32 | 1 | 1.000 | CI 없음 | 0.999 | CI 없음 |
| residency_discovery | 12 | 200 | 6.0 | 1 | 32 | 1 | 0.998 | CI 없음 | 2.004 | CI 없음 |
| source_window_control | 4 | 16 | 6.0 | 2 | 4 | 1 | 1.110 | CI 없음 | 0.905 | CI 없음 |
| source_window_control | 8 | 16 | 6.0 | 2 | 4 | 1 | 0.921 | CI 없음 | 1.036 | CI 없음 |
| source_window_control | 12 | 16 | 6.0 | 2 | 4 | 3 | 0.915 | [0.847, 0.973] | 1.125 | [1.075, 1.196] |
| source_window_control | 4 | 16 | 6.0 | 2 | 32 | 1 | 1.021 | CI 없음 | 0.981 | CI 없음 |
| source_window_control | 8 | 16 | 6.0 | 2 | 32 | 1 | 0.902 | CI 없음 | 1.041 | CI 없음 |
| source_window_control | 12 | 16 | 6.0 | 2 | 32 | 3 | 0.863 | [0.784, 0.923] | 1.563 | [1.080, 2.171] |


## Native-relative Decode5% 검사

ITL p99와 TPOT p95 Ratio의 95% CI 상한이 모두1.05 이하여야 PASS다.
Absolute SLO Pass와 이 검사를 구분한다.

| Stage | Arm | Output | QPS | Batch | Starts | ITL Ratio | 95% CI | TPOT Ratio | 95% CI | 판정 |
|---|---|---|---|---|---|---|---|---|---|---|
| batch4_knob_confirmation | Layer8 | 16 | 3.0 | 4 | 3 | 1.034 | [1.000, 1.072] | 1.044 | [1.040, 1.047] | NOT_ESTABLISHED |
| batch4_knob_confirmation | Token8 | 16 | 3.0 | 4 | 3 | 1.013 | [0.997, 1.032] | 1.042 | [1.038, 1.044] | PASS |
| batch4_knob_confirmation | Layer8 | 16 | 6.0 | 4 | 3 | 1.046 | [1.015, 1.071] | 1.044 | [1.043, 1.045] | NOT_ESTABLISHED |
| batch4_knob_confirmation | Token8 | 16 | 6.0 | 4 | 3 | 1.027 | [1.018, 1.034] | 1.035 | [1.015, 1.047] | PASS |
| decode_batch_counterfactual | Layer8 | 16 | 6.0 | 2 | 3 | 1.034 | [1.023, 1.049] | 1.039 | [1.036, 1.042] | PASS |
| decode_batch_counterfactual | Token8 | 16 | 6.0 | 2 | 3 | 1.024 | [1.013, 1.031] | 1.033 | [1.032, 1.034] | PASS |
| decode_batch_counterfactual | Layer8 | 16 | 6.0 | 4 | 3 | 1.035 | [1.027, 1.045] | 1.040 | [1.037, 1.042] | PASS |
| decode_batch_counterfactual | Token8 | 16 | 6.0 | 4 | 3 | 1.014 | [1.001, 1.025] | 1.042 | [1.038, 1.046] | PASS |
| heldout_confirmation | Layer2 | 16 | 2.0 | 1 | 3 | 1.026 | [1.019, 1.030] | 1.035 | [1.034, 1.035] | PASS |
| heldout_confirmation | Token2 | 16 | 2.0 | 1 | 3 | 1.027 | [1.024, 1.029] | 1.033 | [1.032, 1.033] | PASS |
| heldout_confirmation | Layer4 | 16 | 2.0 | 1 | 3 | 1.032 | [1.024, 1.040] | 1.037 | [1.036, 1.038] | PASS |
| heldout_confirmation | Token4 | 16 | 2.0 | 1 | 3 | 1.028 | [1.023, 1.033] | 1.037 | [1.036, 1.037] | PASS |
| heldout_confirmation | Layer8 | 16 | 2.0 | 1 | 3 | 1.044 | [1.040, 1.047] | 1.040 | [1.039, 1.041] | PASS |
| heldout_confirmation | Token8 | 16 | 2.0 | 1 | 3 | 1.032 | [1.027, 1.035] | 1.039 | [1.038, 1.039] | PASS |
| heldout_confirmation | Layer12 | 16 | 2.0 | 1 | 3 | 1.052 | [1.050, 1.054] | 1.040 | [1.040, 1.041] | NOT_ESTABLISHED |
| heldout_confirmation | Token12 | 16 | 2.0 | 1 | 3 | 1.046 | [1.041, 1.052] | 1.041 | [1.040, 1.042] | NOT_ESTABLISHED |
| heldout_confirmation | Layer2 | 16 | 6.0 | 1 | 3 | 1.029 | [1.021, 1.035] | 1.037 | [1.036, 1.039] | PASS |
| heldout_confirmation | Token2 | 16 | 6.0 | 1 | 3 | 1.030 | [1.025, 1.033] | 1.035 | [1.034, 1.036] | PASS |
| heldout_confirmation | Layer4 | 16 | 6.0 | 1 | 3 | 1.033 | [1.022, 1.039] | 1.039 | [1.037, 1.041] | PASS |
| heldout_confirmation | Token4 | 16 | 6.0 | 1 | 3 | 1.033 | [1.023, 1.039] | 1.039 | [1.037, 1.041] | PASS |
| heldout_confirmation | Layer8 | 16 | 6.0 | 1 | 3 | 1.047 | [1.032, 1.056] | 1.042 | [1.041, 1.043] | NOT_ESTABLISHED |
| heldout_confirmation | Token8 | 16 | 6.0 | 1 | 3 | 1.036 | [1.027, 1.042] | 1.041 | [1.040, 1.041] | PASS |
| heldout_confirmation | Layer12 | 16 | 6.0 | 1 | 3 | 1.054 | [1.041, 1.062] | 1.043 | [1.042, 1.043] | NOT_ESTABLISHED |
| heldout_confirmation | Token12 | 16 | 6.0 | 1 | 3 | 1.049 | [1.040, 1.054] | 1.045 | [1.043, 1.045] | NOT_ESTABLISHED |
| new_QPS_confirmation | Layer4 | 16 | 3.0 | 2 | 3 | 1.033 | [1.024, 1.040] | 1.039 | [1.036, 1.040] | PASS |
| new_QPS_confirmation | Token4 | 16 | 3.0 | 2 | 3 | 1.026 | [1.018, 1.031] | 1.036 | [1.034, 1.037] | PASS |
| output_residency_counterfactual | Layer4 | 16 | 6.0 | 2 | 3 | 1.031 | [1.027, 1.037] | 1.038 | [1.036, 1.040] | PASS |
| output_residency_counterfactual | Token4 | 16 | 6.0 | 2 | 3 | 1.029 | [1.025, 1.036] | 1.037 | [1.035, 1.040] | PASS |
| output_residency_counterfactual | Layer4 | 64 | 6.0 | 2 | 3 | 1.032 | [1.024, 1.039] | 1.036 | [1.035, 1.037] | PASS |
| output_residency_counterfactual | Token4 | 64 | 6.0 | 2 | 3 | 1.030 | [1.026, 1.032] | 1.037 | [1.037, 1.037] | PASS |


## Queue 상태

[analysis/measurement_export_index.json](dense_ar_axis_sweetspot_knob_discovery_20261119/analysis/measurement_export_index.json)와 Start별 Measurement Export에 실제 Arrival QPS를 기록한다.
Warmup을 제외하고 첫 Arrival부터 마지막 Completion까지 모든 요청을 포함한다.
Arrival Window의 Unfinished 증가율과 Drain 시간을 별도로 기록한다.
`FINITE_WINDOW_NO_STRONG_GROWTH`는 장기 안정성을 입증하는 Label이 아니다.
신규 Main Batch1 O16은 N72다. O64/200 Discovery는 N36이다.
이 두 집합을 Output만 바꾼 인과 실험으로 사용하지 않는다.
순수 Output 비교는 별도 Decode2/N72/Output16·64 Paired Seed 실험이다.
여기서 Output cap을 바꾸면 파생된 Runtime Context 상한과 KV Reservation 크기도 바뀐다.
따라서 이 실험은 Output cap의 전체 효과를 검사한다. Decode Residency만의 독립 효과로 해석하지 않는다.

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

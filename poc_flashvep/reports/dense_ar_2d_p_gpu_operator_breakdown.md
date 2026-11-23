# P GPU Operator와 Serial P/D Service 분석

상태: FINAL.

작성 시각: 2026-11-23T12:10:23.364370+00:00.

기준 Commit: `0ae6cd973a4544c6dce13387882479b94858e0d0`. 연구 Branch: `flashvep/dense-ar-p-critical-path-20261123`.

Model: Qwen/Qwen3-8B, Revision `b968826d9c46dd6066d109eabc6255188de91218`.
BI0, Original BF16, Exact Causal Attention, Greedy, Thinking Disabled를 유지합니다.
Full36 Decode와 Prefix Cache Disabled를 유지합니다.
B8192/S1024, GPU당 KV42GiB, KV Block128을 유지합니다.

GPU Pair는 H100 SXM80GB×2입니다. 실제 GPU UUID는 Environment와 각 Raw Runtime 파일에 보존합니다.
Primary QPS는2입니다. C3의 Actual Batch는 동적입니다. B/S를 Batch 수로 표현하지 않습니다.
Client 시각은 Coordinator 전달 시각입니다. HTTP Serving Latency를 측정한 값은 아닙니다.

LongBench400-v2는 변경하지 않습니다. 성능 확인은 동결한 Heldout192를 사용합니다.
Full400 SHA256: `0845a0e44f001acf55dfd16407dc05e80dc56103590a8be3ac772473c1eb6cd9`.
Heldout192 SHA256: `c00eb4dd16deaa319100b47f9b22cc658e8003bdda08b56611ce544dac5e35f7`.
Discovery208과 Heldout192를 구분합니다. Natural400 정합성 검사는 성능 측정이 아닙니다.
Natural EOS와 요청별 독립 C1 길이를 사용한 Equal-output Replay를 구분합니다.
Fixed-output 표의 Quality는 적격 Natural400 구성 검사를 계승합니다. EOS 이후 강제 출력은 새 Natural 정답으로 평가하지 않습니다.
독립 단위는 Server Start입니다. Request와 Token을 독립 Start로 사용하지 않습니다.

동결 SLO는 TTFT≤5000ms, 모든 Client ITL≤34ms, TPOT≤33ms/token입니다.
NativeStrong과 Common-D-policy Native에 대한 ITL p99·TPOT p95의 상대5% Gate를 별도로 검사합니다.
각 상대 비율의 Paired95% CI 상한은1.05 이하여야 합니다. Absolute Decode Attainment는 각 Start에서95% 이상이어야 합니다.

## 실제 C3 P Kernel

아래 값의 분모는 각 Capture의 Request 수입니다.
Kernel Union을 Request 수로 나눈 평균 Resource Demand입니다.
Request의 경과시간이나 Host Queue의 가법 분해가 아닙니다.
GEMM 이름만으로 Q/K/V·O·MLP Projection을 구분하지 못했습니다.
Fused Norm/Residual을 두 비용으로 나누지 않았습니다. Unknown은 그대로 남깁니다.
SM Utilization과 HBM 사용률은 NOT_MEASURED입니다.

| Axis | Start | Evidence | GEMM ms/req | Attention ms/req | RMSNorm ms/req | Fused Norm/Residual ms/req | RoPE ms/req | Unknown ms/req |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Layer | 0 | MEASURED_ONLINE_KERNEL_ACTIVITY | 238.142 | 108.078 | 11.513 | 9.248 | 6.153 | 0.001 |
| StockToken | 0 | MEASURED_ONLINE_KERNEL_ACTIVITY | 196.431 | 70.404 | 0.079 | 7.501 | NOT_RUN / 진행 중 | 3.683 |
| Hybrid | 0 | MEASURED_ONLINE_KERNEL_ACTIVITY | 206.872 | 82.986 | 9.970 | 8.070 | 5.338 | 0.002 |
| Hybrid | 1 | MEASURED_ONLINE_KERNEL_ACTIVITY | 207.549 | 83.052 | 9.988 | 8.074 | 5.349 | 0.002 |
| StockToken | 1 | MEASURED_ONLINE_KERNEL_ACTIVITY | 196.766 | 70.516 | 0.079 | 7.514 | NOT_RUN / 진행 중 | 3.680 |
| Layer | 1 | MEASURED_ONLINE_KERNEL_ACTIVITY | 239.085 | 108.442 | 11.612 | 9.247 | 6.194 | 0.001 |
| StockToken | 2 | MEASURED_ONLINE_KERNEL_ACTIVITY | 196.444 | 70.276 | 0.079 | 7.504 | NOT_RUN / 진행 중 | 3.683 |
| Layer | 2 | MEASURED_ONLINE_KERNEL_ACTIVITY | 238.460 | 108.198 | 11.551 | 9.246 | 6.168 | 0.002 |
| Hybrid | 2 | MEASURED_ONLINE_KERNEL_ACTIVITY | 207.386 | 83.115 | 10.008 | 8.077 | 5.354 | 0.002 |

## Serial Serving Control

실제 Serving API를 사용합니다. 동시에 살아 있는 Request는1개입니다.
P Prefill, D Suffix, D Decode의 실제 Batch1을 검사합니다.
Prompt는 동결한32개입니다. Output은 정확히16개입니다.
이는 Natural EOS나 Poisson Online Capacity 결과가 아닙니다.
Start별 CPC Gate를 실패한 Capture는 DIAGNOSTIC입니다.

| Family | Axis | Start | Evidence | Phase | Host ms | Kernel Union ms | Kernel 미관측 ms |
| --- | --- | --- | --- | --- | --- | --- | --- |
| SERIAL_C3_BATCH1_SERVICE_CONTROL | Layer | 0 | DIAGNOSTIC | D_submit_to_first_worker | 68.264 | 23.274 | 44.990 |
| SERIAL_C3_BATCH1_SERVICE_CONTROL | Layer | 0 | DIAGNOSTIC | D_subsequent_decode | 105.819 | 101.278 | 4.541 |
| SERIAL_C3_BATCH1_SERVICE_CONTROL | Layer | 0 | DIAGNOSTIC | P_arrival_to_ready | 305.284 | 279.378 | 25.906 |
| SERIAL_C3_BATCH1_SERVICE_CONTROL | StockToken | 0 | ISOLATED_KERNEL_MEASUREMENT | D_submit_to_first_worker | 105.998 | 78.411 | 27.586 |
| SERIAL_C3_BATCH1_SERVICE_CONTROL | StockToken | 0 | ISOLATED_KERNEL_MEASUREMENT | D_subsequent_decode | 106.241 | 103.648 | 2.594 |
| SERIAL_C3_BATCH1_SERVICE_CONTROL | StockToken | 0 | ISOLATED_KERNEL_MEASUREMENT | P_arrival_to_ready | 228.439 | 213.259 | 15.180 |
| SERIAL_C3_BATCH1_SERVICE_CONTROL | Hybrid | 0 | ISOLATED_KERNEL_MEASUREMENT | D_submit_to_first_worker | 109.904 | 63.421 | 46.483 |
| SERIAL_C3_BATCH1_SERVICE_CONTROL | Hybrid | 0 | ISOLATED_KERNEL_MEASUREMENT | D_subsequent_decode | 106.458 | 103.332 | 3.126 |
| SERIAL_C3_BATCH1_SERVICE_CONTROL | Hybrid | 0 | ISOLATED_KERNEL_MEASUREMENT | P_arrival_to_ready | 260.094 | 238.074 | 22.020 |
| SERIAL_C3_BATCH1_SERVICE_CONTROL | Hybrid | 1 | ISOLATED_KERNEL_MEASUREMENT | D_submit_to_first_worker | 113.423 | 63.546 | 49.877 |
| SERIAL_C3_BATCH1_SERVICE_CONTROL | Hybrid | 1 | ISOLATED_KERNEL_MEASUREMENT | D_subsequent_decode | 106.016 | 102.994 | 3.022 |
| SERIAL_C3_BATCH1_SERVICE_CONTROL | Hybrid | 1 | ISOLATED_KERNEL_MEASUREMENT | P_arrival_to_ready | 266.267 | 237.708 | 28.559 |
| SERIAL_C3_BATCH1_SERVICE_CONTROL | StockToken | 1 | ISOLATED_KERNEL_MEASUREMENT | D_submit_to_first_worker | 112.510 | 78.475 | 34.035 |
| SERIAL_C3_BATCH1_SERVICE_CONTROL | StockToken | 1 | ISOLATED_KERNEL_MEASUREMENT | D_subsequent_decode | 106.456 | 103.714 | 2.743 |
| SERIAL_C3_BATCH1_SERVICE_CONTROL | StockToken | 1 | ISOLATED_KERNEL_MEASUREMENT | P_arrival_to_ready | 229.909 | 213.342 | 16.567 |
| SERIAL_C3_BATCH1_SERVICE_CONTROL | Layer | 1 | ISOLATED_KERNEL_MEASUREMENT | D_submit_to_first_worker | 62.440 | 22.801 | 39.639 |
| SERIAL_C3_BATCH1_SERVICE_CONTROL | Layer | 1 | ISOLATED_KERNEL_MEASUREMENT | D_subsequent_decode | 105.643 | 101.810 | 3.833 |
| SERIAL_C3_BATCH1_SERVICE_CONTROL | Layer | 1 | ISOLATED_KERNEL_MEASUREMENT | P_arrival_to_ready | 310.035 | 279.561 | 30.475 |
| SERIAL_C3_BATCH1_SERVICE_CONTROL | StockToken | 2 | ISOLATED_KERNEL_MEASUREMENT | D_submit_to_first_worker | 119.474 | 78.353 | 41.121 |
| SERIAL_C3_BATCH1_SERVICE_CONTROL | StockToken | 2 | ISOLATED_KERNEL_MEASUREMENT | D_subsequent_decode | 106.653 | 103.136 | 3.517 |
| SERIAL_C3_BATCH1_SERVICE_CONTROL | StockToken | 2 | ISOLATED_KERNEL_MEASUREMENT | P_arrival_to_ready | 229.108 | 213.049 | 16.059 |
| SERIAL_C3_BATCH1_SERVICE_CONTROL | Layer | 2 | DIAGNOSTIC | D_submit_to_first_worker | 50.105 | 22.174 | 27.931 |
| SERIAL_C3_BATCH1_SERVICE_CONTROL | Layer | 2 | DIAGNOSTIC | D_subsequent_decode | 105.239 | 102.486 | 2.753 |
| SERIAL_C3_BATCH1_SERVICE_CONTROL | Layer | 2 | DIAGNOSTIC | P_arrival_to_ready | 312.525 | 278.892 | 33.632 |
| SERIAL_C3_BATCH1_SERVICE_CONTROL | Hybrid | 2 | ISOLATED_KERNEL_MEASUREMENT | D_submit_to_first_worker | 111.238 | 63.803 | 47.435 |
| SERIAL_C3_BATCH1_SERVICE_CONTROL | Hybrid | 2 | ISOLATED_KERNEL_MEASUREMENT | D_subsequent_decode | 106.256 | 102.721 | 3.535 |
| SERIAL_C3_BATCH1_SERVICE_CONTROL | Hybrid | 2 | ISOLATED_KERNEL_MEASUREMENT | P_arrival_to_ready | 269.796 | 237.494 | 32.302 |
| SERIAL_NOMINAL_ISO_WORK_SERVICE_CONTROL | Layer8 | 0 | ISOLATED_KERNEL_MEASUREMENT | D_submit_to_first_worker | 115.254 | 70.507 | 44.748 |
| SERIAL_NOMINAL_ISO_WORK_SERVICE_CONTROL | Layer8 | 0 | ISOLATED_KERNEL_MEASUREMENT | D_subsequent_decode | 108.216 | 102.888 | 5.329 |
| SERIAL_NOMINAL_ISO_WORK_SERVICE_CONTROL | Layer8 | 0 | ISOLATED_KERNEL_MEASUREMENT | P_arrival_to_ready | 266.330 | 228.597 | 37.733 |
| SERIAL_NOMINAL_ISO_WORK_SERVICE_CONTROL | StockToken | 0 | ISOLATED_KERNEL_MEASUREMENT | D_submit_to_first_worker | 116.711 | 77.900 | 38.811 |
| SERIAL_NOMINAL_ISO_WORK_SERVICE_CONTROL | StockToken | 0 | ISOLATED_KERNEL_MEASUREMENT | D_subsequent_decode | 106.671 | 103.641 | 3.029 |
| SERIAL_NOMINAL_ISO_WORK_SERVICE_CONTROL | StockToken | 0 | ISOLATED_KERNEL_MEASUREMENT | P_arrival_to_ready | 228.474 | 213.165 | 15.309 |
| SERIAL_NOMINAL_ISO_WORK_SERVICE_CONTROL | HybridIso8Approx | 0 | ISOLATED_KERNEL_MEASUREMENT | D_submit_to_first_worker | 127.721 | 79.428 | 48.293 |
| SERIAL_NOMINAL_ISO_WORK_SERVICE_CONTROL | HybridIso8Approx | 0 | ISOLATED_KERNEL_MEASUREMENT | D_subsequent_decode | 106.966 | 102.970 | 3.996 |
| SERIAL_NOMINAL_ISO_WORK_SERVICE_CONTROL | HybridIso8Approx | 0 | ISOLATED_KERNEL_MEASUREMENT | P_arrival_to_ready | 248.885 | 221.271 | 27.614 |

Natural C3의 D Suffix Operator별 기여율은 Serial 결과로 대체하지 않습니다.
직접 계측하지 않은 Natural C3 D Operator 비중은 NOT_MEASURED입니다.

## P RMSNorm Primitive Probe

기존 P Prefix Segment는 CUDA 우선순위를 별도 Scope로 강제합니다.
Runtime-global Native IR 설정만으로 이 임시 Scope를 확인할 수는 없습니다.
새 Probe는 P Scope의 RMSNorm Priority만 바꿉니다. D Code와 작업량은 유지합니다.
Provider별 Cache Directory를 분리합니다. Control Artifact를 Candidate에 재사용하지 않습니다.
Strict8 실패는 보존합니다. 해당 Same-family Replacement를 Primary 성능으로 사용하지 않습니다.
이 검사는 Native 구현의 수학적 타당성을 부정하는 검사가 아닙니다.

## 근거 파일

[configs/research_contract.json](dense_ar_p_critical_path_20261123/configs/research_contract.json)

[analysis/p_critical_causal_summary.json](dense_ar_p_critical_path_20261123/analysis/p_critical_causal_summary.json)

[analysis/c3_credit_confirmation_analysis.json](dense_ar_p_critical_path_20261123/analysis/c3_credit_confirmation_analysis.json)

[analysis/p_geometry_confirmation_analysis.json](dense_ar_p_critical_path_20261123/analysis/p_geometry_confirmation_analysis.json)

[analysis/serial_serving_operator_analysis.json](dense_ar_p_critical_path_20261123/analysis/serial_serving_operator_analysis.json)

[analysis/native_relative_accuracy_analysis.json](dense_ar_p_critical_path_20261123/analysis/native_relative_accuracy_analysis.json)

[analysis/coordinator_stock_compiler_artifact_audit.json](dense_ar_p_critical_path_20261123/analysis/coordinator_stock_compiler_artifact_audit.json)

[analysis/figure_manifest.json](dense_ar_p_critical_path_20261123/analysis/figure_manifest.json)

[analysis/historical_stage_recalculation.json](dense_ar_p_critical_path_20261123/analysis/historical_stage_recalculation.json)

[references/notion_archive_audit.json](dense_ar_p_critical_path_20261123/references/notion_archive_audit.json)

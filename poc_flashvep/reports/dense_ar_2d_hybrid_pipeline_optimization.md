# Hybrid Pipeline과 P/D 작업 재배치 PoC

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

## 기존 DAG

기존 Long-input 경로는 Prefix Upper Layers 후 Suffix Full36을 실행합니다.
이미 적격인 Short-input Fusion은 Suffix Lower Layers 후 전체 Upper Layers를 실행합니다.
이 Fusion은 L≤8192에만 적용합니다. 기존 개선을 새 기여로 계산하지 않습니다.
긴 입력에 Fusion을 강제로 적용해서 B8192를 위반하지 않습니다.
새 Active-decode Insertion은 실행하지 않습니다.

## STRUCTURAL_ACCOUNTING

전체 Layer 수는36입니다. 입력 길이는 L입니다. Upper Layer 수는 m입니다. Token Suffix는 s입니다.
P GEMM Rows는 `(L−s)(36−m)`입니다. D GEMM Rows는 `36L−P Rows`입니다.
P Attention Pair는 `(36−m)(L−s)(L−s+1)/2`입니다.
D Attention Pair는 전체 `36L(L+1)/2`에서 P Pair를 뺀 값입니다.
이는 Work Accounting입니다. GPU 시간이나 측정된 Speedup이 아닙니다.

같은 P GEMM Rows에서 Token-only는 P Attention Pair가 더 작을 수 있습니다.
반대로 Layer는 D Attention과 작은-query Launch 비용을 줄일 수 있습니다.
어느 효과가 TTFT를 지배하는지는 P/D Queue와 Decode Residency에도 의존합니다.

## 새 Geometry Control

첫 후보는 Layer Last2→Last8입니다. Source Credit16과 D Cap2는 유지합니다.
둘째 후보는 Hybrid Last2의 Suffix1/8→3/17입니다. Source Credit8과 D Cap2는 유지합니다.
두 후보는 P 작업을 D로 더 옮깁니다. 동일 작업량의 Kernel 최적화로 표현하지 않습니다.
Hybrid3/17과 Layer8은 명목상 P28L/D8L Rows입니다.
실제16-token Alignment와 추가 실행 Rows는 Raw Work Audit을 사용합니다.

Discovery208의 사전 Screen을 적용한 뒤 Candidate를 동결했습니다.
P 경로5% 감소 또는 Stock 대비 TTFT Point Ratio≤1.05는 후속 확인의 Screen입니다.
이 Screen은 성공 기준이 아닙니다. 최종 우위는 독립 N3 CI와 Decode Gate로 판정합니다.

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

[analysis/p_service_balance_discovery.json](dense_ar_p_critical_path_20261123/analysis/p_service_balance_discovery.json)

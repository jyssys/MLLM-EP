# P Queue와 GPU·Host 경로의 원인 분석

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

## 측정 범위

기존 약1.76초는 과거 N3의 Mean Host 경로 차이입니다.
이 값을 새 Capture의 다른 시각·Pair 결과로 소급 분해하지 않습니다.
새 계측은 같은 Request, 과거 Arrival Seed3개, 같은 Output Work와 GPU0/1을 사용합니다.
Start별 Clean/Profile/Clean을 따로 검사합니다. 모든5개 지연 지표의 차이는5% 이하여야 합니다.

다음 표는 Arrival→P-ready를 P Kernel 실행과 겹친 시간과 나머지 시간으로 나눕니다.
P Kernel 실행에는 앞선 Request와 Shared Batch의 계산도 포함됩니다.
따라서 Kernel과 겹친 대기시간을 해당 Request 자체의 GPU 계산시간으로 표현하지 않습니다.
또한 이 비율은 GPU 계산이 원인인 지연의 인과적 비중이 아닙니다.

| Axis | Start | Evidence | Arrival→P-ready ms | P Kernel과 겹침 ms | P Kernel 미관측 ms | Kernel Union/N ms/request | P Kernel Busy % |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Layer | 0 | MEASURED_ONLINE_KERNEL_ACTIVITY | 2022.195 | 1882.127 | 140.068 | 387.560 | 68.057 |
| StockToken | 0 | MEASURED_ONLINE_KERNEL_ACTIVITY | 747.962 | 737.762 | 10.200 | 289.402 | 52.356 |
| Hybrid | 0 | MEASURED_ONLINE_KERNEL_ACTIVITY | 2575.850 | 1843.688 | 732.162 | 325.841 | 56.313 |
| Hybrid | 1 | MEASURED_ONLINE_KERNEL_ACTIVITY | 3775.970 | 2771.174 | 1004.796 | 326.636 | 61.453 |
| StockToken | 1 | MEASURED_ONLINE_KERNEL_ACTIVITY | 1087.483 | 1074.350 | 13.133 | 289.856 | 56.004 |
| Layer | 1 | MEASURED_ONLINE_KERNEL_ACTIVITY | 3330.444 | 3306.417 | 24.027 | 389.033 | 74.805 |
| StockToken | 2 | MEASURED_ONLINE_KERNEL_ACTIVITY | 661.347 | 650.122 | 11.225 | 289.297 | 54.314 |
| Layer | 2 | MEASURED_ONLINE_KERNEL_ACTIVITY | 2714.287 | 2435.913 | 278.374 | 388.065 | 70.295 |
| Hybrid | 2 | MEASURED_ONLINE_KERNEL_ACTIVITY | 2959.271 | 2018.497 | 940.774 | 326.559 | 58.213 |

P Kernel Busy의 분모는 해당 Cohort 관측 구간입니다. SM Utilization이 아닙니다.
P Queue가 길어도 P Kernel이 관측 구간 전체에서 계속 실행된 것은 아닙니다.
이 사실만으로 즉시 실행 가능한 Decode-safe Headroom이 있다고 주장하지 않습니다.

## Host 경계와 Credit

Arrival→P Submit, Submit→Accept, Accept→Prepare, Prepare→Ready를 따로 보존합니다.
Prepare는 Host 경계입니다. 실제 GPU Kernel 시작 시각이 아닙니다.
Continuous Source에는 Pending Credit Wait의 직접 계측이 없습니다.
빈 `P_wait_segments.jsonl`을 대기시간0으로 처리하지 않습니다.
Arrival→Dispatch와 Full Source Window의 중첩은 관측된 공동 상태입니다.
이 중첩은 독립적인 Credit 지연의 인과적 추정값이 아닙니다.

P Kernel이 없는 구간도 전체 GPU Physical Idle을 뜻하지 않습니다.
DMA는 계측하지 않았습니다. Event Clock 정렬은 Endpoint Anchor와 Drift의 불확실성을 포함합니다.
Single-request Forward Enclosure의 Kernel은 해당 호출과 연관된 Activity입니다.
다른 Stream의 동시 Kernel을 Correlation ID로 제외하지는 못했습니다.
Shared Batch의 Kernel 비용을 Request별로 임의 배분하지 않습니다.

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

## Offered QPS와 순간 도착 집중

| Workload | Start | Offered QPS | 실현된 간격 기준 requests/s | 최소 간격 ms | 직전1초 최대 도착 수 | 동일 Trace |
| --- | --- | --- | --- | --- | --- | --- |
| EqualReplay | 0 | 2.0 | 2.0699 | 1.838 | 7 | True |
| EqualReplay | 1 | 2.0 | 2.6144 | 2.940 | 9 | True |
| EqualReplay | 2 | 2.0 | 1.9466 | 3.019 | 7 | True |
| Natural | 0 | 2.0 | 2.0699 | 1.838 | 7 | True |
| Natural | 1 | 2.0 | 2.6144 | 2.940 | 9 | True |
| Natural | 2 | 2.0 | 1.9466 | 3.019 | 7 | True |

QPS2는 평균 도착률입니다. 동시에2개를 넣는 Batch 설정이 아닙니다.
Poisson Trace에는 짧은 간격의 집중 도착이 있습니다.
느린 P Service는 이런 구간의 Queue를 더 오래 유지할 수 있습니다.
그러나 Arrival Burst만 바꾼 Intervention은 실행하지 않았습니다.
이 표와 Request별 CSV는 도착 집중의 인과적 기여율을 증명하지 않습니다.

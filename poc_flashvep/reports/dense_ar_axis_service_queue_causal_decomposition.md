# GPU Service와 Host Queue의 원인 분리

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

## 핵심 관측

기존 Last8 Token Ready→D 대기 증가의 대부분은 이전 Cohort Guard 구간과 겹친다.
이 Guard는 D Host Submit부터 마지막 Client Completion까지다.
Transfer, Suffix 준비, Full36 Decode와 Delivery를 모두 포함한다.
따라서 이 값만으로 D GPU Compute Saturation을 주장할 수 없다.
기존 해당18개 Confirmation Start에서 Reservation Reject와 공통 Credit Retry는0건이었다.
이는 해당 실행의 주된 대기가 Credit Retry가 아니었다는 근거다. 모든 실패 상황을 배제하지는 않는다.

## 기존 Mean TTFT의 정확한 차이 분해

모든 값은 Layer−Token 밀리초다. P Queue에는 Admission/Host 대기도 들어간다.
각 행의 다섯 구간 합은 Mean TTFT 차이와 같다. 이는 p95 분해가 아니다.

| Cut | Starts | Mean TTFT 차이 | Arrival→P Submit | P Submit→Ready | Ready→D Submit | D Submit→First Worker | Worker→Client |
|---|---|---|---|---|---|---|---|
| 4 | 3 | 337.163 | 357.004 | 4.227 | 3.188 | -36.291 | 9.034 |
| 8 | 3 | -997.537 | 232.787 | 4.548 | -1203.777 | -25.925 | -5.170 |
| 12 | 3 | -1684.885 | -67.317 | 8.868 | -1580.614 | -39.886 | -5.936 |


Last4는 해당 Decode2 Protocol에서 P 단계 부담이 상대적으로 컸다.
Last8/12는 D Guard 부담이 커졌다. 이때 작은 D Service 차이가 반복되어 Queue에 누적될 수 있다.
단계별 항등식만으로 독립적인 인과 효과를 확정하지 않는다.
Stage-balance Replay와 아래 Intervention으로 추가 검사한다.

## 작은 Service 차이의 Queue 증폭

같은 완료 Trace에서 유효 Cycle을 `max(P Service+Gap,D Guard/request+Gap)`로 계산했다.
두 방식의 Cycle 차이에 `0.95×(N−1)`을 곱한 값과 실측 TTFT p95 차이를 비교한다.
모든 차이의 부호는 Layer−Token이다.
다음 계산은 사후 설명용 Deterministic Pipeline 근사다. 새로운 Held-out 예측이나 M/M/1 공식이 아니다.

| Cut | Starts | 유효 Cycle 차이(ms/request) | Cycle×Tail Rank 근사(ms) | 실측 TTFT p95 차이(ms) |
|---|---|---|---|---|
| 4 | 3 | 4.535 | 616.021 | 661.815 |
| 8 | 3 | -14.899 | -2024.071 | -1979.144 |
| 12 | 3 | -23.783 | -3230.939 | -3224.884 |


서비스 차이는 몇~수십 ms/request지만, 과부하 Cohort에서 반복되어 초 단위 Tail 차이가 될 수 있다.
같은 Trace에서 추정한 비용이므로 이 수치 일치만으로 인과관계를 확정하지 않는다.
별도 Discovery→Confirmation Replay와 신규 Batch/Output Intervention을 함께 본다.

## 신규 반복 실측의 Mean TTFT 차이

모든 차이는 Layer−Token 밀리초다. 다섯 구간의 합은 Mean TTFT 차이다.
같은 Batch, Input, Output, QPS, Window와 Arrival Seed를 사용한다.
Output별 상대 우위가 어느 구간에서 바뀌는지 확인한다. 이를 Exclusive Kernel Time 분해로 해석하지 않는다.

| Stage | Cut | Output | QPS | Decode Batch | Starts | Mean TTFT 차이 | Arrival→P | P→Ready | Ready→D | D→First Worker | Worker→Client |
|---|---|---|---|---|---|---|---|---|---|---|---|
| batch4_knob_confirmation | 8 | 16 | 3.0 | 4 | 3 | 45.780 | 92.011 | 7.220 | 0.901 | -38.979 | -15.373 |
| batch4_knob_confirmation | 8 | 16 | 6.0 | 4 | 3 | 186.040 | 231.250 | 6.178 | 5.749 | -41.916 | -15.222 |
| decode_batch_counterfactual | 8 | 16 | 6.0 | 2 | 3 | -495.678 | -52.821 | 3.602 | -418.007 | -23.130 | -5.322 |
| decode_batch_counterfactual | 8 | 16 | 6.0 | 4 | 3 | 146.653 | 197.372 | 6.272 | 5.084 | -46.535 | -15.541 |
| heldout_confirmation | 2 | 16 | 2.0 | 1 | 3 | -281.199 | -4.933 | 1.033 | -251.939 | -25.383 | 0.023 |
| heldout_confirmation | 2 | 16 | 6.0 | 1 | 3 | -996.742 | -24.114 | -0.543 | -943.785 | -28.273 | -0.027 |
| heldout_confirmation | 4 | 16 | 2.0 | 1 | 3 | -269.912 | 8.401 | 4.346 | -271.240 | -11.417 | -0.003 |
| heldout_confirmation | 4 | 16 | 6.0 | 1 | 3 | -194.623 | 250.487 | 8.195 | -448.028 | -5.329 | 0.052 |
| heldout_confirmation | 8 | 16 | 2.0 | 1 | 3 | -292.820 | -11.011 | 0.853 | -264.376 | -18.273 | -0.013 |
| heldout_confirmation | 8 | 16 | 6.0 | 1 | 3 | -494.968 | 335.733 | 8.068 | -824.402 | -14.379 | 0.011 |
| heldout_confirmation | 12 | 16 | 2.0 | 1 | 3 | -342.301 | 0.660 | 6.660 | -336.018 | -13.605 | 0.004 |
| heldout_confirmation | 12 | 16 | 6.0 | 1 | 3 | -795.809 | 224.503 | 7.518 | -1005.773 | -22.069 | 0.011 |
| new_QPS_confirmation | 4 | 16 | 3.0 | 2 | 3 | -22.736 | -0.272 | 0.642 | -0.775 | -31.242 | 8.911 |
| output_residency_counterfactual | 4 | 16 | 6.0 | 2 | 3 | 166.769 | 190.248 | 3.963 | 2.967 | -38.435 | 8.025 |
| output_residency_counterfactual | 4 | 64 | 6.0 | 2 | 3 | -414.753 | -75.379 | 2.868 | -317.513 | -33.151 | 8.422 |


## Output16→64의 공통 통제 실험

Last4, N72, QPS6, 실제 P1/D Suffix1/Decode2를 유지한다. 모든 차이는 Output64−Output16이다.
Runtime Context 상한은8016→8064로 파생된다. 따라서 Output cap의 전체 효과를 검사한다.
Host에서 관측한 준비 비용, Decode Residency, ACK와 대기를 분리한다.
Start별 값과 Paired CI는 별도 JSON에 보존한다.

| Axis | Starts | P Service Δ(ms) | ACK Δ(ms) | D 준비/request Δ(ms) | D Decode/request Δ(ms) | D Guard/request Δ(ms) | Ready→D Δ(ms) | P 앞 대기 Δ(ms) |
|---|---|---|---|---|---|---|---|---|
| Layer | 3 | -3.544 | 0.300 | 1.321 | 558.692 | 560.047 | 15572.097 | 2261.578 |
| Token | 3 | -2.449 | 0.541 | -1.548 | 557.865 | 556.354 | 15892.577 | 2527.205 |


Output16/64에서 전체 GEMM Rows와 Partition은 같다. Decode Work와 KV Lifetime은 달라진다.
Host Decode Residency 증가를 GPU Kernel Time 증가와 동일시하지 않는다.

## Source Outstanding Window32→4

P1, D Suffix1, Decode2와 Partition은 유지한다. Source Outstanding 상한만 변경한다.
Window는 Scheduler B/S나 GPU Batch가 아니다.
다음 값은 W4−W32다. TTFT는 W4/W32 Ratio다.

| Arm | Paired Starts | Ready→D Δ(ms) | 95% CI | P 앞 대기 Δ(ms) | P Service Δ(ms) | D Guard/request Δ(ms) | TTFT p95 Ratio | 95% CI |
|---|---|---|---|---|---|---|---|---|
| Layer4 | 1 | 2.875 | CI 없음 | 287.550 | 8.275 | 2.939 | 1.089 | CI 없음 |
| Layer8 | 1 | -567.079 | CI 없음 | 615.174 | -1.774 | 0.666 | 1.015 | CI 없음 |
| Layer12 | 3 | -2038.397 | [-2362.509, -1859.511] | 2136.777 | 4.794 | 2.523 | 1.039 | [0.917, 1.141] |
| Token4 | 1 | -1.719 | CI 없음 | -19.697 | 3.135 | 1.691 | 1.001 | CI 없음 |
| Token8 | 1 | -1001.384 | CI 없음 | 982.004 | -2.479 | -0.485 | 0.994 | CI 없음 |
| Token12 | 3 | -3118.714 | [-3295.547, -3001.096] | 2886.516 | -1.621 | -6.468 | 0.951 | [0.920, 1.000] |


대기가 D에서 P 앞으로 이동할 수 있다. 대기 위치의 감소를 E2E 성능 향상으로 해석하지 않는다.
Source Window 변화는 Backpressure와 실제 Host Service에도 영향을 줄 수 있다.
서비스 비용이 달라진 Start는 순수 Queue Relocation 실험으로 간주하지 않는다.
Last12의 추가 Seed1950/1951은 Treatment 순서도 바꿨다.

## P 입장 경로의 Reservation ACK Feedback

모든 P Prefix는 D Reservation ACK를 받은 뒤 제출된다.
D Worker는 `engine.step()` 사이에서 Control Message를 처리한다.
따라서 Credit 부족이 없어도 진행 중인 D Step 때문에 ACK가 늦을 수 있다.
실제 GPU 계산과 별개인 이 필수 의존성이 P Critical Path에 들어간다.
요청이 이미 도착했고 Source Window가 막히지 않은 전환에서 P Gap을 세 구간으로 분리했다.
구간은 이전 P Ready→Reservation 시작, Reservation RTT, ACK 수신→다음 P Submit이다.
세 구간 합은 실제 P Gap과 정확히 같다. 이를 GPU Kernel Time으로 부르지 않는다.

| Seed | Axis | Decode Batch | P Submit→Ready(ms) | P Gap(ms) | 그중 ACK(ms) | 유효 P Demand(ms/request) | D Guard(ms/request) | 큰 단계 |
|---|---|---|---|---|---|---|---|---|
| 2026111930 | Layer | 2 | 225.693 | 26.770 | 24.853 | 252.463 | 254.289 | D |
| 2026111930 | Layer | 4 | 234.824 | 21.719 | 19.455 | 256.543 | 167.205 | P |
| 2026111931 | Layer | 2 | 235.119 | 15.547 | 13.442 | 250.666 | 252.031 | D |
| 2026111931 | Layer | 4 | 234.982 | 18.737 | 16.477 | 253.719 | 163.307 | P |
| 2026111932 | Layer | 2 | 234.571 | 17.267 | 15.042 | 251.839 | 264.985 | D |
| 2026111932 | Layer | 4 | 226.600 | 18.597 | 16.446 | 245.197 | 169.097 | P |
| 2026111930 | Token | 2 | 235.555 | 29.610 | 27.730 | 265.165 | 275.861 | D |
| 2026111930 | Token | 4 | 227.096 | 20.846 | 18.175 | 247.941 | 180.687 | P |
| 2026111931 | Token | 2 | 224.397 | 21.145 | 19.232 | 245.542 | 267.342 | D |
| 2026111931 | Token | 4 | 226.568 | 18.782 | 16.515 | 245.350 | 178.352 | P |
| 2026111932 | Token | 2 | 224.627 | 18.824 | 16.948 | 243.451 | 268.139 | D |
| 2026111932 | Token | 4 | 223.926 | 26.817 | 24.873 | 250.743 | 186.249 | P |


[analysis/reservation_feedback_index.json](dense_ar_axis_sweetspot_knob_discovery_20261119/analysis/reservation_feedback_index.json)에 Request별 구간과 D Host Phase별 RTT를 기록했다.
Host Phase는 Idle/Preparation/Decode 구분이다. 실제 Kernel 실행 구간을 직접 측정한 Label은 아니다.
기존 비용표의 Median Gap과50ms 제외 규칙은 이 Bimodal Control 대기를 줄여 추정할 수 있다.
아래 v4 검증은 신규 Calibration에서 비차단 Backlog Gap의 Mean을 사용한다.
이는 GPU WCET나 안전한 Deadline Upper Bound를 뜻하지 않는다.

## 실제 Serving Linear API

`linear_batch_invariant`와 실제 Checkpoint Weight를 사용했다.
기존 `F.linear` 고립 실험과 구분한다. Hopper의 실제 Serving은 Persistent Matmul Kernel을 사용한다.
다음 값은 네 Projection의 고립 Median 합에 Layer 수를 적용한 모델이다.
Attention, RMSNorm, Transfer와 Launch 간 대기를 포함한 전체 Forward 측정값이 아니다.
고립 실험은 Layer0 Weight를 반복한다. 실제 Dense36의 서로 다른 Weight Working Set과 Cache 상태는 복원하지 않는다.

| Cut | Token/Layer Projection 모델 | Layer D 모델(ms) | Token D 모델(ms) |
|---|---|---|---|
| 2 | 1.910 | 11.455 | 21.882 |
| 4 | 1.101 | 22.909 | 25.232 |
| 8 | 1.067 | 45.819 | 48.893 |
| 12 | 1.049 | 68.728 | 72.110 |


Evidence는 `STRUCTURAL_ACCOUNTING`이다. 입력값은 `ISOLATED_KERNEL_MEASUREMENT`의 별도 Operator Timing이다.
Profiler Gate를 통과한 Kernel Profile은 64/72개다.
나머지8개는 작은 O Projection Shape의 `DIAGNOSTIC`이다.
Clean Timing과 Observer Profile을 섞어 Online Exclusive GEMM 시간을 만들지 않는다.

온라인 Event Observer의 네 Cell은 모두5% Gate를 실패했다.
추가 Serial Last4 A/B/A에서는 Layer가 통과했고 Token은 TTFT Gate를 실패했다.
따라서 직접적인 양측 Online Exclusive GPU Breakdown은 `NOT_MEASURED`다.
Serial 실행은 Poisson Serving 결과가 아니다. 이 결과 이후 Profiling을 더 확장하지 않았다.

## 정량적 구조와 한계

동일 GEMM Rows에서도 Token D Attention Pair는 Layer보다 많다.
Token/Layer D Attention Pair Ratio는 Last2/4/8/12에서 약1.9443/1.8888/1.7777/1.6666이다.
전체 P+D Attention Pair는 같다. Attention GPU Time Ratio는 측정하지 않았다.
Matched Aggregate에서 유효 KV Payload는 같다. Layer는 Request당 Hidden+Residual131,072,000bytes를 추가 전송한다.
Layer D는 선택한 m개 Layer의 Weight를 사용한다. Token D는36개 Layer의 서로 다른 Weight를 사용한다.
고유 Projection Weight Footprint는 구조적으로 계산했다. 실제 HBM Read Traffic으로 해석하지 않는다.
실제 Transfer Telemetry는 Start별 Metrics에 있다. API 완료시간을 순수 DMA 시간으로 해석하지 않는다.
SM/Tensor Core Counter, HBM Traffic와 Physical Kernel Idle은 `NOT_MEASURED`다.

[analysis/service_queue_causal_decomposition.json](dense_ar_axis_sweetspot_knob_discovery_20261119/analysis/service_queue_causal_decomposition.json),
[analysis/slot_decomposition_index.json](dense_ar_axis_sweetspot_knob_discovery_20261119/analysis/slot_decomposition_index.json),
[analysis/actual_serving_API_operator_analysis.json](dense_ar_axis_sweetspot_knob_discovery_20261119/analysis/actual_serving_API_operator_analysis.json),
[analysis/matched_geometry_structural_accounting.json](dense_ar_axis_sweetspot_knob_discovery_20261119/analysis/matched_geometry_structural_accounting.json)를 함께 참조한다.

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

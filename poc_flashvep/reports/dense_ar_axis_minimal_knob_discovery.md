# Minimal Knob Discovery와 Held-out 검사

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

## 단일 변수 후보: P/D Pressure Ratio

`R=(1+readyD/b+calibrated_active_D_residual/reference_D_cost)/(1+pendingP)`로 정의한다.
행동 비용은 동결된 Calibration Metadata다. R은 여러 Host Counter를 합친 하나의 Scalar다.
하나의 Sensor만 사용한다는 뜻은 아니다.
기존 Discovery33개 Cell에서 사전 후보 κ0.25/0.5/1/2/4를 비교해 κ4를 동결했다.
이 Training 결과를 Held-out Accuracy로 표현하지 않는다.
R은 대기가 P/D 사이에서 이동하면 달라진다. Queue 위치가 바뀌어도 실제 병목이 같을 수 있다는 반례가 있다.

## 두 변수 후보: Decode Demand와 Total Backlog

첫 변수 `d`는 Calibration Full36 Decode Cohort Residency를 실제 Decode Batch로 나눈 값이다.
두 번째 변수 `n`은 이미 도착했지만 끝나지 않은 요청 수다.
Current P Request, P Pending, D Ready와 Active D를 센다. 미래 Arrival이나 실제 미래 Completion을 읽지 않는다.

각 행동 a의 동결 비용을 `p_a=P Service+P Gap`, `u_a=D Preparation+D Gap/b`로 정의한다.
혼잡 상태의 단순 Cycle 모델은 `C_a=max(p_a,u_a+d)`다.
비혼잡 First-path 모델은 `F_a=p_a+u_a`다.
`α=clip((n-b)/(b+1),0,1)`로 두 값을 연결한다.
점수는 `αC_a+(1−α)F_a`다. 이 함수와 비용 학습 범위는 새 Batch/Output 결과 전에 고정했다.

Token의 p가 낮고 Layer의 u가 낮을 때 모델 경계는 `d*=p_Layer−u_Token`이다.
이는 단순 Max-stage 모델 안의 경계다. 실제 Online TTFT의 정리나 보장된 Deadline이 아니다.
새 Calibration에서는 일부 Cut의 P 우위 부호도 바뀌었다.
차이가1ms 수준이면 Run 간 Host 변동과 구분하기 어렵다. 유리한 부호를 사후에 선택하지 않는다.

`d`는 현재 GPU Idle이 아니다. 이 Runtime의 Whole-cohort Guard Service Demand를 나타낸다.
고정 Output/Batch에서는 d가 거의 고정되므로, 이 변수만으로 Within-workload Dynamic 이득을 주장할 수 없다.

## 더 단순한 비용 균형 Scalar

동결한 Static Metadata Cycle 기준선은 `ΔC(d)=max(p_L,u_L+d)−max(p_T,u_T+d)`로 해석할 수 있다.
ΔC<0이면 Layer, ΔC>0이면 Token을 예측한다. Threshold는0이다.
행동 비용은 Calibration Lookup이다. d는 알려진 Output cap과 실제 Decode Batch에서 추정한 공통 Decode Demand다.
이 Scalar는 현재 GPU Slack을 측정한 값이 아니다. 고정 Workload Metadata와 Calibration에 의존한다.
또한 같은 동질 Workload 안에서 요청별 Axis를 바꾸지 않는다.
따라서 이 기준선의 성공은 간단한 Cross-workload 비용 설명을 지지할 수 있다. State-aware Dynamic의 독립 기여를 입증하지는 않는다.
아래 Table에서 원래 두 규칙과 같은 Held-out Cell을 비교한다.

## 기록된 실제 의사결정 상태의 검증

기존 Static 실행의 P 제출 전 상태에서 두 규칙의 Shadow Vote를 계산한다.
이 Vote는 실제 실행 Axis를 바꾸지 않았다.
Cell 전체의 다수 Vote를 하나의 Fixed-axis 예측으로 평가한다.
Request별 Vote 수를 독립 검증 표본 수로 사용하지 않는다.
CI가1을 포함하는 Cell은 `UNRESOLVED`로 둔다.
Always Layer, Always Token과 Static Metadata Cycle 규칙도 같은 Cell에서 검사한다.
다수 Cell의 승자가 같다는 이유만으로 State-aware 규칙의 기여를 주장하지 않는다.
Static Metadata 규칙은 같은 동질 Workload 안에서 상태에 따라 Axis를 바꾸지 않는다.

| Rule | 확정 Label Cell | 정답 Cell | 평균 Fixed-axis TTFT Regret(%) |
|---|---|---|---|
| one | 8 | 7 | 1.112 |
| two | 8 | 8 | 0.889 |
| AlwaysLayer | 8 | 7 | 1.112 |
| AlwaysToken | 8 | 1 | 6.598 |
| StaticMetadataCycle | 8 | 8 | 0.900 |

| Stage | Cut | Output | QPS | Batch | Starts | 측정 Label | One 예측 | Two 예측 | One Regret(%) | Two Regret(%) |
|---|---|---|---|---|---|---|---|---|---|---|
| batch4_knob_confirmation | 8 | 16 | 3.0 | 4 | 3 | UNRESOLVED | Layer | Layer | 6.067 | 6.067 |
| batch4_knob_confirmation | 8 | 16 | 6.0 | 4 | 3 | UNRESOLVED | Layer | Layer | 4.183 | 4.183 |
| decode_batch_counterfactual | 8 | 16 | 6.0 | 2 | 3 | Layer | Layer | Layer | 0.000 | 0.000 |
| decode_batch_counterfactual | 8 | 16 | 6.0 | 4 | 3 | UNRESOLVED | Layer | Layer | 3.078 | 3.078 |
| heldout_confirmation | 2 | 16 | 2.0 | 1 | 3 | Layer | Layer | Layer | 0.000 | 0.000 |
| heldout_confirmation | 4 | 16 | 2.0 | 1 | 3 | UNRESOLVED | Layer | Layer | 0.000 | 0.000 |
| heldout_confirmation | 8 | 16 | 2.0 | 1 | 3 | Layer | Layer | Layer | 0.000 | 0.000 |
| heldout_confirmation | 12 | 16 | 2.0 | 1 | 3 | UNRESOLVED | Layer | Layer | 0.000 | 0.000 |
| heldout_confirmation | 2 | 16 | 6.0 | 1 | 3 | Layer | Layer | Layer | 0.000 | 0.000 |
| heldout_confirmation | 4 | 16 | 6.0 | 1 | 3 | UNRESOLVED | Layer | Layer | 0.000 | 0.000 |
| heldout_confirmation | 8 | 16 | 6.0 | 1 | 3 | Layer | Layer | Layer | 0.000 | 0.000 |
| heldout_confirmation | 12 | 16 | 6.0 | 1 | 3 | Layer | Layer | Layer | 0.000 | 0.000 |
| new_QPS_confirmation | 4 | 16 | 3.0 | 2 | 3 | UNRESOLVED | Layer | Layer | 0.000 | 0.000 |
| output_residency_counterfactual | 4 | 16 | 6.0 | 2 | 3 | Token | Layer | Token | 3.354 | 0.000 |
| output_residency_counterfactual | 4 | 64 | 6.0 | 2 | 3 | Layer | Layer | Layer | 0.000 | 0.000 |


Regret은 예측된 Fixed Axis와 측정된 Paired Cell 평균을 비교한다.
이를 Online Axis Switching의 성능 손실로 해석하지 않는다.
State-updating Rule Replay는 별도 Oracle 보고서에 있다.
Regret은 실측 Cell 평균에 대한 점 추정치다. 미확정 Cell의 작은 Regret은 입증된 손실이 아니다.
Qualified Token/s의 승자와 Regret도 별도 JSON에 기록했다. 모든 방법의 Goodput이0이면 비율 손실은 정의하지 않는다.
Prompt는 재사용했다. 새 Arrival Seed/QPS/Output/Batch 검증과 새로운 Prompt 분포 검증은 다르다.
Held-out Prompt Distribution과 Input Length Generalization은 `NOT_RUN`이다.
1~2변수가 모든 현상을 설명한다는 주장은 아직 별도 반례와 모델 검증을 필요로 한다.

[analysis/recorded_observable_knob_validation.json](dense_ar_axis_sweetspot_knob_discovery_20261119/analysis/recorded_observable_knob_validation.json),
[configs/frozen_single_pressure_rule_v1.json](dense_ar_axis_sweetspot_knob_discovery_20261119/configs/frozen_single_pressure_rule_v1.json),
[configs/frozen_action_service_costs_v2.json](dense_ar_axis_sweetspot_knob_discovery_20261119/configs/frozen_action_service_costs_v2.json),
[state/minimal_knob_protocol_v2_before_new_batch_output.json](dense_ar_axis_sweetspot_knob_discovery_20261119/state/minimal_knob_protocol_v2_before_new_batch_output.json)에 정의와 입력을 보존한다.

## 비용표 외삽 실패와 별도 v3 검증

v2는 Decode4에서 기존 Decode2의 P/Suffix 비용을 재사용했다.
첫 두 새 Seed에서는 Token P Service가 예측보다 약12ms 짧았다. 이 오차가 P 단계 순위를 바꿨다.
따라서 비용표 외삽 실패와 상태 변수의 실패를 분리하는 추가 실험을 선언했다.
Batch4 새 Calibration6개 Start에서만 별도 v3 비용표를 동결한다.
QPS3/6의 새 Seed1970/1971/1972를 그 이후에 평가한다.
규칙 함수, κ와 SLO는 바꾸지 않는다. v2의 실패 결과도 유지한다.
두 비용표가 같은 신규 Raw Trace를 예측한다. 이를 서로 다른 독립 Server 표본으로 더하지 않는다.
추가 v4는 같은 신규 Calibration만 사용한다. P Gap을 Median/50ms 절단값 대신 비차단 Backlog 전환의 Mean으로 추정한다.
이 변경은 신규 Calibration 전에 별도로 선언했다. v2/v3 파일은 변경하지 않았다.
v4는 새 Confirmation의 첫 측정 Arrival 전에 동결됐는지 Monotonic Clock으로 검사한다.
새 비용표들은 동일한 두 규칙을 평가한다. 이 이후에는 규칙이나 비용표를 다시 튜닝하지 않는다.

| Cost Model | QPS | Batch | Starts | TTFT Label | One 예측 | Two 예측 | One TTFT Regret(%) | Two TTFT Regret(%) | Goodput Label |
|---|---|---|---|---|---|---|---|---|---|
| 3 | 3.0 | 4 | 3 | UNRESOLVED | Layer | Layer | 6.067 | 6.067 | UNRESOLVED |
| 3 | 6.0 | 4 | 3 | UNRESOLVED | Layer | Token | 4.183 | 0.000 | UNRESOLVED |
| 4 | 3.0 | 4 | 3 | UNRESOLVED | Layer | Layer | 6.067 | 6.067 | UNRESOLVED |
| 4 | 6.0 | 4 | 3 | UNRESOLVED | Layer | Layer | 4.183 | 4.183 | UNRESOLVED |


[analysis/recorded_observable_knob_validation_v3.json](dense_ar_axis_sweetspot_knob_discovery_20261119/analysis/recorded_observable_knob_validation_v3.json),
[analysis/frozen_model_step_and_completion_errors_v3.json](dense_ar_axis_sweetspot_knob_discovery_20261119/analysis/frozen_model_step_and_completion_errors_v3.json),
[configs/frozen_action_service_costs_v3.json](dense_ar_axis_sweetspot_knob_discovery_20261119/configs/frozen_action_service_costs_v3.json)는 실행된 경우에만 존재한다.
v4의 대응 파일은 동일한 이름의 `_v4`/`v4` 파일이다.
Budget 때문에 해당 Phase가 미실행이면 `NOT_RUN`이다.

## Calibration 비용 순위의 불확실성

아래 값은 새 Batch4 Calibration의 Paired Seed3개만 사용한다. 부호는 Layer−Token이다.
P Gap은 이미 도착했고 Source Window가 막히지 않은 전환의 Mean이다.
각 비용의 별도 Median으로 만든 Lookup과 Paired Mean 차이는 서로 다를 수 있다.
CI가0을 포함하면 그 비용 순위를 확정하지 않는다. 이 분석으로 동결 모델이나 규칙을 다시 맞추지 않는다.

| 비용 항목 | Paired Mean 차이(ms) | 95% CI(ms) |
|---|---|---|
| P_service_ms | 4.946 | [0.892, 12.218] |
| uncensored_backlog_gap_ms | -1.140 | [-5.185, 3.855] |
| ACK_part_ms | -1.155 | [-5.101, 3.837] |
| D_preparation_ms | -13.388 | [-17.149, -10.089] |
| D_cohort_residence_per_request_ms | 0.274 | [-0.193, 0.725] |
| effective_P_demand_ms | 3.807 | [-4.293, 10.129] |
| effective_D_demand_ms | -13.113 | [-17.342, -9.364] |


[analysis/new_calibration_cost_rank_uncertainty.json](dense_ar_axis_sweetspot_knob_discovery_20261119/analysis/new_calibration_cost_rank_uncertainty.json)에 각 Start의 값과 출처를 보존한다.
이는 비용 Calibration의 불확실성이다. Held-out 예측 성공이나 GPU Kernel 분산을 뜻하지 않는다.

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

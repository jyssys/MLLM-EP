# 연구 의사결정 기록

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

Observe→Hypothesize→Discriminate→Measure→Update 순서로 기록했다.

## A0001 — OBSERVE

가설: Source/Destination effective service imbalance is amplified by the whole-cohort guard.

경쟁 가설: KV reservation blocking, Partner barrier, Transfer/import timing

최소 실험: NOT_RECORDED

사전 예측: NOT_RECORDED

관측: Last8 Token minus Layer mean Ready-to-D1203.777077ms, with no reservation rejections in the18 baseline starts.

다음 선택: NOT_RECORDED

Evidence: TRACE_ANALYSIS


## A0002 — DESIGN

가설: NOT_RECORDED

경쟁 가설: NOT_RECORDED

최소 실험: Source Outstanding Window32 versus4 at fixed P1/D-preparation1/Decode2, Last8, Output16.

사전 예측: A queue-location effect reduces Ready-to-D while moving waiting before P; total TTFT need not improve.

관측: NOT_RECORDED

다음 선택: Run exact Last2 state qualification and low-load action calibration before selecting new held-out cells.

Evidence: NOT_RECORDED


## A0003 — STARTUP_CONTRACT_RECOVERY

가설: Native Full36 요청도 초기 Extension 생성자에는 유효한 Placeholder Cut이 필요하다.

경쟁 가설: Native 모델 계산 또는 출력 경로가 잘못됐다.

최소 실험: Native 초기 Cut을 기존 유효값으로 유지한다. CPU 회귀 검사 후 새 v2 이름으로 N8/N32를 실행한다.

사전 예측: 모델 계산을 바꾸지 않아도 시작 검사가 통과한다.

관측: CPU 4개 검사가 통과했다. Native N8/N32 State Gate가 통과했다. v1 실패 기록은 보존했다.

다음 선택: 실제 요청의 Full36와 Prefix/Suffix 방문 검사를 유지한다.

Evidence: CORRECTNESS_ONLY


## A0004 — REPLAY_VALIDATION

가설: 유효 P/D Service Demand와 Cohort Guard가 주요 Queue 증폭을 설명할 수 있다.

경쟁 가설: 관측하지 못한 KV/Host 상태가 대부분의 TTFT 차이를 지배한다.

최소 실험: 기존 Discovery에서 비용을 추정한다. 별도 Confirmation 54개 Start를 Queue 상태 갱신 Replay로 예측한다.

사전 예측: TTFT p95 오차10%, Token/s 오차5%, Ready→D 오차max(10ms,20%) 기준을 검사한다.

관측: 48/54가 세 기준을 모두 통과했다. Batch2 Last8/12의 일부 대기시간 예측은 실패했다.

다음 선택: 새 Seed 검증 전 비용 모델을 동결한다. 실패한 Cell에서 Oracle을 검증된 Serving 이득으로 주장하지 않는다.

Evidence: OFFLINE_REPLAY


## B0001 — LAST2_QUALIFICATION

가설: 작은 Last2도 기존 Exact Partial Forward와 Token Prefix 계약을 그대로 사용할 수 있다.

경쟁 가설: 432/448 Token 경계 또는 Layer2 State Handoff에서 정합성이 깨진다.

최소 실험: Layer2와 Token432/448을 N8 다음 N32에서 검증한다.

사전 예측: Original Greedy Output, 전체 Prompt KV, Hidden, Layer/Query 방문 및 ACK/Lifetime이 통과해야 한다.

관측: Layer2와 Token2 모두 N8/N32 State Gate를 통과했다. N8/N32는 Aggregate Matched Rows 성능 실험이 아니다.

다음 선택: N36/72의 정확한 GEMM Rows와 Batch1 비용을 측정한다.

Evidence: CORRECTNESS_ONLY


## B0002 — LOW_LOAD_HOST_GAP

가설: Token Last12의 큰 TTFT Tail 일부는 P 제출 이전의 Host 경계에서 발생했다.

경쟁 가설: D GPU Forward 자체가 약1초 느려졌다.

최소 실험: 동일 Request의 Arrival, Decision, KV Reservation, P Forward, D Forward 경계를 비교한다.

사전 예측: D Forward 원인이면 D 제출 이후 비용이 크게 늘어야 한다.

관측: q34는 Arrival→P 제출818.38ms였다. Decision은 Arrival 후816.63ms였다. KV 협상0.85ms, P Service191.61ms, D 제출→첫 Worker135.47ms였다. Host Gap의 하위 원인은 계측하지 못했다.

다음 선택: 점과 원시 기록을 보존한다. 반복 검증과 Predictor 오차에서 이 예외를 제외하지 않는다. Kernel 비용 차이로 해석하지 않는다.

Evidence: TRACE_ANALYSIS


## C0001 — SOURCE_WINDOW_INTERVENTION

가설: 긴 Ready→D 대기는 Source/D Queue 위치와 Guard의 영향을 크게 받는다.

경쟁 가설: 그 대기 자체가 Token의 GPU 계산 또는 DMA 비용이다.

최소 실험: Last8에서 동일 Batch와 Partition을 유지한다. Source Window만32에서4로 줄인다.

사전 예측: 대기가 P 쪽으로 이동하면 Ready→D 감소만큼의 TTFT 개선이 나오지 않을 수 있다.

관측: Token Ready→D 평균은1001.38ms 감소했다. P 제출 전 평균 대기는982.00ms 증가했다. TTFT p95 비율은0.99362였다. 각 설정은1Start다.

다음 선택: Last4 음성 대조군과 Last12 양성 대조군을 측정한다. Service 변화와 대기 위치 변화를 구분한다.

Evidence: MEASURED_ONLINE


## C0002 — NEGATIVE_CONTROL_AND_SERVICE_DRIFT

가설: Queue 위치 외에 유효 Service 시간의 변동이 Tail을 증폭할 수 있다.

경쟁 가설: Window 변경은 항상 대기 위치만 바꾸며 처리율에는 영향이 없다.

최소 실험: 동일 Source-window 통제를 Last4와 Last12에 적용한다.

사전 예측: Service가 같아야 순수 대기 위치 이동으로 해석할 수 있다.

관측: Last4 Layer의 P Service 평균이257.57에서265.84ms로 바뀌었다. Worker 수락 전 비용은3.90에서7.95ms였다. 단일 Start로 Window 효과와 실행 변동을 구분하지 못했다.

다음 선택: Last12를 두 새 Seed에서 재현한다. 단일 Start의 동등성이나 순수 Queue 효과를 선언하지 않는다.

Evidence: TRACE_ANALYSIS


## C0003 — PROFILER_STOP_RULE

가설: 작은 계측 변화가 Queue 근처에서 TTFT Tail을 증폭할 수 있다.

경쟁 가설: 모든 CUDA Event 결과를 Clean Serving GPU 시간으로 사용할 수 있다.

최소 실험: 고정 Seed의 A/B/A를 Layer4,Token4,Layer8,Token8에 실행한다.

사전 예측: Observed 변화와 Clean-control Drift가 Metric별5% 이내여야 한다.

관측: 네 Online Gate가 모두 실패했다. 고립 Kernel Profile은64/72가 통과했다. 작은 O Projection8개는 Diagnostic이다.

다음 선택: Last4 한 쌍에 Serial Control만 추가한다. 이 마지막 대안도 실패하면 추가 Profiling을 중단한다.

Evidence: DIAGNOSTIC


## D0001 — MINIMAL_KNOB_REVISION

가설: 공통 Decode 점유시간/Actual Batch가 P-bound와 D-bound의 축 선택을 구분할 수 있다.

경쟁 가설: P/D Queue Count 비율 하나가 Admission Policy와 무관하게 모든 축 우위를 설명한다.

최소 실험: d=공통 Decode Cohort Residency/Batch를 정의한다. 전체 미완료 요청 수를 두 번째 부하 신호로 비교한다.

사전 예측: Last8 Decode2→4와 Last4 Output16→64에서 Stage Balance에 따른 우위 전환을 예측한다.

관측: 새 Batch/Output 확인 결과를 보기 전에 규칙 v2를 동결했다. 기존 단일 Pressure Rule v1도 보존했다.

다음 선택: 실제 새 Seed와 Batch/Output에서 검증한다. Oracle 이득과 모델 오차를 함께 보고한다.

Evidence: RESEARCH_PROTOCOL


## C0004 — OBSERVER_STOP

가설: 고정 Batch의 실제 Serving CUDA Event 계측은 Tail을 교란할 수 있다.

경쟁 가설: Serial 실행에서는 두 방식 모두 5% Observer Gate를 통과할 수 있다.

최소 실험: 온라인 네 Cell 이후 Serial Last4 Layer/Token을 각각 A/B/A로 한 번만 검사한다.

사전 예측: 양쪽이 통과해야 직접적인 계측 Forward 비교를 적격 결과로 사용한다.

관측: 온라인 4/4 Cell은 Gate를 실패했다. Serial Layer는 통과했지만 Token TTFT p95는 대조군 평균보다 6.70% 낮아 Gate를 실패했다.

다음 선택: 계측을 더 확장하지 않는다. 독립 Kernel 실험과 Host 경계 분석을 구분한다.

Evidence: DIAGNOSTIC


## D0002 — FRESH_COST_MODEL_FREEZE

가설: Decode Demand와 P/D Service Balance가 Axis 우위를 설명할 수 있다.

경쟁 가설: 고정된 작은 Cut/큰 Cut 구분 또는 GEMM 비용만으로 충분하다.

최소 실험: 새 Discovery 30개 Start에서 비용을 동결한다. Decode2/4와 Output16/64를 새 Seed에서 검사한다.

사전 예측: 새 비용에서 Last8은 Layer P 비용도 약간 낮다. 따라서 단순 모델은 Decode4에서 반드시 Token 역전을 예측하지 않는다. 이전 자료 기반 역전 가설과 구분한다.

관측: 동결 비용의 Last8 Decode2 유효 P는 Layer238.60ms, Token239.34ms다. D 준비 비용은 Layer88.01ms, Token99.59ms다. 이 차이는 Host 경계를 포함한다.

다음 선택: 새 Seed의 비용 변화와 예측 실패를 그대로 보존한다. 평가 결과로 동결 비용이나 규칙을 다시 맞추지 않는다.

Evidence: STRUCTURAL_ACCOUNTING


## S0001 — BOOTSTRAP_CDF_CORRECTION

가설: Exact Bootstrap의 재표본 확률질량을 유지해야 한다.

경쟁 가설: 27개 Exact Draw에 선형 분위수 보간을 적용해도 동일한 CI가 나온다.

최소 실험: 독립 Start3개의 모든27개 재표본을 검사한다. 극단 Draw의 확률1/27과 Tail2.5%를 비교한다.

사전 예측: 확률3.70%인 최대 Draw를97.5% 분위수에서 제거하면 안 된다.

관측: 기존 보간 CI 상한0.98547은 신규 empirical-CDF에서1.00002가 됐다. 원시 값은 같다.

다음 선택: 신규 분석 전체에 같은 수정 방법을 적용한다. 이전 파생 Index를 별도로 보존한다.

Evidence: STATISTICAL_METHOD_CORRECTION


## D0003 — BATCH_AWARE_COST_DISCRIMINATOR

가설: 새 Batch에서의 예측 실패는 비용표 외삽 오차일 수 있다.

경쟁 가설: 같은 비용표를 보정해도1~2변수의 상태 표현이 충분하지 않다.

최소 실험: Batch4 새 Calibration6개 후 비용표v3만 동결한다. 새 Seed1970/1971/1972의 QPS3·6에서 원래 규칙을 검사한다.

사전 예측: Batch-aware 비용표가 P 단계 순위를 복원한다면 고부하 Cycle 예측이 개선될 수 있다. 저부하에는 First-path와 Backlog가 중요할 수 있다.

관측: 기존 v2의 첫 두 Batch4 실행에서 Token P Service는 약226~227ms다. v2는 약239ms를 사용했다.

다음 선택: 기존 규칙과 v2 결과를 변경하지 않는다. v3의 새 Held-out 검증과 잔여 예산을 검사한다.

Evidence: TRACE_ANALYSIS


## C0005 — MANDATORY_RESERVATION_FEEDBACK

가설: D 예약 ACK는 Credit이 충분해도 P 입장 경로를 지연할 수 있다.

경쟁 가설: 예약 거절0이면 P 대기는 모두 Forward 또는 Source Window 때문이다.

최소 실험: Source1의 Ready→다음 Send를 Reservation 전·RTT·이후로 분리한다. 이미 도착했고 Window가 차단되지 않은 경우만 집계한다.

사전 예측: ACK가 필수 선행 조건이면 RTT 구간이 P Gap에 정확히 포함돼야 한다.

관측: 새 Seed1930의 Last8에서 ACK는 P Gap21~30ms 중18~28ms다. Gap 합 항등식이 통과했다. D 논리 Phase별 RTT도 달랐다.

다음 선택: 새 Calibration 전에 v4의 Mean Gap 추정 방식을 선언한다. 같은 새 Seed에서 v2/v3/v4를 평가한다. 추가 GPU Arm은 만들지 않는다.

Evidence: TRACE_ANALYSIS


## D0004 — THIRD_SEED_REVERSAL_RETAINED

가설: Decode4에서 Token이 우수할 수 있다.

경쟁 가설: Host ACK와 Source Service 변동 때문에 방향이 재현되지 않을 수 있다.

최소 실험: 미리 선언한 세 번째 Seed를 그대로 실행한다.

사전 예측: 세 Seed의 Paired CI가1을 넘는지 확인한다.

관측: Decode4 세 번째 Seed는 Layer가 빨랐다. 통합 TTFT Ratio Layer/Token1.03078, CI[0.95731,1.11749]로 역전은 미확정이다. Decode2는0.88837, CI[0.80939,0.97790]다.

다음 선택: 반례를 제외하지 않는다. 별도 새 Calibration 및 Held-out은 비용 오차와 상태 표현을 구분하는 검증으로만 사용한다.

Evidence: MEASURED_ONLINE


## D0005 — CLASS_IMBALANCE_CONTROLS

가설: 간단한 상태 규칙은 Fixed Axis보다 유용할 수 있다.

경쟁 가설: 대부분의 Cell에서 같은 Axis가 이기면 높은 Accuracy는 상수 규칙으로도 나온다.

최소 실험: Always Layer/Token과 Static Metadata Cycle을 같은 Held-out Cell의 Baseline으로 추가한다.

사전 예측: State-aware 규칙은 Accuracy뿐 아니라 Fixed-axis TTFT/Goodput Regret도 줄여야 한다.

관측: 새 Seed1970/1971/1972 이전에 Baseline을 선언했다. 기존 규칙이나 κ는 변경하지 않았다.

다음 선택: 미확정 Cell을 확정 승자로 라벨링하지 않는다. 성능과 모델 복잡도를 함께 평가한다.

Evidence: TRACE_ANALYSIS


## D0006 — FROZEN_CONTROL_MODEL_FALSIFICATION

가설: ACK 경계를 명시하면 고정 Mean Gap보다 완료시간을 더 잘 예측할 수 있다.

경쟁 가설: 실제 Async Step 경계를 요청당 Preparation 하나로 근사하면 ACK 대기를 과대평가할 수 있다.

최소 실험: 새 Calibration6개만 사용한 C1을 새 Seed1970의 QPS3·6 Fixed Layer/Token에 적용한다.

사전 예측: 두 Fixed Axis의 Timing과 SLO Gate가 통과해야 Mixed Oracle을 해석한다.

관측: 첫 Seed의 C1은 Layer2개 Gate를 통과했지만 Token2개 Timing Gate를 실패했다. Token ACK 예측의 Signed Mean 오차는 약18~19ms였다. C1 Mixed Search는 두 Trace 모두 NOT_RUN이다. v4 Mean-gap 모델은 같은 Seed의 QPS6 두 Axis를 통과했지만 QPS3 Layer를 실패했다.

다음 선택: 새 비용이나 경계를 다시 맞추지 않는다. 미리 선언한 나머지 독립 Seed를 끝낸다. CPU Calibration의 Effective P 순위 CI도 보존한다.

Evidence: OFFLINE_REPLAY


## C0006 — MATCHED_OUTPUT_CAP_CROSSOVER

가설: 같은 Cut에서도 P/D Service Balance가 달라지면 Axis 우위가 바뀔 수 있다.

경쟁 가설: GEMM Shape 또는 작은 Cut이라는 사실만으로 Online TTFT 우위가 고정된다.

최소 실험: Last4/Token 약889, Input8000, N72, QPS6, 실제1/1/2를 유지한다. Output16·64를 새 Paired Seed3개에서 비교한다.

사전 예측: 긴 Output이 D Cohort Demand를 증가시키면 D 준비 비용이 낮은 Layer의 상대적 이점이 커질 수 있다.

관측: Output16 TTFT Layer/Token1.03354 CI[1.00213,1.08012]로 Token이 우세했다. Output64는0.98175 CI[0.96787,0.99267]로 Layer가 우세했다. Output64 Qualified token/s Ratio는1.10680 CI[1.02558,1.15709]였다. 네 Native-relative5% 검사가 모두 통과했다. 이들은 Finite-cohort 결과다. Runtime Context cap도 함께8016에서8064로 바뀐다.

다음 선택: 새 QPS와 나머지 Held-out Seed를 완료한다. 동결한 두 변수 규칙과 동일 비용의 Static Metadata Cycle을 비교한다. Queue와 Host 준비 구간을 GPU 시간으로 표현하지 않는다.

Evidence: MEASURED_ONLINE


## E0001 — SCIENTIFIC_STOP_AND_PUBLICATION

가설: 최소 비용 균형 지표가 현재 Runtime의 Axis 우위를 설명할 수 있다.

경쟁 가설: 새 Dynamic Controller가 Strong Fixed Axis보다 큰 순이익을 낼 수 있다.

최소 실험: 선언한241개 성공 Dispatcher를 완료했다. 새 Seed15개 Paired Cell의 승자, Regret와 상태 갱신 Oracle을 검사한다.

사전 예측: 복잡한 Controller 개발 전 검증된5% Headroom과 정확한 Fixed-trace 비용 예측이 필요하다.

관측: 확정 Cell8개에서 두 변수 규칙과 Static Metadata Cycle 모두8개를 맞췄다. 나머지7개는 미확정이다. 모든 Fixed Gate를 통과한 세 Start Fine-search Cell3개의 모델 Headroom은 최대0.04601%였다. C1은12개 Fixed Trace 중7개를 통과했다. 이러한 값은 유한 Search이며 전역 상한이나 실제 Dynamic Speedup이 아니다.

다음 선택: QPS8·10과 추가 Output 반복으로 같은 과부하 현상을 다시 찾는 대신 분석을 마무리한다. 새 GPU Grid를 추가하지 않는다. 보고서, Lossless Archive, Remote Push와 Burner Closeout을 검증한다.

Evidence: TRACE_ANALYSIS_AND_OFFLINE_ORACLE


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

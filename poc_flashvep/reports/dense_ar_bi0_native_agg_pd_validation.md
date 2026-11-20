# BI0 Native PD / Agg DP2 검증

상태: `BI0_REFERENCE_VALIDATED_WITH_SCOPED_NUMERICAL_CONTRACT`. Primary는 BI0다. 과거 BI1 성능을 새 BI0 표에 섞지 않는다. Cross-geometry Task Quality 통과는 전체200개 Strict Greedy 일치가 아니다. Clean GPU-exclusive 분해와 Active-decode Window 검증은 별도 미완료 Gate다.

## 실행 계약

Qwen3-8B Revision은`b968826d9c46dd6066d109eabc6255188de91218`이다. GPU0/1의 H100 SXM80GB를 사용한다. Original BF16, Greedy, Thinking Disabled와 Natural EOS를 유지한다. 모든 Engine에 B8192/S1024와 KV42GiB를 명시했다. Prefix Cache는 비활성화했다.

Native는 공식 NIXL Connector와 일반 Continuous Scheduler를 사용한다. P는 Full36 Prefill을 수행한다. D는 마지막 Prompt Query1개를 다시 실행한다. 이후 Full36 Decode를 수행한다. P의 첫 Sample은 버린다. 기존 Cohort Guard와 Custom Reservation ACK는 이 기준선에 없다.

Agg는 Full-model Replica2개를 사용한다. Round-robin으로 분배한다. 기본 Chunked Prefill과 Continuous Batching을 유지한다. Primary 실행은 기본 Compile3/Async 설정을 사용한다. 실제 설정과 Batch Geometry는 각 Raw Folder에 저장한다.

## N8 상태 검사

`CORRECTNESS_ONLY` 결과다. Native P의 전체 유효 Prompt KV와 Final Hidden은 같은 Geometry의 Single-GPU Reference와8/8 일치했다. D의 Import 직후 전체 Prompt KV도8/8 일치했다. KV Credit은 초기값으로 복원됐다.

기존 BI0 Natural C1 Reference 대비 Greedy Sequence 일치는 Standard8/8, Native7/8, Agg7/8이었다. 서로 다른 Geometry의 QMSum Output이 달랐다. 이는 전체 KV 전송의 손상과 구분한다. 작은 N8 Task 품질 결과를 전체200개 정확도 인증으로 사용하지 않는다.

## 신규 Natural200 낮은 부하

아래 표는 `v1` 실행 계약의 결과다. 계측 Wrapper가 공식 UCX 기본값 설정 전에 SDK를 Import했다. 따라서 기본 초기화가 복구된 최종 기준선으로 사용하지 않는다. `v3`에서 공통 Grid를 다시 측정한다. 기존 수치와 Raw Log는 보존한다.

QPS0.4, Arrival Seed2026112051, 독립 Start는 각1회다. 다음 값은 `MEASURED_ONLINE`이다. 두 방법은 같은200개 Prompt와 Arrival Trace를 사용했다.

| Method | TTFT p95 | ITL p99 | TPOT p95 | Output tokens/s | Joint SLO | SLO-goodput |
|---|---:|---:|---:|---:|---:|---:|
| Native | 2308.981ms | 9.293ms | 8.333ms/token | 9.0768 | 95.0% | 8.2316 tokens/s |
| Agg DP2 | 1632.673ms | 9.445ms | 8.369ms/token | 9.0489 | 96.0% | 8.7961 tokens/s |

Frozen SLO는 TTFT≤5000ms, 모든 ITL≤34ms, TPOT≤33ms다. Native는4779 Tokens, Agg는4761 Tokens를 생성했다. Reference의4723 Tokens와 같지 않다. Natural BI0 출력의 길이 차이를 숨기지 않는다. 이 값은 Arrival부터 Drain까지의 Finite-cohort 처리율이다. Sustainable Capacity가 아니다.

두 방법은 사전 연구 기준인 Task별 점수차−2.0점의 Prompt-paired CI 검사를 통과했다. 이 품질 CI는 Server-start 성능 CI가 아니다. Native의 Strict Sequence 일치는174/200, Agg는189/200이었다. 사용자에게 품질 허용폭을 비차단 질문으로 제시했다. 현재2.0점은 답변 전 연구용 기본값이다. Production 품질 기준으로 주장하지 않는다.

이 낮은 부하에서 Native P Waiting의 시간 가중 평균은0.0236개였다. 지속적인 P Backlog를 확인한 조건이 아니다. 따라서 낮은 D 부하만으로 Prefill-heavy 또는 Decode-safe Headroom을 선언하지 않는다.

## NIXL 개입

N32/QPS2.35의 별도 진단에서 기본4 Thread,1 Thread,0 Thread를 비교했다. 각 설정은 독립 Start1회다. ITL p99는27.250/17.317/125.882ms였다. 0 Thread는 악화됐다. 후속 Full200의3개 Paired Start에서1 Thread/4 Thread의 ITL p99 비율은1.3808이었다.95% CI는[1.1239,1.7155]였다. 작은 Pilot의 개선은 재현되지 않았다. 공통 Main은 기본4 Thread를 유지했다.

KV Block128도 N8의 전체 KV/Hidden/전송 검사를 통과했다. Block 크기 변경은 별도 Transport Geometry 개입이다. 기본 Block16 결과를 대체하지 않는다. Byte 수, Descriptor 수와 Memory Rounding을 함께 기록한다. Host Posting 시간을 순수 DMA 시간으로 표현하지 않는다.

## 근거

- [전체 N8 상태 검사](dense_ar_bi0_shortanswer200_20261120/analysis/BI0_N8_official_state_transport_gate.json)
- [공통 Main 설계](dense_ar_bi0_shortanswer200_20261120/configs/frozen_BI0_baseline_design_v1.json)
- [Thread 진단](dense_ar_bi0_shortanswer200_20261120/analysis/NIXL_thread_screen_v1.json)
- [KV128 상태 검사](dense_ar_bi0_shortanswer200_20261120/analysis/KV128_N8_state_gate.json)

Shared KV128의 고부하 Held-out3Start 비교를 완료했다. 모든15개 Start가 Full200 Task-quality CI를 통과했다. Kernel Activity와 Active-decode Safe Window의 적격성은 별도 판정이다.

## 초기화 감사와 실패 보존

공식 vLLM의 `nixl_utils.py`는 SDK Import 전에 `UCX_RCACHE_MAX_UNRELEASED=1024`를 설정한다. 초기 Wrapper의 Import 순서는 이 설정을 건너뛰었다. 새 실행은 공식 초기화를 먼저 수행한다. 설치된 Runtime은 변경하지 않았다.

수정한 N8에서 Native P의 전체 유효 KV와 Hidden은 Reference와8/8 일치했다. D Import도8/8 일치했다. 별도 N32 초기화 A/B는 각1회다. ITL p99는37.447ms와77.360ms였다. 이 결과는 초기화 복구의 성능 개선을 입증하지 않는다. 공식 실행 계약을 복구한 것과 성능 원인 규명은 다르다.

`v2`의 첫 Main은 Serving 전에 실패했다. D의 NIXL Handshake Listener가 `Address already in use`를 기록했다. 측정 요청은0개다. 포트를 선택한 뒤 닫는 동안 Runtime의 다른 Socket이 그 포트를 사용할 수 있었다. 새 Launcher는 OS Ephemeral 범위 밖에서 가용 포트를 선택한다. Listener 실패는 시작 대기를 해제하고 명시적으로 실패한다. 원래 실패 Folder는 유지한다. 새 결과는`v3` 이름으로 저장한다.

## 수치 계측의 준비 상태

동기화 없는 Sparse Tensor Probe는 Standard와 Native P의 차이를 기록했다. Device-wide Synchronization을 추가한 별도 N8에서 P의 Hidden과 Sampled KV는8/8 정확히 일치했다. 따라서 첫 Probe 차이를 모델 계산 오류로 확정하지 않는다.

수정한 Probe에서 Native D는 마지막 Prompt Query1개를 재계산했다. q0의32개 Sampled KV 위치 중 앞31개는 정확히 일치했다. 마지막 위치만 달랐다. 전체8개 요청의 Import 직후 Full KV 검사와 이 재계산 후 Probe를 구분한다. BI0의 Query Geometry 차이와 Task 품질은 별도로 보고한다.

## 빠른 실행 경로의 추가 검증

일반 Single-GPU BI0 C1 Reference의 Mean TBT는7.4578ms였다. 별도 KV Block128 Continuous Native의 Mean TBT는7.4908ms였다. 각 값은 독립 Start1회다. 전자는 단일 Engine 경계다. 후자는 Coordinator Delivery 경계다. 두 값을 GPU-exclusive Decode Kernel Time으로 표현하지 않는다.

처음 작성한 Compile0/Async-off Actual Batch1 Control의 Mean TBT는 약11ms였다. Decode Graph Replay는 실제로 발생했다. 따라서 Graph가 없어서 느렸다고 단정하지 않았다. 이 느린 Control만을 Native 기준으로 삼아 Decode5% 보호를 주장하지 않는다.

별도의 Native N32 A/B는 Async 설정 하나만 바꿨다. 두 Arm은 Compile0, Full Decode Graph, KV Block128, B8192/S1024와 Actual Batch1을 유지했다. Source Credit은4다. QPS는2.35다. Arrival Seed는2026112033이다.

| Async | Mean TBT | ITL p99 | TPOT p95 | TTFT p95 | Strict Greedy |
|---|---:|---:|---:|---:|---:|
|Off|10.9442ms|12.6131ms|11.6257ms/token|14696.52ms|32/32|
|On|7.3798ms|9.1757ms|8.0830ms/token|14008.11ms|32/32|

각 Arm은 독립 Start1회다. 이 개입은 Async 설정의 전체 Runtime 비용 차이를 보여준다. GPU Kernel, Host Gap과 Sampling 각각의 인과 비중은 이 A/B만으로 식별하지 못했다. 빠른 경로의 모든 Layer/Token Cut은 성능 실행 전에 전체 State와 Lifetime 검사를 다시 통과했다.

Async-on N32는 요청마다 실제 Generation Query1개를 더 실행했다. 이 Query는 Client에 출력되지 않았다. 마지막 Worker Output 뒤에 새 Query를 준비한 것은 아니었다. 따라서 추가 작업을 무조건 EOS 뒤의 별도7ms로 더하지 않는다. 보이는 마지막 Token을 Device-wide GPU 완료로도 해석하지 않는다.

기본 KV Block16의 v3 공통 QPS Sweep은 완료했다. 결과와 Start-level CI는 [Online 결과](dense_ar_bi0_online_serving_qps_results.md)에 기록했다. BI0 Numerical Variation을 허용하는 Task-quality 검사는 Strict Token Match와 별도다. 복잡한 새 Method의 품질 실패를 성능 표에서 숨기지 않는다.

## 같은 실행 설정의 독립 Single-GPU Reference

고정된 Calibration20 Prompt에서 추가 비교를 수행했다. 두 Arm은 BI0, Compile0, Async-on, Full Decode Graph, KV Block128과 실제 Request Batch1을 사용했다. 모든 Engine은 B8192/S1024다. State Capture와 Kernel Collector는 사용하지 않았다. 두 시간 경계는 모두 Coordinator Delivery다.

| 경로 | 독립 Start | Mean TBT | ITL p99 | TPOT p95 | 실제 Output Tokens |
|---|---:|---:|---:|---:|---:|
| 공식 Single-GPU |1|7.7790ms|9.2171ms|8.6349ms/token|421|
| 공식 NIXL Native PD |1|7.5444ms|9.0232ms|8.4159ms/token|403|

Single은 GPU1 한 대를 사용했다. Native는 GPU0/1 두 대를 사용했다. 따라서 이 표는 동일 자원 Throughput 비교가 아니다. Decode 경로의 독립 진단이다. Source Prefill, Handoff와 TTFT를 비교한 속도표도 아니다.

Greedy Sequence는18/20 일치했다. Native D는 마지막 Prompt Query1개를 재계산한다. Natural Output 수의 차이를 숨기지 않는다. 같은 Context와 Decode Batch를 사용해도 첫 Query Geometry의 차이가 전체 출력을 바꿀 수 있다. 이 표의 Prompt20개를 독립 Server Start로 세거나 성능 CI로 사용하지 않는다.

현재 BI0 Native의 Client-visible Decode는 이 독립 Reference와 비슷한 범위였다. BI1에서 관측한22–25ms를 새 BI0 경로의 정상 비용으로 사용하지 않는다. BI0가 BI1과 동일한 전체 출력을 보존하는 수정이라고 주장하지도 않는다. GPU-exclusive Decode Kernel Time의 Clean 분해는 별도의 미완료 Gate다.

근거는 [Matched Reference](dense_ar_bi0_shortanswer200_20261120/analysis/BI0_matched_single_native_decode_reference_v1.json)에 저장했다. Actual Batch, Worker Config, Output IDs와 Source ZIP Hash도 보존했다.

## Full200 Numerical 범위

아래 값은 C1 Natural Reference와 전체 Greedy Sequence가 같은 요청 수다. 각 분모는200이다. 같은200개 Prompt에 대한 Held-out Arrival Seed2026112051/52/53을 사용했다.

| Method | Seed51 | Seed52 | Seed53 |
|---|---:|---:|---:|
|Continuous Native|176/200|175/200|175/200|
|Agg DP2|175/200|180/200|179/200|
|Controlled Native|164/200|164/200|164/200|
|Layer Last8|157/200|157/200|157/200|
|Token Suffix≈L×8/36|157/200|157/200|157/200|

모든15개 Start는 동결된 Task별 점수차−2.0Point CI Gate를 통과했다. 이 연구용 품질 기준을 Strict Token Parity로 바꾸어 표현하지 않는다. 같은 Geometry의 N8/N3 State Gate와 Full200 Cross-geometry 품질 Gate도 구분한다.

# 소수 독립 Start의 CI 분위수 민감도

진행 중 초안이다. 완료하지 않은 조건과 최종 판정을 구분한다.

## 공통 실행 조건

Model: Qwen/Qwen3-8B. Revision: `b968826d9c46dd6066d109eabc6255188de91218`.
Original BF16, Exact Causal Attention, Greedy, Thinking Disabled, Full36 Decode를 유지한다.
Input8000과 Output16/32/64/128/200을 사용한다. `ignore_eos=True`이다.
GPU0/1은 H100 SXM80GB이다. KV Budget은 GPU당42GiB이다.
GPU UUID는 `GPU-f217c8a0-1142-20f4-d84b-af29f3a47a0d`와 `GPU-a77f3471-67d4-20b0-9fab-e502d4de5adb`이다.
Runtime은 vLLM0.26.0+cu129, PyTorch2.11.0+cu129, NIXL1.4.1이다.
모든 Worker에 B8192/S1024를 적용한다. Prefix Cache는 비활성화한다.
PD의 공통 D Admission 상한은8이다. 실제 준비 Group과 Decode Cohort는 다를 수 있다.
실제 모델 Context Limit은40960이다. Worker의 실행 한도는 Input+Output에서 정한다.
따라서 Output16과 Output200의 Worker 한도는 각각8016과8200이다.
같은 Output에서는 모든 비교 방법의 한도가 같다.

Branch: `flashvep/dense-ar-matched-axis-crossover-20261117`.
Publication Parent: `228a041f35cedcee9f45beba6c462b453fb3a2c5`.
Parent SHA를 전체 실행 Source SHA라고 표시하지 않는다.
각 실행의 `command.json`에 실제 `source.zip` SHA256을 저장한다.
Worker별 실제 설정, Prompt Hash, Seed 및 설정 Hash도 원시 Evidence에 저장한다.
[environment.json](../state/environment.json)에 GPU와 Model 설정을 보존한다.
[frozen_new_inference_sources_v3.json](../state/frozen_new_inference_sources_v3.json)에 동결한 Inference Source Hash를 보존한다.

## 표본과 지표

성능 비교는 실제 문서144개를 사용한다. 동일 Prompt를 독립 Start로 세지 않는다.
기존 Held-out128개에 기존 Discovery의 다른 문서16개를 추가했다.
새로운 미관측 Prompt Pool이라고 주장하지 않는다.
Discovery Seed는2026111780이다.
QPS6 Confirmation Seed는2026111790/2026111791/2026111792이다.
Discovery와 Confirmation은 CI 계산에서 합치지 않는다.
CI의 재표본 단위는 독립 Server Start이다. n1에는 CI가 없다.
표의 p95/p99는 Start 안에서 계산한 분위수를 Start 사이에서 평균한 값이다.
기존 CI는 Bootstrap Draw 값의 Type7 선형 보간을 유지한다.
별도 역CDF 분위수 민감도 분석도 같은 독립 Start에 적용한다.
두 방식에서 결론이 바뀌면 확정적인 우위나 비열등성으로 판단하지 않는다.
표는 표시용으로 반올림한다. 반올림하지 않은 값과 CI는 JSON에 보존한다.
Raw Output tokens/s는 전체 유한 Cohort의 Arrival-to-drain 처리율이다.
이를 Maximum Sustainable Capacity라고 표시하지 않는다.
Client-visible ITL에는 첫 Token Interval을 포함한다.
Worker Notice 시간과 Client Delivery 시간은 별도로 보존한다.

현재 집계한 Discovery는172/232회이다.
현재 집계한 Confirmation은0/120회이다.
표에 없는 예정 조건은 `NOT_RUN`이다.
상태 Matrix는 [full_execution_matrix.csv](../analysis/full_execution_matrix.csv)에서 확인한다.
개별 요청은 [per_request_metrics_and_work.csv](../analysis/per_request_metrics_and_work.csv)에서 확인한다.
독립 Start 통계는 [independent_start_statistics.json](../analysis/independent_start_statistics.json)에서 확인한다.

## SLO 계약

기존 Output별 Frozen SLO를 변경하지 않았다.
Output200 Balanced는 TTFT5000ms, 모든 ITL34ms, TPOT33ms이다.
Output Sweep의 공통 보조 SLO도5000/34/33ms이다.
기존 Output64 Balanced의 ITL35ms와 공통 보조34ms를 구분한다.
Goodput은 세 조건을 모두 통과한 요청의 Output Token 수를 분자로 사용한다.
Reference Goodput이0이면 비율은 정의하지 않는다.
Native-relative 보호는 ITL p99와 TPOT p95 비율의95% CI 상한을 검사한다.
두 상한이 모두1.05 이하여야 PASS이다. 최소3개 독립 Start가 필요하다.
Absolute SLO와 Relative 비열등성은 서로 다른 Gate이다.
[frozen_SLO_regimes.json](../configs/frozen_SLO_regimes.json)와 [common_output_sweep_SLO.json](../configs/common_output_sweep_SLO.json)을 보존한다.

## 외부 중단과 재개

제어 세션 재시작으로 중단한 실행1건은 FAILED로 보존한다.
원래 Manifest Slot은 유지하고, name만 다른 대체 실행을 연결한다.
중단한 부분 결과는 독립 표본이나 Clean 성능에 포함하지 않는다.
완료한 기존 GPU 실행은 반복하지 않는다.
[복구 기록](../state/daemon_recovery_20261118.json)에 원래 Receipt Hash와 대체 이름을 저장한다.

## 정의와 원인

기존 코드는 Type7 선형 보간 분위수를 계산한다. 산술 오류로 판정하지 않았다.
역CDF는 관측된 이산 Bootstrap 분포의 누적 확률에서 분위수를 정한다.
두 방법은 서로 다른 정의이다. [NumPy quantile 문서](https://numpy.org/doc/stable/reference/generated/numpy.quantile.html)가 두 정의를 구분한다.
3개 Start의 정확한 Bootstrap Draw는27개이다.
각 Draw를 반복해서 늘려도 이산 확률분포는 바뀌지 않는다.
그러나 Type7 보간의 끝점은 표현된 Sample 수에 따라 바뀔 수 있다.
역CDF 분석은 이 표현 변경에 영향을 받지 않는다.

발견한 N36/Last12/Output16/cap1 예에서는 기존 TTFT 비율 CI 상한이0.9995468303958746이다.
역CDF 방식의 상한은1.0109602351406097이다.
따라서 이 경계 결과를 강한 Layer 우위로 채택하지 않는다.
원시 Metric, 기존 CI 및 Source는 바꾸지 않는다.
새 점검의 시점과 원래 통계 Snapshot Hash를 별도로 보존한다.
이 점검은 Control 결과를 본 뒤 추가했다. Primary Confirmation 시작 전이다.
모든 방법과 조건에 같은 점검을 적용한다.
SLO와 비열등성 Threshold1.05는 변경하지 않는다.

## 경계 결론이 바뀐 조건

| Stage | 비교 | Last | Output | cap | Starts | 원래 TTFT CI | 역CDF TTFT CI | 원래 NI | 역CDF NI |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| admission_control | Token→Layer | 12 | 16 | 1 | 3 | [0.9565176649792706, 0.9995468303958746] | [0.9540868913974111, 1.0109602351406097] | NOT_APPLICABLE | NOT_APPLICABLE |

Start가 적다는 한계는 어느 정의를 사용해도 남는다.
정확한 Draw 열거가 모집단95% Coverage나 WCET 보장을 뜻하지 않는다.
Within-start Latency Percentile의 정의는 변경하지 않는다.
전체 민감도 기록은 [CI_quantile_sensitivity.json](../analysis/CI_quantile_sensitivity.json)에 보존한다.

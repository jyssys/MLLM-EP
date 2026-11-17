# Dense AR의 Host-state 실행 예산과 변동성

## 범위

이 보고서는 실제 Online Decision과 상태를 갱신하는 Replay를 구분한다.
물리적인 GPU Kernel Idle은 `NOT_MEASURED`다.
Host Queue나 CUDA Event Enclosure를 Kernel Idle로 해석하지 않는다.

Model은 Qwen/Qwen3-8B다.
Revision은 `b968826d9c46dd6066d109eabc6255188de91218`다.
Original BF16과 Exact Attention을 유지했다.
모든 요청은 Full36 Decode를 수행한다.
공정 비교에는 H100 SXM 80GB GPU0/1을 사용한다.

## 실행 예산의 정의

정책은 P Cut Ready, D Cohort Retirement와 Request별 Deadline을 사용한다.
이미 도착한 FIFO Request만 짧은 Lookahead에 포함한다.
미래 Arrival이나 실제 미래 완료시각은 읽지 않는다.
현재 Cut을 선택한 뒤 P Prefix를 제출한다.
선택 전에 D의 KV·Hidden Reservation을 확인한다.
Commit 이후 Prefix를 버리거나 처음부터 재계산하지 않는다.

현재 D Cohort가 끝나기 전에는 새로운 Suffix를 Admission하지 않는다.
이 Guard는 기존 Decode와 Suffix를 임의로 동시에 실행하지 않는다.
이미 제출한 Kernel을 선점할 수 있다고 가정하지 않는다.
따라서 Temporal Gap만을 탐색하는 정책도 아니다.
새 Cohort의 Suffix와 Decode가 차지하는 D Residency를 함께 예측한다.

기록한 Deadline Residual은 다음 값이다.

`min(해당 Group의 TTFT Deadline) − 예측한 First-output Completion Upper Bound`

이 값은 모델 출력이다.
실제 사용 가능한 GPU Idle의 실측값이 아니다.
예측된 D Available부터 P Cut Ready까지의 차이도 같은 한계를 갖는다.
이미 TTFT Deadline을 넘긴 요청의 Migration은 Queue Benefit을 위해 허용할 수 있다.
그 요청을 SLO 통과로 처리하지 않는다.

## 비용 모델과 한계

P Source와 D Preparation의 Host Enclosure를 Calibration에 사용했다.
D Preparation에는 Handoff와 Host Scheduling이 포함될 수 있다.
별도의 Wire Duration을 더해 이 비용을 중복 계산하지 않는다.
이 모델을 Exclusive GPU Compute Time이라고 표시하지 않는다.

실측 Shape가 없는 Cell은 기존 Lookup을 빌린다.
모델 파일은 Borrowed Cell을 기록한다.
실제 Source Macro8/D Cohort8 Calibration 뒤 v3를 별도로 동결했다.
Held-out 결과로 같은 버전의 모델을 다시 Fit하지 않는다.

Completion Upper Bound는 Stage별 관측 Tail과 추가 110ms Margin을 사용한다.
표본이 작은 p95와 이 Margin은 WCET 보장이 아니다.
실제 Coverage와 Underprediction은 별도 Trace에서 계산한다.
정책은 Input8000만 Calibration했다.
다른 Input 길이에서는 Native를 선택한다.
Mixed-input에 대한 최적 Dynamic 정책을 구현했다고 주장하지 않는다.

## 근사 Oracle의 의미

Replay는 Cut에 따라 P/D 완료시간과 Credit Lifetime을 갱신한다.
Frozen-state에서 Cut 하나의 비용만 빼는 방식과 다르다.
모델은 Source KV, Destination KV와 Hidden Credit도 추적한다.
반면 실제 Kernel Timeline이나 미래 Host Stall을 정확히 재현하지 않는다.

Dynamic 탐색은 유한한 Beam을 사용한다.
따라서 전역 최적 Oracle이나 성능 상한이 아니다.
더 강한 Static이 이 Beam보다 좋은 값을 낼 수 있다.
그 결과를 Dynamic의 구조적인 불가능성 증명으로 사용하지 않는다.

실제 Macro8 Calibration을 사용한 Mixed Output 탐색에서 QPS4.8의 근사 이득은 4.1984%였다.
이때 모델의 Joint SLO Attainment는 96.875%였다.
기준 Static은 한 가지 Cut 또는 Output Cap별 Cut을 사용했다.
더 강한 Output Cap별 Fraction Control은 9009개 조합을 탐색했다.
이 Control의 최대 SLO Token Rate는 234.905944 tokens/s였다.
근사 Dynamic Beam은 233.999757 tokens/s였다.

최대 Token Rate를 낸 Static의 Joint Attainment는 94.53125%였다.
따라서 이 Static을 Joint95% 성공으로 표시하지 않는다.
Joint95%를 만족하는 Static의 별도 Frontier도 계산했다.
같은 v3 모델과 같은 Arrival Trace를 유지했다.

| 모델 정책 | SLO output tokens/s | Joint Attainment | TTFT p95, ms |
|---|---:|---:|---:|
| Joint95% 제한을 통과한 Best Cap/Fraction Static | 231.036054 | 95.3125% | 9905.942 |
| Beam256 | 233.999757 | 96.875% | 9652.554 |
| Beam1024 | 235.761290 | 96.875% | 9720.368 |

Beam1024의 Joint95% Static 대비 이득은 약 2.045%다.
이는 유망 기준 5%보다 작다.
Beam1024도 성능 상한은 아니다.
위 값은 모두 `OFFLINE_ORACLE`다.
실제 Serving Speedup 또는 Sustainable Capacity가 아니다.

## 근거와 재현

- [모델 v3](dense_ar_layer_token_dynamic_headroom_20261117/configs/empirical_host_cost_model_discovery_v3.json).
- [상태 갱신 Replay](../dense_ar_layer_token_pd/replay.py).
- [Online Policy](../dense_ar_layer_token_pd/flow_policy.py).
- [실제 Batch8 Oracle 탐색](dense_ar_layer_token_dynamic_headroom_20261117/analysis/flow_macro8_oracle_screen_v3.json).
- [강한 Fraction Control](dense_ar_layer_token_dynamic_headroom_20261117/analysis/strong_per_cap_fraction_oracle.json).
- [Joint95% 제한과 확장 Beam](dense_ar_layer_token_dynamic_headroom_20261117/analysis/strong_per_cap_fraction_oracle_v2_Joint95.json).
- [실제 후보·예산 분포](dense_ar_layer_token_dynamic_headroom_20261117/analysis/online_safe_budget_variance.json).
- [예측 오차와 Coverage](dense_ar_layer_token_dynamic_headroom_20261117/analysis/online_prediction_and_deadline_validation.json).

후보 거절 사유는 서로 중복될 수 있다.
거절 사유 합계를 Request 수로 해석하지 않는다.
Source Group별 예측 오차도 독립 Server Start처럼 취급하지 않는다.

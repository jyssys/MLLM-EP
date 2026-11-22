# Output Prior와 Selector 검증 범위

상태: ACTIVE_SNAPSHOT.
생성 시각: 2026-11-22T18:39:15.245786+00:00.
경과 연구 시간: 11.540시간.
기준 Commit: `5ebed239073b4fa3709361038cbec026d1c32b78`.
연구 Branch: `flashvep/longbench400-v2-output-aware-2d-20261122`.

## 관측 가능한 Output Prior

Training은 Discovery208의 독립 Compiled C1 Output만 사용한다.
Feature는 사전에 주어진 Task Label과 Tokenized Input Bucket이다.
Held-out192의 미래 EOS 길이는 Feature나 Training Label로 사용하지 않는다.
Task/Input 평균은 Log1p 공간에서 8개 Pseudo-observation으로 축소한다.
모든 요청의 max_tokens=2048은 알려진 값이다. 하지만 요청을 구분하는 Feature는 아니다.

| Model | Held-out Requests | MAE Tokens | Median AE Tokens | Output-bin Accuracy | Reference >200 |
| --- | --- | --- | --- | --- | --- |
| Constant | 192 | 48.110 | 21.868 | 0.2240 | 1 |
| InputOnly | 192 | 47.054 | 27.965 | 0.2240 | 1 |
| TaskPrior | 192 | 32.231 | 14.740 | 0.4062 | 1 |
| TaskPlusInput | 192 | 30.219 | 14.845 | 0.4740 | 1 |

예측 정확도는 Axis Winner 정확도 또는 실제 Serving 이득이 아니다.
Dataset 선정을 위해 Candidate EOS를 사용했다. Held-out은 C3 Tuning에 대한 분리다.
이 요청을 모든 연구 단계에서 처음 보는 완전 미관측 데이터라고 표현하지 않는다.

## Controlled-output Rule

C0 Rule은 Discovery의 Known Fixed Output과 Input Bucket으로 동결한다.
이 실험의 Output Budget은 이미 알려져 있다. Natural EOS의 미래 길이와는 다르다.
C0 Rule의 Held-out 예측은 Diagnostic이다. C3 Queue-updating Replay로 해석하지 않는다.

| Rule | 평가 범위 | Cells | Point Winner Accuracy | Mean Cell TTFT p95 Regret ms | Mean Relative Regret |
| --- | --- | --- | --- | --- | --- |
| FixedAxis | All completed cells | 80 | 0.5125 | 30.492 | 0.0179 |
| FixedAxis | Statistically confirmed cells | 53 | 0.5660 | 39.567 | 0.0217 |
| FixedAxis | Both arms decode-qualified cells | 52 | 0.4808 | 32.118 | 0.0217 |
| OutputOnly | All completed cells | 80 | 0.6875 | 10.232 | 0.0120 |
| OutputOnly | Statistically confirmed cells | 53 | 0.7736 | 11.858 | 0.0131 |
| OutputOnly | Both arms decode-qualified cells | 52 | 0.6731 | 9.493 | 0.0151 |
| InputAndOutput | All completed cells | 80 | 0.6625 | 9.415 | 0.0106 |
| InputAndOutput | Statistically confirmed cells | 53 | 0.7925 | 9.503 | 0.0081 |
| InputAndOutput | Both arms decode-qualified cells | 52 | 0.6538 | 6.200 | 0.0120 |

[c0_output_selector_rules_frozen_v2.json](dense_ar_longbench400_v2_20261122/configs/c0_output_selector_rules_frozen_v2.json)

[c0_observable_axis_rule_diagnostic_v2.json](dense_ar_longbench400_v2_20261122/analysis/c0_observable_axis_rule_diagnostic_v2.json)

[heldout_output_prior_prediction_v2.json](dense_ar_longbench400_v2_20261122/analysis/heldout_output_prior_prediction_v2.json)

## Dynamic Contribution

검증된 Decode-safe Window가 없으면 Online Axis 또는 2D Selector는 실행하지 않는다.
미래 EOS를 읽는 Oracle은 Online Selector가 아니다.
Queue, KV Credit 및 ITL Tail을 검증하지 못한 Oracle은 Serving Headroom으로 사용하지 않는다.
복잡한 Dynamic Controller를 구현하지 않은 경우 상태를 `NOT_RUN`으로 유지한다.

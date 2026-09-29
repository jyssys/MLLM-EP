# Rank-aware physical bridge + visual commitment campaign

This campaign followed the authoritative specification at
`poc_flashvep/reports/rank_aware_physical_bridge_and_visual_commitment_poc_spec.md`
(SHA-256 `c78172d5b2daf4027f211ae0b2660a5abf6f19fc1f2bc9677182612c5e4c9bab`).
Only physical GPUs 2, 3, 6, and 7 were exposed. Both experiments used the
frozen Qwen3-VL-30B-A3B-Instruct revision
`9c4b90e1e4ba969fd3b5378b57d966d725f1b86c`, BF16, top-8, and TRUE
TP2-DP2-EP4 with DeepEP high-throughput dispatch/combine.

## PoC-1: physical bridge

The final run contains 2,304 complete 48-layer invocation aggregates across
three interleaved repeats and decode steps 16/32/64/96. At a matched 25% path
reduction, rank-aware K6 reduced max-rank active experts by 18.724%, versus
16.058% for native-tail K6 (+2.667 percentage points). This structural
increment did not become reliable physical latency gain:

- Rank-aware vs K8 expert execution: +1.261%, 95% CI [-0.538%, 3.106%].
- Rank-aware vs K8 decode MoE: +1.638%, 95% CI [-2.269%, 5.214%].
- Direct rank-aware vs native-tail expert execution: +0.484%, 95% CI
  [-1.185%, 1.990%].
- Direct rank-aware vs native-tail decode MoE: +0.716%, 95% CI
  [-3.012%, 3.575%].
- The controlled same-pair collapse reduced max-rank active experts 11.138%
  while total pair count was identical and max-rank pair count differed only
  0.237%; expert execution changed +0.424% (95% CI [-1.505%, 2.351%]) and
  decode MoE changed -0.677% (95% CI [-4.951%, 3.391%]).

Verdict: **STRUCTURAL_ONLY_NO_RUNTIME_GAIN**. Counters alone do not bridge to
measurable TRUE EP4 latency in the tested B8/fixed128 regime. No TPOT or E2E
claim is made.

## PoC-2: visual commitment event

The causal run evaluated 8 calibration, 24 untouched held-out, and 4
same-image repeat-noise pairs. Each pair used the same prompt, the same frozen
128-token continuation, a same-domain visual counterfactual, and 56 full-vocab
logit comparisons. The calibration search found no valid persistent-low JSD
rule (calibration event rate 0%). The frozen rule found 0/24 held-out events.

Average sensitivity declined at a generic position-64 split by 0.090406 JSD
(request-bootstrap 95% CI [0.069891, 0.111050]), but this was not a persistent
commitment state: mean held-out JSD was 0.205120 at steps 1–32, 0.119598 at
36–64, 0.088565 at 68–96, then rose to 0.109589 at 100–128. The matched
pseudo-event control produced the same 0.090406 mean delta. Same-image repeat
noise was numerically zero.

Verdict: **NO_VISUAL_COMMITMENT_EVENT**. Per the gate, pre/post K6 damage,
post-commit rank-aware closure, and a cheap proxy were not run.

## Decision

Both remaining chains failed their decisive gates. The existing structural
rank-aware observation remains valid, but it does not justify a physical
speedup claim; the causal visual trajectory does not supply an MLLM-specific
`WHEN` signal. The active-expert-reduction line should therefore be closed for
this tested model/workload rather than extended with another visual scalar.

```text
GPU_SET: 2,3,6,7

POC1_MAX_RANK_ACTIVE_REDUCTION: 18.724% rank-aware K6; 16.058% native-tail K6; +2.667 pp increment
POC1_EXPERT_EXECUTION_GAIN: +1.261% vs K8, 95% CI [-0.538%, 3.106%]; +0.484% vs matched native-tail, 95% CI [-1.185%, 1.990%]
POC1_DECODE_MOE_GAIN: +1.638% vs K8, 95% CI [-2.269%, 5.214%]; +0.716% vs matched native-tail, 95% CI [-3.012%, 3.575%]
POC1_VERDICT: STRUCTURAL_ONLY_NO_RUNTIME_GAIN

POC2_EVENT_RATE: 0/24 held-out (0%); calibration 0/8
POC2_PRE_POST_VISUAL_SENSITIVITY_DELTA: N/A_BY_NO_EVENT; generic fixed-position-64 delta 0.090406 JSD
POC2_PRE_POST_95CI: N/A_BY_NO_EVENT; generic fixed-position-64 95% CI [0.069891, 0.111050]
POC2_POST_EVENT_K6_DAMAGE_GAIN: NOT_RUN_BY_EXISTENCE_GATE
POC2_GENERIC_POSITION_CONTROL: generic and matched pseudo-event delta both 0.090406 JSD; late JSD rebounds at steps 100-128
POC2_RANK_AWARE_CLOSURE_GAIN: NOT_RUN_BY_EXISTENCE_GATE
POC2_PROXY_STATUS: NOT_RUN_BY_EXISTENCE_GATE
POC2_VERDICT: NO_VISUAL_COMMITMENT_EVENT

OVERALL_NEXT_ACTION: Close this active-expert-reduction line for the tested model/workload; do not add another visual scalar or claim physical latency gain.
BRANCH: flashvep/mllm-rank-aware-physical-bridge-visual-commitment
COMMIT: resolve the pushed branch head with `git rev-parse HEAD`
```

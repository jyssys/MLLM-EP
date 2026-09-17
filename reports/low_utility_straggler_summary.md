# Low-Utility EP Straggler — Summary

**Verdict: HOLD.** The proposed enrichment mechanism fails, while a separate EP-pressure pruning oracle survives only at aggressive, quality-unvalidated mass budgets.

## Gate result

- G1 low-utility enrichment: **FAIL**. EP8 bottom-25 enrichment is 0.947x (request-bootstrap 95% CI 0.943-0.952x) and the difference is -1.89pp on the all-row audit.
- G2 excess share: **PASS structurally**. `mass<=10%` routes carry 39.41% of sampled EP8 `ATTRIBUTED_EXCESS`, but they are common everywhere.
- G3 EP-specific advantage: **PASS only in the sampled systems oracle**. At 5% mass, EP8 P2/P3/P4 stage gains are 1.64% / 6.24% / 4.91%.
- G4 scaling: **PASS**. The P3 increment at 5% is 3.23pp EP4 versus 4.60pp EP8.

## Interpretation

Both the exact GSM8K-128 current-block population and the GSM8K-32 all-physical-row audit show low-mass fractions lower on critical than non-critical ranks, so the proposed enrichment mechanism is false. Low-mass routes are nevertheless common: the full-row audit attributes 39.41% of EP8 critical excess to <=10%-mass routes. On 2,048 exact full-row invocation replays, EP-aware selection beats utility-only by 4.60pp at a 5% mass budget and 7.69pp at 10%, while reducing max/mean and wait. This is HOLD rather than GO because the effect needs aggressive unvalidated approximation, the full-row policy is only a systems sample, and its opportunity is not shown to arise specifically from dLLM refinement.

No generation, quality rollout, weight renormalization, or production runtime was executed. EP4/EP8 remain calibrated simulations. The 128-request analysis is exact for current-block slots; the full-row anatomy and systems counterfactual use existing GSM8K-32 heavy-trace samples.

## Required final summary

```text
TRACE_SOURCE:
artifacts/virtual_ep/20260917_four_hypothesis_discovery/aggregate_v2/gsm8k_128_discovery_v2_fixed.npz (exact current-block slots, GSM8K-128); artifacts/virtual_ep/20260917_phase3a_threshold_ep_smoke/threshold_095/trace.npz (all-physical-row audit, GSM8K-32); grouped-mm calibration extended with 105 actual post-pruning owner-rank replays

EP4_LOW_MASS_CRITICAL_ENRICHMENT:
0.9713x / global bottom-25% normalized effective route mass, all-physical-row audit (-1.0316 pp; request-bootstrap 95% CI 0.9638-0.9806x)

EP8_LOW_MASS_CRITICAL_ENRICHMENT:
0.9475x / global bottom-25% normalized effective route mass, all-physical-row audit (-1.8912 pp; request-bootstrap 95% CI 0.9429-0.9523x)

EP4_LOW_MASS_CRITICAL_EXCESS_SHARE:
41.3106% / normalized per-route mass <=10%, all-physical-row sampled ATTRIBUTED_EXCESS

EP8_LOW_MASS_CRITICAL_EXCESS_SHARE:
39.4135% / normalized per-route mass <=10%, all-physical-row sampled ATTRIBUTED_EXCESS

EP4_TAIL_7_8_CRITICAL_EXCESS_SHARE:
24.6689%

EP8_TAIL_7_8_CRITICAL_EXCESS_SHARE:
23.6362%

BEST_GENERIC_PRUNING_POLICY:
P2 utility-only / 10% removed router-mass budget / sampled full physical rows (EP8 stage gain 3.365%)

BEST_EP_AWARE_PRUNING_ORACLE:
P3 four-round calibrated-pressure ORACLE / 10% removed-mass budget (EP8 stage gain 11.052%)

BEST_EP_AWARE_PRACTICAL_HEURISTIC:
P4 assignment-rank-pressure / 10% removed-mass budget (EP8 stage gain 9.199%)

EP4_GENERIC_STAGE_GAIN:
4.388%

EP4_EP_AWARE_STAGE_GAIN:
6.925%

EP4_EP_SPECIFIC_INCREMENT:
+2.537 percentage points

EP8_GENERIC_STAGE_GAIN:
3.365%

EP8_EP_AWARE_STAGE_GAIN:
11.052%

EP8_EP_SPECIFIC_INCREMENT:
+7.686 percentage points

EP8_MAX_MEAN_BEFORE_AFTER:
1.2206 -> 1.1051

EP8_WAIT_FRACTION_BEFORE_AFTER:
0.1750 -> 0.0928

EP8_ROUTER_MASS_NEEDED_FOR_50_PERCENT_EXCESS_REMOVAL:
7.3283% / sampled all-physical-row low-mass-first ATTRIBUTED_EXCESS curve (not a measured removal or quality-safe point)

DLLM_PHASE_EFFECT:
Exact GSM8K-128 current-block analysis remains near/below 1x enrichment across early/middle/late; no growing dLLM-specific phase effect was found.

QUALITY_ROLLOUT_RUN:
NO

PRIMARY_INTERPRETATION:
EP-specific systems oracle, but the proposed critical-rank low-mass enrichment is absent

VERDICT:
HOLD

NEXT_ACTION:
Do not build a production runtime; if continued, run a separate conservative quality study against utility-only at min_k=4 before making any method claim.
```

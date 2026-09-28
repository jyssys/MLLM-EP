
# Final summary

This bounded PoC measured actual selected-branch outputs for 9,312 paths and
actual next-token functional damage for 256 drop-one interventions on TRUE EP4.
The oracle/predictor distinction is preserved, and no future-decode truth was
used as a feature. The Stage-B decision was `TEXT_SIGNAL_SUFFICIENT`.

There is substantial *oracle* safe-skip headroom (72.66%),
but the tested vision-separated estimator does not identify it better than the
text-prefill control with statistical support. At the frozen 20% point, F5 and
F2 make an indistinguishable safety decision, while F5's point gains over F1
have confidence intervals spanning zero. Rare-but-critical paths are also only
0.78% of this held-out sample and F5 did not rescue the one
observed case. Local mixture error is a poor proxy for functional damage
(Spearman 0.0136).

The recent negative frequency-prediction result is not repackaged here: this
experiment targets `vision signal → skip damage`. The frozen vision-increment gate failed, so C0--C6 quality propagation and physical candidate timing were correctly stopped.

## Final Required Summary

BASELINE_STATUS: PASS_REUSED_VALIDATED_TRUE_EP2_EP4

DAMAGE_CAPTURE_STATUS: MEASURED_LOCAL_9312_AND_FUNCTIONAL_256_TRUE_EP4

DROP_ONE_ORACLE_SAFE_SKIP_HEADROOM: 72.66%

PRIMARY_DAMAGE_METRIC: max(0,target_logprob_drop_nats)+2*JSD+1[top1_flip]

GENERIC_ROUTER_DAMAGE_PREDICTION: Spearman 0.1851; critical recall@20 85.19%

MODES_STYLE_DAMAGE_PREDICTION: Spearman 0.2004; critical recall@20 88.89%

TEXT_PREFILL_DAMAGE_PREDICTION: Spearman 0.1925; critical recall@20 92.59%

FULL_PREFILL_DAMAGE_PREDICTION: Spearman 0.1847; critical recall@20 88.89%

VISION_SEPARATED_DAMAGE_PREDICTION: Spearman 0.2072; critical recall@20 92.59%

VISION_INCREMENT_VS_MODES: recall +3.70 pp [-8.33, +10.00]; safe precision +4.00 pp [-8.00, +12.00]

VISION_INCREMENT_VS_TEXT: recall +0.00 pp [-8.34, +8.11]; safe precision +0.00 pp [-8.00, +8.00]

RARE_BUT_CRITICAL_PATH_RATE: 0.78% (1/128)

BEST_FROZEN_SKIP_BUDGET: N/A_STAGE_B_GATE_FAILED

GENERIC_MODES_QUALITY_AT_BUDGET: N/A

VISION_POLICY_QUALITY_AT_BUDGET: N/A

ACTUAL_EXPERT_PATH_REDUCTION: N/A_NO_PROMOTED_RUNTIME_POLICY

EP2_TPOT_GAIN: N/A_NOT_RUN_BY_GATE

EP4_TPOT_GAIN: N/A_NOT_RUN_BY_GATE

EP8_STRUCTURAL_EFFECT: STRUCTURAL_EP8_PROJECTION_ONLY; bounded offline 20% masks in report; no latency claim

VERDICT: TEXT_SIGNAL_SUFFICIENT

NEXT_ACTION: Stop this vision-conditioned branch at Stage B; any future generic/text-aware skipping study must be qualified independently.

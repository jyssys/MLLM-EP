
# Final summary

## Outcome

The bounded PoC supports the physical and structural halves separately: distinct active expert working set changes actual expert-kernel latency at fixed path count, and a future-truth rank-aware collapse upper bound reduces the maximum rank working set substantially more than native tail removal. It does **not** support the MLLM-specific protection chain: measured current-token visual dependency did not meet the held-out, causally aligned skip-damage gate.

Consequently the full named method is not promoted, no modified-route runtime was built, and no speedup is claimed. The defensible verdict is **GENERIC_RANK_AWARE_ONLY**, limited to a bounded generic EP observation plus a nondeployable structural oracle.

## Evidence discipline

- Physical: TRUE EP2/EP4 DeepEP stage and installed Triton kernel measurements.
- Structural: frozen caption route replay, including EP8 structural mapping only.
- Functional: reused exact TRUE EP4 drop-one labels joined to current VDS.
- Not measured: P2 candidate quality, modified-route EP2/EP4 timing, any EP8 latency.

## Final Required Summary

```text
BASELINE_STATUS: PASS_REUSED_VALIDATED_FOUNDATION; TRUE_EP2_AND_EP4_RECONFIRMED

Q1_DECODE_MOE_STAGE_BREAKDOWN: MEASURED_TRUE_DEEPEP_DIAGNOSTIC
Q1_DISPATCH_SHARE: EP2=36.365%; EP4=42.798%
Q1_EXPERT_EXECUTION_SHARE: EP2=51.124%; EP4=43.227%
Q1_COMBINE_SHARE: EP2=12.511%; EP4=13.975%
Q1_TOKEN_COUNT_CORRELATION: EP2=0.0652; EP4=0.1229
Q1_ACTIVE_EXPERT_CORRELATION: EP2=0.0596; EP4=0.1094
Q1_ACTIVE_WEIGHT_BYTES_CORRELATION: EP2=0.0596; EP4=0.1094
Q1_FIXED_TOKEN_ACTIVE_EXPERT_EFFECT: cold 32-path, active2→16 = +55.603% mean
Q1_PHYSICAL_VERDICT: PHYSICAL_ACTIVE_EXPERT_BOTTLENECK_SUPPORTED_BOUNDED

Q2_VDS_ATTN_RESULT: held-out Spearman=-0.2548
Q2_VDS_CONTRIBUTION_RESULT: held-out Spearman=-0.0644
Q2_CAUSAL_VDS_ALIGNMENT: PARTIAL_EXTERNAL_CAUSAL_ONLY; DIRECT_ALIGNMENT_NOT_PROVEN
Q2_VDS_SKIP_DAMAGE_CORRELATION: -0.2548
Q2_HIGH_VS_LOW_VDS_DAMAGE: delta=-0.0554781; 95%CI=[-0.17318413139824318, 0.0028926955343391864]
Q2_MLLM_VERDICT: VISUAL_DEPENDENCY_NOT_LINKED_TO_SKIP_DAMAGE

Q3_MATCHED_PATH_REDUCTION: 20.312%
Q3_NATIVE_TAIL_ACTIVE_EXPERT_REDUCTION: EP4 max-rank=13.816%
Q3_RANK_AWARE_ACTIVE_EXPERT_REDUCTION: EP4 max-rank=25.383%
Q3_MAX_RANK_WEIGHT_BYTES_REDUCTION: EP4=25.383%
Q3_EP2_HEADROOM: 24.049% max-rank structural
Q3_EP4_HEADROOM: 25.383% max-rank structural
Q3_EP8_STRUCTURAL_HEADROOM: 28.172% max-rank; STRUCTURAL_EP8_PROJECTION_ONLY
Q3_VERDICT: RANK_AWARE_STRUCTURED_HEADROOM_SUPPORTED_NONDEPLOYABLE_UPPER

Q4_RANK_AWARE_ONLY_FUNCTIONAL_DAMAGE: NOT_ESTABLISHED_FOR_BATCH_O4
Q4_VISION_PROTECTED_FUNCTIONAL_DAMAGE: NOT_RUN_BY_Q2_GATE
Q4_PHYSICAL_HEADROOM_RETAINED: NOT_RUN_BY_Q2_GATE
Q4_VERDICT: NO_MLLM_SPECIFIC_POLICY_PROMOTION

NATURAL_SERVING_RESULT: EP4 O4 max-rank structural reduction=21.630%
STRESS_SERVING_RESULT: EP4 O4 max-rank structural reduction=24.297% (discovery-only)
PHASE_MIXED_RESULT: EP4 O4 max-rank structural reduction=17.838%

Q5_EP2_TPOT_GAIN: N/A_NOT_RUN_BY_GATE
Q5_EP4_TPOT_GAIN: N/A_NOT_RUN_BY_GATE
Q5_DECODE_MOE_GAIN: N/A_NOT_RUN_BY_GATE
Q5_RUNTIME_VERDICT: NOT_PROMOTED

OVERALL_VERDICT: GENERIC_RANK_AWARE_ONLY
NEXT_ACTION: If pursued, collect dense multi-path functional safety labels and validate a generic rank-aware candidate separately; do not claim visual protection from this PoC.
```

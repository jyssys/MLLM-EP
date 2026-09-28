# Q2-2 final summary

## Answers to the five required questions

1. Changing only visual content causally changed same-path routing/contribution measurements (held-out membership flip 11.979%); this alone is not the functional gate.
2. Greater PCS predicted greater signed visual-specific drop damage: **FAIL** under the joint held-out rule.
3. Generic `D_orig` and signed `D_visual` are reported separately; the verdict is based only on signed `D_visual`.
4. Cheap modality-aware proxy screen: **NOT_RUN_Q2_2A_GATE_FAILED**.
5. Proceed to a vision-protected rank-aware candidate: **NO**.

No TPOT, E2E, throughput, benchmark-quality, production, or EP8-speedup claim is made.

```text
Q2_2_PRIMARY_PCS_DVISUAL_SPEARMAN: -0.058697
Q2_2_HIGH_MINUS_LOW_DVISUAL: 0.003837
Q2_2_HIGH_MINUS_LOW_95CI: [-0.009163, 0.022229]
Q2_2_ROUTER_BASELINE: N/A
Q2_2_BEST_MODALITY_PROXY: NOT_RUN_Q2_2A_GATE_FAILED
Q2_2_FRESH_HELDOUT_REQUESTS: 24
Q2_2_VERDICT: VISUAL_ROUTING_CHANGES_BUT_NOT_PATH_IMPORTANCE
NEXT_ACTION: Stop the MLLM-specific protection branch and retain the generic rank-aware EP direction.
```

# Ablations

Status: `NOT_RUN_AFTER_BASELINE_GATE`.
Model and runtime identities appear in `00_runtime_audit.md`.

| Ablation | Required control | Status |
|---|---|---|
| Identity hook | Same full expert work | PASS on 32 HumanEval requests and two smoke requests per EP4 group |
| Uniform versus router | Same actual executed edges | NOT_RUN |
| Refinement versus EP-only versus joint | Same actual executed edges | NOT_RUN |
| Refinement signal | Same EP cost and router information | NOT_RUN |
| EP signal | Same refinement state and edge budget | NOT_RUN |
| Weight normalization | Original retained weights versus explicit renormalization | NOT_RUN |
| Edge budget | 0%,5%,10%,15%,20% | NOT_RUN |
| NFE | Natural generation and additional matched-NFE control | NOT_RUN |
| Hardware | Group A/B crossover | NOT_RUN |
| Isolation | One group quiet versus EP4×DP2 | NOT_RUN |
| Batch | 1,2,4,8,16,32 | Only B1 smoke ran |
| Block length | 32,64 | Only block32 smoke ran |
| Layer and routing skew | Separate strata | NOT_RUN |

The structural traffic counterfactual is not a quality ablation.
Its removed edges did not execute through a sparse runtime.
It cannot establish actual kernel work reduction.
The campaign must not fill missing policy rows with projected speedups.

# Prior-work boundary (audited before implementation)

Audit date: 2026-10-02. This is not a novelty claim for MoE SD, adaptive depth, expert footprint, or EP traffic optimization. Six primary papers were inspected at the pinned versions below, including their method/cost and experimental sections.

| Work/version | Existing cost/mechanism | Boundary for this discovery |
|---|---|---|
| [SpecMoE v1](https://arxiv.org/html/2604.10152v1), §§II–IV | GPU-resident hot-expert self-draft; CPU-offloaded target; expert parameter migration and acceptance | CPU↔GPU parameter transfers are not GPU-resident activation A2A. No self-assisted novelty claim. |
| [Cascade v1](https://arxiv.org/html/2506.20675v1), §§2–4 | Utility = progress/aggregate speculative-step cost; dynamic depth and speculation disabling; primarily single RTX6000 Ada, n-gram/EAGLE | Adaptive depth is already prior art. Aggregate measured verification cost can implicitly include EP; independent explanatory value, not merely adding a cost label, must be demonstrated. |
| [MoE-Spec v1](https://arxiv.org/html/2602.16052v1), §3 | Expert-union budgeting, shortlist ranking, truncation/substitution | Changes expert coverage; forbidden here. Expert footprint is an existing feature, not a contribution. |
| [EVICT v2](https://arxiv.org/html/2605.00342v2), §3.2.3 | Lossless tree truncation; acceptance estimate divided by offline profiled average end-to-end cost C(k); SGLang integration | Average cost already absorbs runtime expenses. Compare explicit rank features against expert and aggregate-cost controls; no superiority claim from this discovery. |
| [EcoSpec v1](https://arxiv.org/html/2607.12696v1), §§4–5 | Marginal expert union in draft-tree selection; expert-weight HBM reads; 8-H200 evaluation | GPU count alone does not establish a particular EP topology. Do not claim that it ignores all communication; its explicit candidate objective is expert footprint. |
| [Speculative MoE v1](https://arxiv.org/html/2503.04398v1), §§3.1–3.4, 5.5 | Predict token→expert routes; shuffled reduce-scatter; offline co-cluster experts to increase local activation and reduce A2A | Directly covers EP communication, locality, fanout and token/expert scheduling. It does not use a draft-model token continuation with longest-prefix acceptance/correction. |

## Important version correction

arXiv:2503.04398 **v5**, dated 2026-02-27, is titled [Semantic Parallelism: Redefining Efficient MoE Inference via Model-Data Co-Scheduling](https://arxiv.org/html/2503.04398v5). §§3–4 add Attention-DP request rebatching and Attention-TP token reshuffling, including Qwen3-30B-A3B / DeepEP. Both v1 and v5 were inspected. Routing prediction/pre-scheduling must not be confused with ordinary draft/verify/accept SD. Neither version specifies the complete width→accepted-prefix→rank traffic→dispatch/combine-per-committed-token experiment requested here.

`PRIOR_OVERLAP_STATUS=RELATED_EP_PRIOR_NOT_IDENTICAL_SD_MECHANISM`.

This permits discovery, not a claim that the gap exists. If explicit physical features do not improve held-request prediction or decisions over existing cost views, the prescribed negative gate applies. No upstream methods are reimplemented or modified.

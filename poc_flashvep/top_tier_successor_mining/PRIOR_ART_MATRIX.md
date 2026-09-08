# Adversarial prior-art matrix — live screening

No winner or novelty claim yet. Primary sources only; downloaded PDFs and hashes
are in the result root's `references/` directory.

| Candidate family | Closest primary source | What is already covered | What would still require fresh evidence |
|---|---|---|---|
| SERE batch-union substitution | [SERE](https://github.com/JL-Cheng/SERE) | Batch-union top-S, calibrated expert affinity, preservation of low-similarity experts; small-vs-large batch is intrinsic to its stated setting. | A material quality-efficiency failure after reproducing the intended batch regime, not a single-request misuse. |
| Batch-aware expert sharing | [XShare](https://arxiv.org/abs/2602.07265) | Batchwise expert activation selection, gate-mass/heterogeneous batch considerations and distributed budgets. | Orthogonal failure cause; a different threshold or primary union is insufficient. |
| Libra prediction or prefetch-window upgrade | [Libra](https://proceedings.iclr.cc/paper_files/paper/2026/hash/9ff1ac9a659085fed0735362cafe5e53-Abstract-Conference.html), supplied source | Next-gate prediction, actual-route sharding, local/remote splitting, overlapped replication. Short compute windows are acknowledged. | Real MLLM failure of the actual predictor consumer/runtime with material direct benefit, not lower recall alone. |
| Better learned Libra predictor / overlap-aware replication | [PROBE](https://arxiv.org/abs/2602.00509) | Frozen-gate prior plus residual predictor and online distillation; latency-window-bounded replication and split-phase prefetch prioritization. | A cause not solved by these mechanisms. Predictor accuracy or communication contention alone is crowded. |
| MoDES modality-dependent skipping | [MoDES](https://openaccess.thecvf.com/content/CVPR2026/papers/Huang_MoDES_Accelerating_Mixture-of-Experts_Multimodal_Large_Language_Models_via_Dynamic_Expert_CVPR_2026_paper.pdf) | Layer/modality importance, dynamic weighted-router threshold, task-dependent quality loss and calibration. | A material residual limitation after the original calibration and obvious recalibration/fallback controls. |
| Semantic rather than gate-score capacity | [AnyExperts](https://openaccess.thecvf.com/content/CVPR2026F/papers/Gao_AnyExperts_On-Demand_Expert_Allocation_for_Multimodal_Language_Models_with_Mixture_CVPRF_2026_paper.pdf) | Learned importance independent of routing confidence, variable real/virtual expert budget; OCR/NLP more sensitive than general images/video. Full multimodal training, not a drop-in training-free port. | Training-free compatibility could differ, but is not sufficient novelty by itself. Requires a separate cause and Pareto improvement. |
| Semantic load and modality-aware EP capacity | [MACS](https://arxiv.org/abs/2605.05225) | Hidden-feature entropy weighting, calibrated expert modality groups, batch modality ratio, local semantic rerouting and fail-safe dropping. Qwen/InternVL/Kimi, EP8 DeepSpeed. | Distinguish directly measured request latency from layer-only speed and assignment-count proxies. Do not rebrand semantic capacity/rerouting. |

## Read-depth notes

AnyExperts: read methods, training objectives, evaluation and ablations. Its
importance estimator is trained jointly, with hidden-state modulation, virtual
experts and auxiliary objectives. A post-hoc threshold patch is not a faithful
AnyExperts baseline. The paper explicitly reports stronger OCR/NLP sensitivity;
merely finding ChartQA degradation is insufficient successor novelty.

MACS: read formulation, all three mechanisms and efficiency analysis. Semantic
weight is based on hidden-feature entropy, not router entropy. Its text uses
both image-wise and batch normalization language; do not silently choose one
for an implementation. Its reported efficiency figures prominently measure
MoE-layer stages; do not transfer those numbers to our request E2E. No MACS
implementation is in scope unless needed as a tightly bounded causal control.

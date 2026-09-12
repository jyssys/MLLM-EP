# Prior-art and collision audit

## Evidence date and scope

Audit date: 2026-09-12. This file distinguishes the measured LLaDA2.0-Flash
substrate from method claims. Links point to primary papers or official code.

## Model/runtime substrate

- [LLaDA2.0](https://arxiv.org/abs/2512.15745) introduces the 100B sparse
  diffusion LM studied here.
- [dInfer](https://github.com/inclusionAI/dInfer) is the official inference
  framework and documents the LLaDA2.0-Flash four-GPU command as tensor
  parallel. It does not establish sparse EP merely by using four GPUs.
- [DeepEP](https://github.com/deepseek-ai/DeepEP) supplies the sparse
  dispatch/combine transport used by the bounded true-EP4 bridge.

## The strongest direct dLLM collision: Epoch

[Epoch](https://arxiv.org/abs/2609.09748) already treats a diffusion block as
a compilation unit. Its Expert Atlas, Liveness and FreshLane mechanisms retain
dense logical state while carrying only live/new/refresh-required token-expert
work through EP. Therefore the following are **not** novel candidates here:

- removing accepted/dead/stable positions from every routed-expert wave;
- compiling active expert support per layer;
- propagating a compact fresh worklist through dispatch, kernels and combine;
- calling those savings a new form of dLLM liveness.

Candidate F is retained only as a measured residual check. Any oracle that
counts work Epoch already removes is excluded from the successor score.

## Candidate-by-candidate collision matrix

| Candidate | Closest existing axis | Collision assessment |
|---|---|---|
| A. Denoising-aware expert work packing | Grouped-GEMM/continuous batching and generic expert batching | Packing rows for the same expert is established. A paper gap requires a measured repeated-refinement cadence failure that ordinary continuous batching cannot capture. |
| B. Refinement-aware EP microbatch coalescing | Continuous batching; [dInfer](https://github.com/inclusionAI/dInfer) batched block decoding | Static large batching is not novel. A gap would require phase/iteration deadlines plus bounded waiting to beat the strongest static dInfer batch policy. |
| C. Block-scoped replication | Predictive expert replication and dynamic placement | Strong collision risk. Exact reusenant residency must beat copy/HBM cost and show dLLM block-scoped persistence not captured by generic hot-expert replication. |
| D. Denoising-aware load shaping | EPLB, communication-aware routing, dynamic replication | Near-tie rerouting or hot replicas are crowded and may change routing. Only an exact, large residual tied to refinement phase could survive. |
| E. Phase-adaptive parallel topology | Hybrid/dynamic TP-EP; [HD-MoE](https://arxiv.org/abs/2509.09420), [HDA-MoE](https://arxiv.org/abs/2609.08682) | Current papers target heterogeneous/NMP accelerators and generic routing. A CUDA-cluster dLLM crossover is a phenomenon, but per-phase switching must include incompatible weight-layout and state-transition cost. |
| F. Fresh-work residual | [Epoch](https://arxiv.org/abs/2609.09748) | Direct collision unless a residual remains after Epoch's exact work definition. |
| G. Router/dispatch decoupling | Generic prefetch, CUDA-graph metadata caching, DeepEP overlap | Route-plan prediction alone is not new. Exact delta metadata must be both stable and request-critical. |
| H. Communication/compute pipeline | DeepEP overlap and serving overlap systems | Generic overlap is crowded; only a measured dependency-safe dLLM cross-request window with >5% direct E2E value is eligible. |
| I/J. Locality/support-aware request grouping | MoE request scheduling and expert-affinity batching | Similar-support batching is not itself novel. It must beat token-count batching without wait/SLO cost. |
| K. Phase precision adaptation | Mixed precision and phase-adaptive diffusion compute | Requires a new quality/latency frontier, not simply enabling FP8. |

`MoE Parallel Folding` is relevant background for different TP/EP mappings,
but its primary setting is large-scale training rather than diffusion serving:
[paper](https://arxiv.org/abs/2504.14960).

## Durability rule

All communication-derived oracles are recomputed at communication cost
1.0/0.75/0.5/0.25. A candidate that disappears under 2x faster communication
is recorded as backend-fragile even if its current-runtime oracle is large.

## Current novelty boundary

At this point the defensible output is **characterization**: the official TP4
path and a capacity-faithful sparse EP4 path can have opposite winners as
request batching changes. It becomes a method claim only if a feasible policy
recovers at least 12% direct E2E or meaningful throughput beyond best static,
survives transition costs, and is not subsumed by Epoch or ordinary batching.

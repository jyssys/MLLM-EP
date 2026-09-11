# dLLM EP2 denoising-adaptive EP policy PoC

Final status: **EP2 NO-GO**.

This two-H100 mechanism screen verified true TP1/DP2/EP2 LLaDA-MoE execution,
measured fixed-work scaling from local M=2 through 256, validated vLLM Naive and
DeepEP HT communication paths, captured real dInfer mask and physical-work
trajectories, and computed the best-static/per-forward oracle.

The central hypothesis fails for two independent reasons:

1. In stock dInfer, active/unresolved positions are a decoder state, not the
   number of positions forwarded through the model. Within each diffusion
   block the physical MoE M stays constant while the mask count collapses.
2. DeepEP HT is the best full-request static backend in every early/middle/late
   aggregate. Perfect future switching saves only 0.723 ms of traced MoE time,
   or an optimistic 0.144% of the clean request median.

The 6.01% clean median gain of DeepEP HT over Naive is a static-backend result,
not a denoising-adaptive policy result. Dynamic switching is not implemented.

Detailed reports:

- `ep2_runtime_audit.md`
- `heavier_regime_scaling.md`
- `backend_regime_comparison.md`
- `denoising_adaptive_ep_policy.md`
- `epoch_prior_art_boundary.md`
- `final_decision.md`

Result root: `poc_dllm_ep2_policy/results/ep2_20260911_170000/`.

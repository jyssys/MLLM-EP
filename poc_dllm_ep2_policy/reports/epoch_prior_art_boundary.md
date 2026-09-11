# Prior-art boundary

This is a boundary audit, not a novelty claim. The tested adaptive-policy
direction fails its economic gate.

| Work | What it changes | Boundary to this PoC |
|---|---|---|
| [dInfer](https://arxiv.org/abs/2510.08666) | Modular dLLM inference, threshold decoding, and KV-cache policies | Runtime substrate. Its official LLaDA-MoE example is TP-oriented; this PoC verifies a minimally patched dispatch-capable TP1/DP2/EP2 path. |
| [Epoch](https://arxiv.org/abs/2609.09748) | Compiles the diffusion block and sends only live/new/refresh-required token-expert work through EP | Owns the large logical-work reduction that stock dInfer leaves on the table. This PoC intentionally keeps required work fixed and only chooses its communication backend. |
| [TEAM](https://arxiv.org/abs/2602.08404) | Uses temporal/spatial consistency to reduce expert activations and improve token acceptance | Changes activated experts/decoding; excluded here. |
| [DES](https://arxiv.org/abs/2602.00879) | Selects sequence-level expert coresets to reduce expert explosion | Changes expert selection/top-k-like work; excluded here. |
| [TIDE](https://arxiv.org/abs/2605.20179) | Uses temporal expert stability for interval-based CPU/GPU expert offload | Placement/offload under memory constraints; CPU offload is outside scope. |
| [DICE](https://openaccess.thecvf.com/content/ICCV2025/html/Luo_DICE_Staleness-Centric_Optimizations_for_Parallel_Diffusion_MoE_Inference_ICCV_2025_paper.html) | Staleness-aware displaced/interweaved EP and selective communication for diffusion image MoE | Changes overlap/synchronization and may consume stale state; not exact per-iteration backend selection for required fresh dLLM work. |
| [DeepEP](https://github.com/deepseek-ai/DeepEP) | Provides high-throughput and low-latency dispatch/combine kernels | Backend substrate. Merely selecting HT/LL by size is not novel, and the tested pinned LL integration is blocked. |

## Exact negative insight

The intended residual question was technically distinct:

> Given an already determined exact fresh worklist, should one dLLM request
> change EP transport policy as refinement reduces its active state?

On this stock dense-forward LLaDA-MoE path, refinement does **not** reduce that
physical worklist. M stays fixed within a block, so the intended state variable
does not induce a communication regime. Epoch's worklist compilation is not a
competitor that this PoC should reinvent; it is evidence that converting
semantic liveness into physical work reduction is a different, already claimed
problem.

Even if a future runtime exposes such a worklist, DeepEP already defines HT and
LL kernels. The publishable burden would be an EP4/current-runtime crossover
with material request headroom beyond static selection—not simply an active-M
threshold. EP2 supplies no reason to pursue that burden here.

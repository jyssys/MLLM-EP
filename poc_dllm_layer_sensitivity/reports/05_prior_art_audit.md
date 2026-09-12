# Adversarial prior-art audit

The measured gate did not produce a promoted method, so this audit focuses on
the closest work that would reject a layer/MoE stale-refresh claim.

| work | system/assumption broken | overlap with this PoC | remaining gap after measurement |
|---|---|---|---|
| [DICE, ICCV 2025](https://openaccess.thecvf.com/content/ICCV2025/html/Luo_DICE_Staleness-Centric_Optimizations_for_Parallel_Diffusion_MoE_Inference_ICCV_2025_paper.html) | Diffusion-MoE EP can overlap communication using stale activations, but staleness must be controlled at step/layer/token granularity. It adds interweaved parallelism, layer-level selective synchronization, and token-level conditional communication. | Direct collision for previous-iteration routed output and sensitivity-selected layer refresh. DICE is image diffusion rather than language dLLM, but purpose and stale-layer mechanism are too close for the raw candidate to be a clean successor. | Language unmask decisions and routed-MoE-only refresh are different evaluation variables, but measured post-liveness headroom is only 3.98% mean and trajectory safety fails. Insufficient to support a new paper core. |
| [Epoch](https://arxiv.org/abs/2609.09748) | The diffusion block, not one forward, is the compilation unit; only live/new/refresh-required positions traverse a fresh EP lane while gate logits are refreshed. | Removes the dead/accepted physical rows that inflate current-runtime phase cost. | Our independent residual must be measured on live work. Live-weighting cuts the strongest raw stale ceiling from 8.77% to 3.98%. |
| [ES-dLLM](https://arxiv.org/abs/2603.10088) | Training-free token skipping in early layers uses inter-iteration tensor variation and confidence. | Crowds confidence/update-magnitude based decision-critical layer/token selection in language dLLMs. | Our layer update magnitude is a weak sensitivity predictor and no fresh layer set passes the trajectory gate. |
| [dLLM-Cache](https://arxiv.org/abs/2506.06295) | Prompt and response intermediate states can be cached/adaptively refreshed across dLLM iterations. | Crowds generic periodic cached-layer/activation refresh. | An EP-routed-only contract could be narrower, but credible residual economics are below threshold. |
| [Window-Diffusion](https://arxiv.org/abs/2601.20332) | Active/buffer/far-field token windows enable token pruning and representation caching in dLLMs. | Crowds phase/token stability based caching and periodic refresh. | It is not EP-specific, but this sharpens the novelty requirement; this PoC does not meet it. |
| [Layer Collapse in Diffusion Language Models](https://arxiv.org/abs/2605.06366) | Early LLaDA layers show representational collapse, but the apparent outlier/redundancy remains causally important. | Strong methodological warning against equating similarity with dispensability. | Our 100B MoE result independently reaches the same caution: rel-L2/cosine have near-zero rank correlation with causal decision sensitivity. |
| [TEAM](https://arxiv.org/abs/2602.08404) | Temporal/spatial consistency can reduce activated expert work in MoE dLLMs. | Crowds decision-critical expert-work allocation at token/expert granularity. | This PoC holds routing fixed for its causal probes; no remaining large layer-level ceiling justifies a finer TEAM-like allocator. |

Generic early-exit, LayerSkip/SkipDecode-style layer dropping, expert skipping,
and MoE layer pruning further weaken a claim phrased merely as “skip
insensitive layers.” The only potentially distinct coupling would be:

> language-dLLM unmask-decision sensitivity × routed-MoE-only physical EP
> cost × phase-dependent exact refresh/verification.

That coupling is conceptually distinct, but the empirical gates fail: fresh
routed removal is decision-sensitive, stale reuse leaves large future
trajectory divergence, and the independent post-Epoch ceiling is below 5%.
Accordingly the work is not promoted to a novelty claim.

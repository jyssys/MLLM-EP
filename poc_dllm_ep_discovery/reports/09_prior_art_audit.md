# Adversarial prior-art audit

The audit asks what would reject each possible claim, even though no candidate reached PROMISING.

## Epoch boundary

[Epoch](https://arxiv.org/abs/2609.09748) explicitly treats a diffusion block as a compilation unit, routes only live/new/refresh-required positions through fresh expert work, and carries that work through dispatch, expert kernels and combine. Therefore liveness compaction and repeated dense execution are not novel here. This PoC's only potentially orthogonal claim would be making the remaining live sparse work efficient. Its robust residual is only 1.13--2.12% E2E, so that successor framing fails economically.

## Work-allocation methods

[TEAM](https://arxiv.org/abs/2602.08404) exploits temporal/spatial routing consistency to reduce activated experts and change speculative decoding work. [REFLEX](https://arxiv.org/abs/2608.01784) explicitly allocates expert computation according to refinement state and reports lower allocated expert compute. [DES](https://arxiv.org/abs/2602.00879) selects a sequence-level expert coreset to reduce unique activations. Consequently confidence-conditioned top-k/scope reduction or approximate expert reuse would enter an already occupied work-allocation space and also lacks the hypothesized confidence/router paradox in our data.

## Communication staleness

[DICE](https://openaccess.thecvf.com/content/ICCV2025/papers/Luo_DICE_Staleness-Centric_Optimizations_for_Parallel_Diffusion_MoE_Inference_ICCV_2025_paper.pdf) overlaps diffusion-MoE communication using stale activations and manages sensitivity by step/layer/token. Recasting volatile exact routes or hidden states as approximate delayed work would require a distinct language-model quality result; this PoC provides none.

## Grouped/fused expert execution

[MegaBlocks](https://github.com/databricks/megablocks) reformulates dropless MoE computation as block-sparse/grouped operations and recommends grouped GEMM on Hopper. [DeepEP](https://github.com/deepseek-ai/DeepEP) supplies high-throughput/low-latency EP dispatch and combine; its current upstream V2 further unifies modes and changes resource use. The measured substrate already performs one fused-MoE invocation over owner-local dispatched rows per layer/wave. Thus “sort rows by expert and run grouped GEMM” is baseline machinery, not novelty. Cross-wave queues or special 1--4-row kernels would need a dLLM-specific, material E2E result; the perfect upper bound here is below 3.4%.

## Exact novelty disposition

| possible claim | closest collision | disposition |
|---|---|---|
| remove dead refinement work | Epoch | direct collision |
| allocate fewer experts by refinement/confidence | REFLEX / TEAM / DES | direct or close collision |
| overlap by stale activation | DICE | direct collision/risk |
| group expert rows efficiently | grouped/fused MoE systems | generic and already partly baseline |
| post-Epoch dLLM live-row fragmentation | no exact collision established | conceptually distinct, but oracle too small |

No first-observation or novelty claim is made.


# Prior-art audit

## Work-removal boundary

| System | Work changed or removed | Collision with measured residual |
|---|---|---|
| TEAM | Uses temporal/spatial consistency, cached decoded state, restricted expert activation, and speculative candidates to reduce NFE/active experts | Base positive control; none of its removed work is claimed as new |
| Epoch | Compiles a diffusion block plan; routes only live/new/refresh-required positions through a fresh EP lane while recomputing gates | Directly attacks dead-position work, routing structure, and EP payload; kills “compile/reuse the same plan” framing |
| DES | Selects a sequence-level expert coreset to reduce unique experts and memory traffic | Any expert-set sharing/restriction candidate changes the same semantic axis |
| REFLEX | Allocates variable expert budgets by refinement state while keeping the base router | Kills “use fewer experts for stable tokens” as an independent direction |
| TIDE | Uses denoising-step expert stability for interval-based, I/O-aware expert placement under offload | Adjacent to temporal placement/migration; not relevant to the no-offload fused residual |
| DICE | Overlaps communication/computation with controlled activation staleness in diffusion MoE | Direct risk for stale/cross-step overlap; our PoC found no large independent overlap window |
| ODB-dLLM | Uses adaptive length prediction and jump-share speculative decoding | Direct risk for early speculative-branch commitment |
| Generic fused MoE / EP | Packs expert GEMMs and optimizes communication/fusion | Existing vLLM fused primitive already recovers the largest reference-path bottleneck |

TEAM reports up to 2.2x speedup by matching activated work to accepted tokens.^1
Epoch goes further at the serving/runtime boundary: it treats a diffusion block
as the compilation unit and carries only fresh work through EP.^2 DES changes
the expert set via a sequence coreset,^3 whereas REFLEX changes per-token expert
budgets based on refinement state.^4 These systems occupy the obvious semantic
work-removal space around TEAM.

TIDE is lossless but offload-oriented: it amortizes placement refresh using
temporal expert stability.^5 DICE targets diffusion-MoE communication overlap
and controls the quality cost of stale activations.^6 Neither creates novelty
for simply optimizing NCCL A2A or reusing stale branch activations.

## Adversarial result by candidate

1. **Early speculative commitment:** only 6.28% perfect oracle and adjacent to
   TEAM/ODB speculative control. No serious gap.
2. **Delta route-plan reuse:** 4.81% generous oracle and directly adjacent to
   Epoch block compilation/fresh-lane routing. Kill.
3. **Router elimination:** changes semantics; Epoch deliberately recomputes
   live gate logits. Kill.
4. **Communication elimination/overlap:** physically impossible upper bound is
   below 10%; DICE and generic MoE overlap systems occupy the mechanism. Kill.
5. **Expert packing:** existing fused expert primitive recovers it. Trivial.

## Novelty conclusion

The measured data do not expose an orthogonal >=5% feasible problem after the
trivial-fix attack.  The only >5% independent perfect oracle lacks a feasible
mechanism and is too close to existing speculative scheduling.  The correct
label is `POSITIVE-CONTROL-ONLY`, not a weak novelty claim.

## Sources

1. Wei et al., “[TEAM: Temporal-Spatial Consistency Guided Expert Activation for MoE Diffusion Language Model Acceleration](https://arxiv.org/abs/2602.08404),” 2026.
2. Zhu et al., “[Epoch: Compiling Diffusion Blocks for Sparse MoE Serving](https://arxiv.org/abs/2609.09748),” 2026.
3. Chen et al., “[Dynamic Expert Sharing: Decoupling Memory from Parallelism in Mixture-of-Experts Diffusion LLMs](https://arxiv.org/abs/2602.00879),” 2026.
4. Xia et al., “[REFLEX: Rethinking MoE Inference as Refinement-Aware Compute Allocation in Diffusion Language Models](https://arxiv.org/abs/2608.01784),” 2026.
5. Chen et al., “[TIDE: Efficient and Lossless MoE Diffusion LLM Inference with I/O-aware Expert Offload](https://arxiv.org/abs/2605.20179),” 2026.
6. Luo et al., “[Staleness-Centric Optimizations for Efficient Diffusion MoE Inference](https://arxiv.org/abs/2411.16786),” 2024.
7. PKU-SEC-Lab, “[ODB-dLLM official repository](https://github.com/PKU-SEC-Lab/ODB-dLLM),” accessed September 11, 2026.

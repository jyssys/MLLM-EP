# Previous negative evidence and non-repetition boundary

This PoC starts from four constraints established by earlier Qwen3-VL runs.

1. Critical-rank, same-expert pair coalescing exposed too little request-level latency mass. This study therefore measures all visual branch rows and never promotes a critical-rank-only saving.
2. Spatial/routing chunk policies and route-complementary request batching were negative. Spatial adjacency is used here only as a cheap proxy for an **output-space** compression oracle, not as evidence by itself.
3. CP-to-EP fine-grained splitting lost to repeated invocation overhead. Any performance prototype here must form one compact representative list and issue one dispatch/expert/combine, followed by one local expansion.
4. Hierarchical rebatching's feasible direct-BCT median was about 1.35%, with only about 0.50% attributable to MoE. No batching result is recycled as headroom for this PoC.

Two earlier output captures also make this a deliberately adversarial test:

- `visual_expert_functional_redundancy` found an effective routed-branch count near the original top-k and poor top-4 reconstruction.
- `intra_expert_visual_redundancy` found that cross-image same-expert visual outputs were not more clusterable than text outputs; at 50% representatives, even the output-oracle visual reconstruction was far outside its local quality gate.

Those experiments did **not** preserve a token's other branches while testing same-image 2D groups, did not gate on weighted combined-update error, and did not inspect sub-expert intermediates. These are the only remaining distinctions being tested.

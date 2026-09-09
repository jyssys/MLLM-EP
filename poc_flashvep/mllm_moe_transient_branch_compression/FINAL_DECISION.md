# Final decision

**Status: `NO_OUTPUT_COMPRESSIBILITY`.**

The candidate fails first at geometry and local quality: even an impossible
same-expert output-aware oracle reduces a median 0% of visual branch rows under
the registered tolerance, with maximum 0.0283% in any sample/layer. Qwen's
experts generally expand visual hidden-state differences, and exact centroid
representatives still produce about 32% affected-update error at group size
two.

It then independently fails at quality propagation and economics. A favorable
5%-row oracle already reduces next-layer route agreement to 97.83% after one
layer and 41.93% over six selected layers. In the measured high-resolution
serving regime, the *entire* expert-compute span is only 3.26% of request E2E;
the strict-safe oracle therefore maps to approximately 0% rather than the
required 10–12%.

The implementation and Kimi gates were not entered. This is not an environment
block: the intended TP2/DP2/EP4 DeepEP HT/TritonExperts path was directly
verified and produced 106,128 fresh visual routed assignments. The surviving
relative facts—visual pairs are slightly closer than matched Text pairs and
spatial route runs are common—have no deployable quality-safe or economic
mass. Sub-expert sharing, contribution weighting, layer subsets, exact
packing, codebooks, and cheap repair were each closed by an independent
quality, algebraic, prior-art, or direct-headroom gate.

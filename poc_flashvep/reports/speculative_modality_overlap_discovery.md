# Speculative Modality Overlap Discovery

**Decision: `SEARCH_SPACE_NO_GO` for the tested Qwen3-VL/DeepEP EP4
speculative-downstream-overlap branch.**

This was a bounded discovery probe, not an optimization implementation.  The
new live run used Qwen3-VL-30B-A3B-Instruct, BF16, TP2/DP2/EP4, linear expert
placement, eager execution, DBO off, prefix caching off, and the validated
`deepep_high_throughput` path on physical GPUs 1–4.  vLLM reported
`DeepEPHTAll2AllManager`, `DeepEPHTPrepareAndFinalize`, and TritonExperts; the
backend proof is in the result directory.

## Evidence and protocol

The mandatory spec was read before execution.  A fresh live multimodal capture
was completed after correcting two environmental issues: a conda interpreter
reported `has_deep_ep=False`, and the first DeepEP run did not install the
diagnostic hook because `sitecustomize` ran before the per-worker control path
was set.  The final run explicitly installs the existing local hook in each
worker before importing vLLM.  No installed vLLM/DeepEP source was changed.

Fresh capture: six real images (two each from the existing natural,
fine-grained, and chart/document suite), three representative MoE layers
(4, 24, 47), 919 selected token/layer samples (870 visual, 49 text), and 108
raw rank/layer files.  The earlier 24-image, 10-layer capture remains as a
reference and showed mean functional K 7.985 (visual) and 8.000 (text), so the
fresh result was not treated as a one-run anomaly.

Exact top-8 reconstruction from the fresh raw outputs was correct at the
expert-output level (minimum cosine 0.9999954, median 0.9999965; maximum
relative L2 0.00308).  This validates the attribution hook, not a model-level
speculative output.

## Partial completion quality

The exact target is the weighted top-8 expert output.  Plain partial output
keeps the original router weights and omits the remaining slots; the
renormalized variant rescales the selected weights.  Values below are fresh
live measurements (means over selected token/layer samples):

| Prefix | Vision cosine / relative L2 | Text cosine / relative L2 | Observation |
|---|---:|---:|---|
| top-1 | 0.589 / 0.825 | 0.655 / 0.740 | Not usable |
| top-4 | 0.866 / 0.514 | 0.924 / 0.350 | Not usable |
| top-6 | 0.945 / 0.324 | 0.967 / 0.212 | Not usable |
| top-7 | 0.974 / 0.214 | 0.984 / 0.137 | Not usable |
| 90% router mass | 0.995 / 0.051 | 0.936 / 0.194 | Tail-sensitive; text fails |
| 95% router mass | 1.000 / 0.0006 | 0.974 / 0.088 | Vision passes, text fails |
| 99% router mass | 1.000 / 0.00007 | 1.000 / 0.0000 | Effectively all 8 experts |

The preregistered quality gate was cosine ≥0.99 and relative L2 ≤0.05.  At
95% mass, pass rates were 99.9% visual and 73.5% text; at 99% mass, both pass
but average selected K is 7.97 visual and 8.00 text.  Thus the only
cross-modality quality-safe point retains essentially every expert
contribution.  Plain top-m and renormalized top-m had the same cosine (the
renormalization changed the error magnitude but did not rescue a prefix).

Output-affinity diagnostics did not change this conclusion.  A same-expert
output mean improved visual top-4 from relative L2 0.514 to 0.314 and visual
top-7 to 0.139, but remained far outside the gate.  A nearest same-expert
visual-token surrogate was worse (top-4 0.470, top-7 0.211).  These are
output-space controls only: hidden-state and weight-space affinity were not
captured, so they are not a claim about SpecMoE affinity.

No next-layer hidden-state or logits were captured.  Consequently propagation
and benchmark-quality metrics are **UNKNOWN**, rather than inferred from
expert-output cosine.

## Overlap/headroom gate

No downstream speculative execution was implemented.  The quality gate is
intentionally first: a provisional state that omits no expert work creates no
new dependency slack.  Therefore the quality-gated overlap oracle is zero,
not an optimistic latency extrapolation:

| Candidate | Quality gate | Expert work omitted | Created window | Perfect E2E oracle | Feasible oracle |
|---|---|---:|---:|---:|---:|
| top-1/2/4/6/7 or mass ≤90% | FAIL | non-zero | not eligible | 0% | 0% |
| mass95 | FAIL on text | small visual-only | not eligible | 0% | 0% |
| mass99 / exact top-8 | PASS | 0 | 0 | 0% | 0% |
| cross-modality safe candidate | PASS only at exact | 0 | 0 | **0%** | **0%** |

This is a gate result, not a claim that every hypothetical downstream
speculator has zero benefit.  It says the measured expert-output evidence
provides no quality-safe early boundary on which to build one.  Timing of
downstream QKV/attention overlap was consequently not justified.  A fabricated
E2E percentage would violate the spec.

## Research tree outcome

The maintained tree contains 12 nodes and the active frontier contains the
required independent branches.  Fresh measurements tested partial completion,
router-mass thresholds, output-affinity (SpecMoE-inspired), spatial-neighbour
surrogates, confidence conditioning, and the overlap gate.  Partial,
mass-threshold, affinity, spatial, cross-layer, and verification branches are
closed by the quality/headroom gate.  No finalist was promoted, so no Kimi run
was spent on a weak branch.

The direct systems distinction remains important: [SpecMoE](https://arxiv.org/abs/2604.10152)
uses draft/target verification and expert affinity primarily to reduce
CPU-offloaded expert migration and speculative tokens.  [Speculative MoE](https://arxiv.org/abs/2503.04398)
and [Speculating Experts](https://arxiv.org/abs/2603.19289) are adjacent
expert-prediction/verification work.  The untested space here would have been
partial distributed EP combine → provisional hidden state → downstream compute
while remote experts finish.  Similarity substitution itself is therefore a
baseline, not a novelty claim.

## Final judgement

The strongest new observation is negative but causal: in live Qwen3-VL
DeepEP, router mass thresholds only become output-safe when they select nearly
all eight experts, and this is true even though the exact reconstruction hook
is numerically sound.  Vision is not more tolerant than text under the tested
controls; the apparent 95% visual tolerance does not transfer to text.  A
same-expert or spatial output surrogate does not create a safe low-work prefix.

This closes the current speculative-overlap branch before any scheduler,
kernel, or Kimi implementation.  The artifacts distinguish observed live
expert-output evidence from unmeasured propagation/E2E behavior.


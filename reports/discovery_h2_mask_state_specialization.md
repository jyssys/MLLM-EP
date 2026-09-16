# H2 — Mask-State Expert Specialization

## Verdict

**HOLD (structural-only characterization).** MASKED, NEWLY_ACCEPTED, and DECODED positions route differently, including within the same position across refinements, but the effect is diluted to a very small change in the full physical EP workload.

## Evidence boundary

- Primary comparisons are within the current diffusion block; prompt/prefix and prior generated blocks are separate classes.
- Statistical units for inference are requests/blocks. Millions of token-layer rows are not treated as independent experimental samples.
- EP4 cost/load results are `SIMULATED-EP4-EP2-CALIBRATED`.

## Routing specialization

| Comparison | JSD P50 / P95 / P99 | TV P50 | Cosine P50 |
|---|---:|---:|---:|
| MASKED vs NEWLY_ACCEPTED | 0.667 / 0.975 / 1.000 | 0.771 | 0.351 |
| MASKED vs DECODED | 0.562 / 0.902 / 0.971 | 0.682 | 0.450 |
| PROMPT_PREFIX vs PRIOR_GENERATED_BLOCKS control | 0.234 / 0.431 / 0.717 | 0.427 | 0.102 |

The block-bootstrap MASKED-vs-DECODED JSD excess over the generic prompt/prior-token control is 0.357, with 95% CI [0.346, 0.369] over 422 request/block units.

These distributional JSD values are inflated by small per-state supports in some refinements and are therefore not treated as systems headroom by themselves.

## Same-position transition

For 263,606 tracked position/layer transitions:

- MASKED→DECODED top-k expert Jaccard: 0.231 P50, 0.600 P90/P95, 0.778 P99.
- Destination-rank-set Jaccard: 0.750 P50 and 1.000 P90–max.

Expert identity changes much more than physical rank support. This is important: semantic/router specialization does not automatically become an EP topology change.

## Isolated state versus full physical workload

When state classes are viewed in isolation, masked positions have EP4 max/mean load 1.722 P50 versus 1.500 for decoded positions, a 12.9% relative median gap. That is not the request-level physical effect because current-block state rows coexist with a much larger prompt/prior-block workload.

After mapping the state composition back into the full vanilla physical invocation, the EP4 max/mean change is:

| Statistic | Full-workload change |
|---|---:|
| P50 | 0.167% |
| P90 | 0.757% |
| P95 | 1.067% |
| P99 | 1.864% |
| Max | 5.616% |

By phase, the P50 effect falls from 0.437% early to 0.231% middle and 0.071% late. Mask ratio versus full EP4 max/mean Spearman correlation is effectively zero (-0.00003).

## AR-MoE control and interpretation

Token-age/type specialization also exists in AR MoE, so the dLLM-specific evidence is the same-position MASK→accepted→decoded transition. That transition is real, but its rank-level effect is largely absorbed by stable destination support and by the rest of the vanilla physical sequence.

The prescribed >=5% practical EP consequence appears only in one maximum-tail observation, not reproducibly in P50–P99. This is `H2_STRUCTURAL_ONLY`, mapped to `HOLD` in the required final labels. No mask-state placement or routing method is justified.

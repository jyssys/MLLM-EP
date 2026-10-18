# Stage 3: admission-only depth oracle

The oracle knows future online depth traces and, in its combined arm, true output lengths.
That information is unavailable to a deployable admission policy.
The oracle chooses a GPU before prefill and keeps each request on that GPU.
The local CDB execution obeys the selected cache and recurrence rules.

## Controls and search

The comparison includes causal queue/length/cost policies, true-length information, mean-depth information, and full depth trajectories.
The true-length-only heuristic uses calibration depth profiles, not future true depth.
That heuristic can perform worse than a generic baseline.
Therefore, its larger raw gap is not evidence of depth-specific headroom.
The main comparison uses the best observed non-depth control, including the true-length candidate.

Tiny C1/C2/C4 bursts have a solved symmetric placement optimum and zero additional gain.
Larger bursts use three complete one-request-move neighborhoods and bounded pair swaps.
The stronger search evaluates hundreds of placements and preserves the first search as v1 evidence.
It is best-known clairvoyant search, not a proof of global optimality.

| C | Best observed non-depth control | Control P95, ms | Oracle P95, ms | Model gain | 95% paired CI |
| --- | --- | --- | --- | --- | --- |
| 8 | round_robin | 7206.7 | 7136.6 | 0.973% | [-6.341, 2.220]% |
| 16 | round_robin | 12968.7 | 11803.0 | 8.989% | [-0.662, 12.894]% |
| 32 | round_robin | 20720.5 | 18953.4 | 8.528% | [0.000, 17.618]% |
| 64 | strong_cost | 33202.1 | 28962.4 | 12.770% | [3.696, 16.809]% |

The table contains validated-cost estimates. It does not contain measured GPU speedups.
The intervals resample paired request IDs at fixed selected placements.
They exclude model-fit uncertainty and selection uncertainty.
The natural-EOS traces have cap 256.
The large-burst oracle is exploratory and does not establish a universal upper bound.

## Physical hindsight-placement checks

| C | Repeats | Generic mean P95, ms | Hindsight mean P95, ms | Measured gain | 95% paired CI |
| --- | --- | --- | --- | --- | --- |
| 16 | 3 | 12948.7 | 11573.6 | 10.619% | [-0.466, 14.793]% |
| 32 | 3 | 21407.6 | 19165.3 | 10.475% | [-1.745, 19.211]% |
| 64 | 3 | 33051.6 | 31593.9 | 4.410% | [-8.298, 13.341]% |

These runs use the real online gate and physical DP4, not synthetic depth replay.
The assignments come from the frozen trace oracle.
Each check has three paired repeats with alternating arm order.
They reuse the pilot requests used for oracle selection. They are not untouched confirmation.
Their outputs and actual depths can differ after placement changes.
Thus they test a hindsight plan's physical realization, not exact portability of every B1 trace.
Generated-token counts, exact-output fractions, and natural K-token controls remain in `live_hindsight_confirmation.json`.
No benchmark-quality equivalence is established for those changed placements.

Depth-homogeneous grouping is an offline diagnostic, not a new production scheduler.
Joint microbatch optimization is `NOT_RUN`.
Free token migration, free KV movement, and independent cross-rank token stealing are forbidden.
No zero-cost bound is presented as a physical result.

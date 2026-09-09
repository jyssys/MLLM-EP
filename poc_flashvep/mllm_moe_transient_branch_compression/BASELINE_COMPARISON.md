# Baseline comparison and evidence boundary

| Baseline | Fresh result | Interpretation |
|---|---|---|
| Vanilla | Exact Qwen3-VL output; capture reconstruction cosine >=0.9999943 and exact checkpoint replay cosine >=0.9999983. | Correct reference. |
| Random sharing | Median safe reduction 0%; rare cancellation reaches at most 1.339% in one layer/sample. | Not stable or branch-local safe. |
| Spatial-only | 1D/2D output rel-L2 0.967/0.935; strict pair-safe 0.063%/0.091%. | Route runs do not imply functional reuse. |
| Hidden-sim-only | Output rel-L2 0.831; strict pair-safe 0.150%. | Best deployable geometric baseline, but unusable. |
| Routing-sim-only / route control | Route Jaccard adds only 0.034% held-out RMSE reduction; high-overlap pairs have 0 strict-safe cases. | No additional selector. |
| Impossible output oracle | Output rel-L2 0.816; strict pair-safe 0.162%; group-safe reduction median 0%. | Upper bound itself fails. |
| Whole-token pruning / FastMMoE | Not reimplemented after the branch-output upper bound failed. Official FastMMoE uses token pruning/merging and expert-width reduction, a different semantic contract. | Prior-art comparator, not performance evidence in this PoC. |
| MoDES-style low-contribution removal | Fresh exact zero-output control at the same row budgets is only slightly worse than output-oracle sharing. | Sharing does not create a distinct Pareto frontier. |

No speed result from an output-mismatched run is admitted. The full DeepEP
compact-list prototype was intentionally not implemented: the spec forbids it
when both the >=15% strict-safe row gate and >=8% feasible direct-E2E gate
fail. Likewise, Kimi validation is gated off because Qwen never reached the
promotion criterion.

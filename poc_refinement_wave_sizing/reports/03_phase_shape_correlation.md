# Phase-to-Physical-Shape Correlation

On the canonical mini32 scheduling trajectory (65 waves), decision-live ratio has
the following Spearman correlations:

| physical/cost feature | rho |
|---|---:|
| physical rows | 0.915 |
| active experts | 0.758 |
| rows per active expert | 0.911 |
| tiny-expert fraction | -0.771 |
| rank fanout | -0.378 |
| rank-load CV | 0.273 |
| dispatch bytes | 0.917 |
| whole model-forward time | 0.825 |

The correlations establish that refinement progress is associated with a changing
aggregate sparse shape. They do **not** establish that phase itself should select
a different mini size. Ready-pool size is the shared cause for much of the change:
as requests finish, both physical M and live work fall. Per-request physical M
remains 32.

Thus the first two links in the hypothesis chain are only partially supported:
logical state changes strongly, and aggregate physical shape changes, but current
runtime liveness does not compact the per-request EP payload.

Evidence: [`PHASE_CORRELATIONS.csv`](../PHASE_CORRELATIONS.csv) and
[`PHASE_SHAPE_SUMMARY.csv`](../PHASE_SHAPE_SUMMARY.csv).

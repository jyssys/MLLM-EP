# 02 — Near-tie EP-cost heterogeneity

Real top-k routes from five representative MoE layers (1, 8, 16, 24, 31)
were joined to each captured unmask decision using the latest exact MoE
invocation, source-row partition, and token position. No expert IDs, router
weights, or routing decisions were changed.

At delta 0.02, the observed selected/alternative comparisons number 89 for
GSM8K and 59 for partial HumanEval. Expert sets differ in 100% of these
pairs. Median rank-vector L1 difference across the five sampled layers is
10 and 12 assignments, respectively; median difference in remote assignments
is 2 and 3. Thus near-equal confidence does *not* imply the same EP route.

For the same position across adjacent baseline refinement forwards, median
sampled-layer rank-vector cosine is 0.9762 on GSM8K and 0.9729 on the partial
HumanEval trace. Rank-level cost is fairly persistent even though exact
expert identity changes; this is a descriptive stability result, not a
held-out future-latency predictor.

Crucially, C1 total expert assignments per position are fixed by top-k=8:
five sampled routed layers give 40 assignments, and all 32 routed layers
give 256 if executed. Choosing a different position changes *which* ranks
and experts receive work, not this total. C2 remote assignments and C3
max-rank contribution differ, but C4 calibrated rank service time and C5
counterfactual next-wave GPU latency were not established because the oracle
screen failed. The previous matched EP4 composition replay at the same
production scale also found tiny wall-time response to rank complementarity.

`TOKEN_EP_SIGNATURES.parquet`, `NEAR_TIE_EP_PAIR_DIFFERENCES.csv`, and
`TEMPORAL_POSITION_EP_COST.csv` contain the captured diagnostics. HumanEval
rows are marked as partial trajectory in the aggregated parquet artifact.

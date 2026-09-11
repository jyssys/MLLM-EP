# Final decision

## Status: EP2 NO-GO

**Answer to the decisive question: No.**  In this measured substrate, there is
no independent post-Epoch MoE/EP work-removal opportunity whose EP2 evidence
supports a credible path to 10%+ request E2E on EP4.

## Evidence in one table

| Finding | Result | Interpretation |
|---|---:|---|
| True runtime | TP1/DP2/EP2, 32 of 64 experts/rank | mechanism screen is valid EP, not TP |
| Clean request median | 620.63 ms (3 restarts, CV 5.01%) | request denominator |
| Whole MoE share | 53.77% | MoE is material |
| Stock physical assignments | 884,736 | full M=128, top-k=8, all iterations/layers |
| Epoch-equivalent reduction | 52.20% assignments | large but already solved prior-art space |
| Epoch-equivalent E2E oracle | 2.68% median | EP2 startup erases most assignment benefit |
| Fresh top-k set same | 13.87% | weak exact route reuse |
| Fresh destination set same | 98.56% | coarse metadata stable, payload not reusable |
| Fresh branch identity persists | 59.62% | necessary but insufficient for reuse |
| Persistent expert-output change | median 28.0% (weighted 29.1%) | same expert output is not stable enough |
| Best independent perfect oracle | 2.693% (entire router free) | below 5% kill gate |
| Best branch-reuse oracle | 0.0233% E2E at <=5% one-step change | negligible and approximate |

## Gate application

The user's gate is `<5% perfect-oracle request E2E -> NO-GO`.  Every
independent post-Epoch candidate is below it.  Because these bounds already
assume perfect future knowledge, zero validation cost, and zero cache/metadata
overhead, a production implementation cannot reverse the result on EP2.

## What was learned

1. The current cache-off dInfer runtime has a very large logical/physical work
   gap as live masks shrink.
2. Epoch already names and physically removes precisely that work via
   Liveness-Shard plus FreshLane dispatch.
3. Inside the residual fresh lane, EP destination identity is stable but actual
   expert selection/output is not.  `expert/rank metadata stability != reusable
   expert computation`.
4. EP2's M=2--128 DeepEP/GEMM curve is fixed-cost dominated, so even large row
   reduction translates poorly to request latency.

## Exact evidence boundary

- **Measured:** true-EP2 ownership/path, clean request time, stage timing,
  temporal router/hidden/branch values, cross-rank agreement.
- **Offline calibrated oracle:** Epoch-equivalent E2E, router/layout/branch
  candidate bounds.
- **Not claimed:** implemented Epoch, candidate runtime speedup, benchmark
  quality, EP4 communication saving, EP4 E2E.

## EP2 -> EP4

Temporal redundancy structure and the live-work gap can be observed at EP2.
Absolute communication, rank fanout, payload cost, backend behavior, and speedup
must be remeasured at EP4 for any promoted candidate.  Since none is promoted,
an EP4 run of these candidates would be low-value.  A future EP4 study should
instead reproduce Epoch itself or examine a different structural source of
work, not relabel stable destination metadata as expert-output reuse.

## Recommendation

Do not implement delta routing, fresh-lane branch caching, or layer-selective
temporal reuse from this PoC.  If Epoch reproduction is a separate engineering
goal, it can still be worthwhile, but it is not a new research direction.

Result root:
`poc_dllm_ep2_residual_work/results/ep2_residual_20260911_191336/`

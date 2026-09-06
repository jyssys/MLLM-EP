# RETHINK checkpoint after twenty-one distinct diagnostic hypotheses

The live set now includes the audited H03–H12/H14/H15/H23/H25/H28 controls
plus the fresh H16, H20, H24, H26 and H27 diagnostics. This exceeds the
twentieth-test checkpoint; H31–H33/H36–H48 remain planned source-derived or
replication controls.

## What the negative space says

1. Equal-work DP partition, phase order, long/short placement, chunk residue,
   and within/cross-DP heterogeneity have not produced a stable multi-percent
   request-level MoE effect.
2. Attention/MoE co-tails are enriched in the trace, so a generic GPU/runtime
   event remains a stronger explanation than an EP-only law until controlled.
3. A common latency-regime transition (ANOMALY_01) and a stable ~4% arrival
   cadence effect (H12, still under test) carry more mass than the small local
   sampled-stage differences. These must be checked against request and stage
   semantics rather than promoted from a single scalar.

## Assumptions to abandon

- “More DP synchrony necessarily means slower requests.” The equal-work
  randomized controls do not support this.
- “Wave makespan is a proxy for every request.” H06/H10 demonstrate that
  request cardinality and completion spread can dominate wave metrics.
- “A sampled MoE stage is the critical path.” H05 local dispatch/MoE changes
  exceed its E2E change, so rank-critical and request-level joins are required.

## New non-cosmetic candidates

- **H45 common-state replay:** reproduce ANOMALY_01 on a fresh worker and a
  no-hook worker to test allocator/KV/clock state rather than A/B shape.
- **H46 turnover persistence:** determine whether request-set turnover changes
  the common state level after output churn is controlled.
- **H47 completion-spread causal control:** repeat H06 with one DP churn only,
  then compare tokens/s and per-request p99 rather than only wave makespan.
- **H48 mixed-phase state:** test whether the H11 mixed phase changes the next
  block's common latency regime.
- **H49 CPU-rendezvous path:** run the same equal-work block with async
  scheduling disabled; if the effect persists, async scheduler is not the
  explanatory variable.

The remaining campaign prioritizes these high-information state controls and
source-derived metadata/RPC observations. A candidate remains closed unless it
has a repeatable direct request-level effect and a causal mechanism.

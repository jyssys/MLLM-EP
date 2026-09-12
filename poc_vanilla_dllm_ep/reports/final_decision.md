# Final decision

## Label

**CHARACTERIZATION-ONLY**

## One-sentence reason

Vanilla SDAR's exact reference EP4 path has a real 18.9% clean latency penalty
and highly persistent routing/load, but numerical expert work remains fresh and
all nine independent method families have less than 5% feasible direct E2E
headroom; the strongest exact oracle is only 0.153%.

## Gate accounting

| Gate | Evidence | Result |
|---|---|---|
| Fair fixed-work topology baseline | 3 restarts; 1,166 forwards and 14,487,552 assignments each | PASS |
| True sparse ownership / remote execution | 128 experts; 128/64/32 per rank; real NCCL outbound/reverse A2A | PASS |
| Numerical and bounded quality | layer cosine ≥0.99999988; median bounded score 62.5% for all topologies | PASS |
| Direct candidate oracle ≥5% | best exact candidate 0.153% | FAIL |
| Candidate oracle ≥12% for prototype | none | FAIL |
| Independent novelty | strongest mechanisms collide with speculation/replication/Epoch/DICE spaces | FAIL |
| Faster-runtime durability | candidate headroom shrinks at 0.75/0.5/0.25× communication cost | FAIL |

## What was learned

1. The released small-block dLLM work shape does not amortize EP in the exact
   reference runtime; EP1 wins every M from 1 to 256.
2. Aggregate temporal routing/load prediction is easy, but exact dispatch plans
   and expert values remain dynamic. This is why load predictors look strong
   without enabling exact work reuse.
3. Physical expert shards are poor diffusion drafters. Four parallel local
   views do not supply a sufficiently accurate/covered agreement signal, and
   each draft repeats most of the dense model.
4. Communication is visible but not independently removable: replicas move
   expert critical work, deltas keep the same dense element count, and strict
   refinement dependencies expose no batch-one overlap slack.

## Do not pursue

- Rank-local refinement views with consensus/global verification on this SDAR
  physical partition.
- Multi-step local drafting without a fundamentally cheaper drafter.
- Block-local hot replicas as an independent paper core.
- Per-iteration EP1/2/4 switching for the measured M range.
- Dense temporal-delta dispatch, layer skipping, or wait-moving overlap.
- TEAM adaptation of any of the above: no vanilla candidate passed its gate.

## Scope boundary

This decision does not claim that optimized DeepEP EP4 or high-concurrency
serving has the same absolute latency. It says the proposed *independent
training-free method families* have no measured, durable ≥5% oracle here; a
faster communication backend only reduces their available mass. A future study
of production EP should first reproduce the topology characterization rather
than inherit the reference path's absolute stage shares.

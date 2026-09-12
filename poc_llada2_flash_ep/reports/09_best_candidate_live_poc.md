# Best-candidate live PoC

## Gate result

No new candidate qualified for live implementation. The specification permits
a complex prototype only for a request-level oracle of at least 12%; the
strongest novelty-eligible credible upper bound was perfect rank balancing at
5.34%.

Two large numbers were deliberately not promoted:

1. EP4 BCT improves 47.36% when the existing dInfer `mini_batch_size` is tuned
   from 4 to 8. This is an important baseline correction and a deployment
   recommendation, but it is a trivial existing knob.
2. Fresh-work compaction has a 25.00% optimistic full-routed-pipeline bound,
   but Epoch already implements this problem framing with Expert Atlas,
   Liveness, and FreshLane.

The largest communication-derived candidate, block-scoped exact replication,
has a 14.33% proportional fantasy bound but only a 2.99% zero-copy upper bound
after retaining the measured DeepEP fixed/startup floor. A replica manager
would add copy, HBM, planning, and synchronization cost. Implementing it would
therefore violate the oracle-first rule.

## Live work that was performed

The PoC did implement only the minimum substrate and diagnostics needed to
establish the bounds: true owner-rank EP4, a DeepEP primitive sweep, clean
paired topology/batch/microbatch runs, route/liveness traces, and bounded
quality evaluation. No production optimization, routing change, custom
communication kernel, or semantic work removal was implemented.

## Status

`NO_PROTOTYPE_BY_GATE`, consistent with final `CHARACTERIZATION-ONLY`.

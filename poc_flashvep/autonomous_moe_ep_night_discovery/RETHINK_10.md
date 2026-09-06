# RETHINK checkpoint after ten distinct live hypotheses

## Count and strongest evidence

The ten live hypotheses counted here are H01, H03, H04, H05, H06, H07,
H08, H09, H10 and H11. H03/H04/H05 used the audited equal-work controls;
H06 is still being interpreted with a request-level completion metric rather
than a wave-only metric.

## What assumptions are being repeated?

1. A shorter GPU critical span is assumed to imply lower user-visible
   latency. H06 shows that a wave's last completion can change while the
   per-request median does not.
2. DP partition and phase alignment are assumed to matter more than request
   composition. Randomized, equal-work H03/H04/H05 runs have not reproduced a
   stable >=3% request effect.
3. A high local sampled-MoE effect is assumed to survive the full request
   critical path. H05 currently shows a local sampled effect larger than the
   wave effect, so this must be treated as an instrumentation/critical-path
   question, not an optimization result.

## New system facts

- Equal-work controls are substantially more stable than the initial
  first-case-ordered observations.
- Completion spread and request median are different observables under
  heterogeneous output lengths.
- The low-perturbation observer is useful for directionality but carries a
  measured approximately 3% E2E tax against no-hook serving; future claims use
  within-server randomized comparisons.

## New non-cosmetic hypotheses

- **H37 completion spread:** equal aggregate decode work with one-DP output
  churn increases request p99/last-completion time without increasing request
  median; test with a one-DP heterogeneous output control.
- **H38 metadata tax:** per-layer expert metadata materialization and host-to-
  device count copies create a recurrent host cadence tax; instrument the
  allocation path and compare a storage-reuse diagnostic.
- **H39 kernel regime:** phase placement changes the selected packing/GEMM
  shape regime at matched work; correlate kernel identity and sampled stage
  duration, not only wave latency.
- **H40 whole-GPU co-tail:** attention and MoE tails co-occur because of a
  shared GPU/runtime event; use same-step non-MoE stage timing as a control.
- **H41 response semantics:** a large wave-completion effect can be an API
  completion/queueing artifact rather than model execution; compare serving
  step cadence, throughput and per-request completion.

## Next decision rule

Continue the live campaign through H28 and then run H36/H37 plus source-derived
metadata/scheduler controls. Any candidate below 3% direct request effect is
recorded and closed unless it exposes a new high-frequency mechanism.

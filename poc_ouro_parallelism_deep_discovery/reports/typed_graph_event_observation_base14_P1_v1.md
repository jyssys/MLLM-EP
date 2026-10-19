# Bounded typed-event observation: Base1.4B P1

Evidence class: `GPU_MICROBENCH`.
This result describes one controlled observer experiment on physical GPU5.
It does not establish a new inference architecture or task quality.

The [raw observation](../outputs/native_typed_graph_events_base14_P1_v1/typed_event_observation.json) retains before/during/after captures and replay envelopes.
The [analysis](../artifacts/typed_graph_event_observation_base14_P1_v1.json) retains independent numerical comparisons.
The runner and resource guard both finish successfully.

## Numerical and dependency checks

All four captured positions match logits and hidden states bitwise across the three lanes.
All sampled written KV values also match bitwise.
All captured values are finite, and all four top-1 tokens match.
These numerical checks do not establish natural generation or benchmark accuracy.

The graph retains 96 programmatic dependency records.
The runtime checks every original edge-data byte before the observation.
The added event cuts cross only existing default edges.
After removal, the original typed graph fingerprint matches exactly.
The record retains fingerprints, counts, and the original DOT.
It does not retain complete typed snapshots before and after insertion.

## Timing boundary

The last observed replay spans 7.709408 ms between its endpoint events.
Its four kernel-group intervals are 1.927072/1.925312/1.925920/1.931104 ms.
A fused normalization crosses the nominal recurrent boundaries.
These intervals are therefore not exact semantic pass timings.
Preserved programmatic dependencies do not establish their overlap duration.
No interval becomes an additive request breakdown.

The first instrumented envelope is 74.005219 ms because it includes graph mutation and reinstantiation.
It cannot estimate normal replay overhead.
The two later instrumented envelopes have median 7.715104 ms.
Four nearby clean later envelopes have median 7.696848 ms.
Their observed difference is 0.018256 ms, or a ratio of 1.002372×.
This is one small diagnostic, not a population estimate.

The internal events describe only the third replay.
Multiplication by replay count is forbidden.
Natural quality, candidate speedup, and candidate E2E remain `NOT_RUN`.

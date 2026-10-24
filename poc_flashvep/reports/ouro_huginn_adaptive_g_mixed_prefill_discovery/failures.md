# Failures and unresolved controls

## Heterogeneous prompt remainder failure

Attempt `dp_ouro_w1_hybrid_s0_gpu0` failed during the first DP burst campaign.
The executor rejected a batch with unequal loop counts.
The failing batch did not execute GPU model operations.
Earlier completed work in that attempt does not become a completed throughput result.
The original failure JSON, log, job definition, and source diff remain intact.

Mixed G1 can advance prompt rows by different amounts.
Two later pure prompt rows can therefore have different remaining recurrence.
The old prompt grouping patch clipped G separately for each row.
One immutable execution descriptor requires a common G for every row.
The scheduler did not reconcile those separately clipped groups.

The CPU regression recreates prompt depths 3 and 0 under Ouro R4 with nominal G2.
Before the fix, the real scheduler produced G1 and G2 descriptors together.
The executor raised the same execution-boundary error as the live attempt.
The test failed in 1.33 seconds.

The fix selects the minimum legal group across the selected rows.
Any decode row forces common G1.
The scheduler also recomputes prompt completion flags when it reduces a group's width.
It does not change token ranges, KV values, or model depth.
The CPU regression and the existing common-boundary test then passed.
The diagnostic workflow used a failing regression before the source fix.

The full DP retry uses new `_r1` attempt and barrier IDs.
The original failed attempt remains excluded from performance aggregation.
The retry's completion status belongs to `dp_scaleout.csv` and the final evidence audit.
An unfinished retry must not appear as a measured speedup.

The earlier fixed-grid cells completed without this descriptor error.
Their actual Full-depth ledgers remain available.
Their single target prompt does not recreate the later multi-prompt remainder conflict.
The final reports distinguish those source versions from the hardened DP source.

## Initial runtime outliers

The exploratory screen contains large first-repeat latency outliers.
The initial short warm-up did not cover every later batch geometry.
Some large outliers also remain after a longer warm-up.
Graph capture is therefore not a proved explanation for every outlier.
Host contention, lazy work, and runtime effects remain possible contributors.

The quiet confirmation repeats the complete workload during warm-up.
It records actual core and coda capture/replay/fallback increments.
It crosses GPUs and policy order across three process starts per shape.
It does not delete the exploratory outliers.
No initial apparent benefit receives promotion without this confirmation.

## Evidence limits

The mechanism text is one repeated source family, not a task-accuracy dataset.
Token agreement on that family does not establish universal batched fidelity.
Three process starts give limited tail uncertainty.
One contention start per worker count does not prove absence of all shared-host interference.

Huginn P8K exceeds the checkpoint's 4096-position configuration.
No unsupported RoPE extension or context override runs.
Huginn remains synchronous in every arm.
Ouro's native async path remains the stronger within-model reference.

State-hash interventions include an intentional diagnostic pause.
Their observed gaps do not qualify as clean delivered-token speedups.
They require equal source fingerprints before a matched-state interpretation.
CUDA event intervals include stream elapsed time, not only kernel service.
The reports do not add overlapping intervals into an E2E percentage.

Qualified open-loop capacity and benchmark accuracy require separate progression gates.
Missing results remain NOT_RUN, not zero accuracy or a simulated serving curve.

The optional Graphviz command failed because `dot` is not installed.
The editable DOT source remains available.
Matplotlib generates the corresponding editable SVG without that optional dependency.

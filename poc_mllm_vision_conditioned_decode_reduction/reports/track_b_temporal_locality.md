# Track B — Temporal Locality

## Evidence boundary

This report uses **COMMITTED_EXACT_CONDITION_A_TP0_ROUTE_EXPORT**.  The source contains condition A,
TP0, one logical decode row.  It does **not** use the old histogram compact
trace, which sums a real TP row and an idle-DP dummy row.

The historical tracer recorded checkpoints [1, 2, 4, 8, 12, 16, 20, 24, 28, 32, 40, 48, 56, 64, 72, 80, 88, 96, 104, 112, 120, 128], not every
decode step.  Recurrence is therefore a sparse-checkpoint structural diagnostic,
not a full W=1/2/4/8/16 cache hit rate.

## Route recurrence

- requests: 20
- route points: 21,120
- W16 all-split recurrence among checkpoint-eligible paths: 42.6370%
- W16 held-out recurrence: 42.8168%
- hidden vectors: **AVAILABLE**
- expert outputs: **AVAILABLE**

Route recurrence is necessary but insufficient for output reuse.  No local
error or quality claim is derived from this table.

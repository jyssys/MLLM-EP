# Oracle versus deployable allocation

Status: `PARTIAL_STRUCTURAL_ACCOUNTING`.
Model and runtime identities appear in `00_runtime_audit.md`.
No oracle speedup is a measured acceleration.

## Fresh structural observation

Two completed native EP4 sanity traces expose the actual token–expert assignments.
The analysis removes the smallest current routing weights retrospectively.
It preserves at least one edge per token.
It then recounts unique remote token–destination pairs.

At a nominal 10% edge reduction, remote-pair reductions were 2.8873% and 2.0966%.
At a nominal 20% edge reduction, they were 5.9551% and 5.6240%.
Per-layer integer rounding changes the actual edge fraction slightly.
The raw artifact retains the exact removed-edge counts.

These are `TRACE_ESTIMATE` results from two sanity prompts.
They do not predict benchmark quality.
They do not measure communication latency reduction.
They show why expert-edge savings need not equal dispatch savings.
Multiple selected experts can share one destination message.

## Deployment constraints

A policy can avoid an activation message only after removing every selected edge for that destination.
Collective startup remains even when payload becomes smaller.
The fixed shared expert and dense paths also remain.
Global critical-rank selection may require additional synchronization.
That synchronization must enter the deployable cost estimate.

| Comparison | Oracle ceiling | Realistic estimate | Measured isolated E2E gain |
|---|---|---|---|
| P3 versus P0 | NOT_MEASURED | NOT_MEASURED | NOT_MEASURED |
| P4 versus P0 | NOT_MEASURED | NOT_MEASURED | NOT_MEASURED |
| P5 versus P3 at equal edges | NOT_MEASURED | NOT_MEASURED | NOT_MEASURED |
| P6 hindsight oracle | NOT_MEASURED | NOT_APPLICABLE | NOT_APPLICABLE |

Do not promote an oracle-only result to a runtime implementation.
The specification requires conservative E2E headroom of at least 5%.
It also requires a credible additional EP-aware opportunity of at least 2% over P3.
Both gates remain untested.

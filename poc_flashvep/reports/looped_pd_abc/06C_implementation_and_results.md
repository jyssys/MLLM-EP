# Track C: measured implementation results

DRAFT: final scientific review is pending. No GO claim is accepted.

Campaign status: `RUNNING`. Evidence snapshot: 2026-10-28T19:56:43.852559+00:00.

These are actual GPU measurements from completed runs.
Unfinished and failed runs do not contribute latency values.
Natural generation permits EOS and preserves the declared output cap.
Cap termination is not proof of answer completeness.
A lower latency with different output IDs is not an accepted lossless speedup.

NOT_MEASURED. Repeated matched comparisons are pending.

## Mechanism and controls

C changes initial placement only. A and B remain disabled.
The main feature ablation compares loop-state placement with calibrated service-time placement.
Both policies receive the same legal progress telemetry.
C2 migration and dynamic role reassignment are NOT_MEASURED.
A one-prefill/one-decode pool has no initial routing choice.

## Limits

Official-reference sequence agreement is incomplete for all three models.
Full GSM8K test accuracy and HumanEval accuracy are `NOT_MEASURED`.
A closed-loop cohort does not establish sustainable open-loop capacity.
No global queue simulator or infinite-bandwidth serving oracle is validated.
GPU kernel times and overlapping transfer durations must not be added to E2E.
Short profiled runs are separate from clean serving runs.

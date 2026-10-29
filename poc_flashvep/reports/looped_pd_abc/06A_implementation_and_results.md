# Track A: measured implementation results

DRAFT: final scientific review is pending. No GO claim is accepted.

Timing correction: legacy foreground-I/O timings are invalid for clean method comparisons. Read `timing_correction.md`. Canonical nonblocking retiming is pending.

Campaign status: `RUNNING`. Evidence snapshot: 2026-10-28T22:34:18.763953+00:00.

These are actual GPU measurements from completed runs.
Unfinished and failed runs do not contribute latency values.
Natural generation permits EOS and preserves the declared output cap.
Cap termination is not proof of answer completeness.
A lower latency with different output IDs is not an accepted lossless speedup.

NOT_MEASURED. Repeated matched comparisons are pending.

## Mechanism and controls

A transfers full-prompt recurrent state at an intermediate cut.
The destination completes prefill and then executes full-depth decode.
The matched A control uses the same whole-prompt chunk and staged execution.
Generic logical-depth migration is a novelty counterexample.
The cut sweep is separate tuning evidence, not an accepted single-start gain.

## Limits

Official-reference sequence agreement is incomplete for all three models.
Full GSM8K test accuracy and HumanEval accuracy are `NOT_MEASURED`.
A closed-loop cohort does not establish sustainable open-loop capacity.
No global queue simulator or infinite-bandwidth serving oracle is validated.
GPU kernel times and overlapping transfer durations must not be added to E2E.
Short profiled runs are separate from clean serving runs.

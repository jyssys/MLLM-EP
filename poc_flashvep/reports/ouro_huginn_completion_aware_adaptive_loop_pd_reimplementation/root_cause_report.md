# Previous Scheduler: request coverage and starvation

## Scope

This report analyzes the six frozen N128 Loop confirmation runs.
The arrival seeds are 2026110911, 2026110912, and 2026110913 for each model.
The replay uses every measured request, not only successful migrations.
The request inventory contains 768 rows.

The source is `poc_flashvep/idle_intervention_dynamic_loop/runtime.py` from the previous campaign.
The previous published commit is `0adb7491c866d6d66cf71ea30dca63f82c46d135`.
The new inventory is `old_request_gate_inventory.csv`.
The event summary is `old_gate_summary.json`.
The host waiting records are `old_continuation_gaps.csv`.

## Observed gates

| Observation | Ouro, N384 | Huginn, N384 |
| --- | ---: | ---: |
| Initial arm evaluations | 384 | 384 |
| Armed requests | 8 | 15 |
| Initial evaluations with D unavailable | 370 | 351 |
| D-unavailable advertisements within an accepted request's commit-to-first-output interval | 253 | 250 |
| Boundary offers | 18 | 40 |
| Prepared offers | 5 | 5 |
| Actual committed requests | 3 | 5 |
| Whole-suffix-budget rejections | 6 | 18 |
| Continuation-slot rejections | 5 | 7 |
| P-pressure rejections | 2 | 5 |
| Execution-budget rejections | 0 | 5 |
| Source continuation after stale-or-expired commit | 2 | 0 |
| Source lease-timeout events | 0 | 0 |

Initial conditions can overlap. Their counts are not additive rejection attribution.
The residence overlap is a host observation, not an exact reconstruction of every destination queue state.
The combined stale-or-expired reason does not distinguish lease expiry from a changed execution budget.
The relay timestamps cross the 10 ms expiry boundary in both ambiguous Ouro cases.
The log therefore does not identify their unique cause.

## A. One-time arming suppresses later opportunities

`source_planned()` initializes `idle_armed` only when the request lacks that attribute.
A false value selects the full remaining Native grouped execution.
The code does not reevaluate that false value at a later source scheduling call.
An earlier boundary inside an already submitted atomic graph cannot be recovered retrospectively.

`can_arm()` combines candidate eligibility with destination availability and predicted budget.
This conflates a future source opportunity with current destination admission.
Most initial decisions observe D unavailable.
The single-slot rule and persistent suspension can both produce that advertisement.

The redesign separates checkpoint monitoring from destination acceptance.
It records the overhead of exposing boundaries.
It does not claim that additional boundary monitoring is free.

## B. One resident request can exclude later candidates

The destination marks capacity unavailable when its continuation occupancy reaches one.
An accepted request remains in that occupancy while receiving or executing Prefill.
The first-seed Ouro request waits 23.207262 s between continuation submissions.
The corresponding Huginn request waits 85.872055 s.
Both remain unfinished during most later arrivals.

Increasing the count alone would create a longer waiting queue.
The redesign requires completion reservations and includes existing accepted work in new admission predictions.
The two-slot cap is a secondary resource bound, not a claim of two requests' safe execution capacity.

## C. The old completion test excludes scheduling delay

The old admission check uses guarded serial remaining cost and handoff cost.
Its predictive window scales a possible quantum across a fixed horizon.
It does not reserve the execution opportunities needed after commit.
It does not include the actual queue of future Decode-safe opportunities.

The destination can therefore accept work during an empty-batch advertisement and later fail to schedule its remainder promptly.
The recorded import-to-output intervals exceed the serial completion estimates.
They do not prove a particular amount of physical idle time.

The redesign prices permitted quanta and repeated Decode intervals.
It records predicted completion, realized completion, progress age, and reservation misses.
These empirical predictions do not guarantee safety under unknown future arrivals.

## D. Suspension also blocks already accepted work

The old `deadline_margin()` returns zero when suspension is active and Decode requests remain.
The suspension state persists until the measured cohort finishes.
There is no continuation completion deadline or hard progress reservation.
The old controller can therefore preserve Decode output while retaining an accepted Prefill request for many seconds.

The new path separates a stop on new admission from execution permission for committed ownership.
Committed work still requires a live deadline-safe quantum.
A completion miss remains a recorded failure.
The implementation does not discard work or invent an unqualified reverse transfer.

## E. Host time is not GPU time

The old submission-completion records include Python submission and event polling.
They are not kernel unions.
The long gaps identify missing continuation submissions, not physical GPU idle.
Other requests produce outputs during these gaps.

The new campaign records separate CUDA Event controls and short CUPTI controls.
Invalid or intrusive captures remain diagnostics.
The final physical-headroom report must state which evidence passes the interference checks.

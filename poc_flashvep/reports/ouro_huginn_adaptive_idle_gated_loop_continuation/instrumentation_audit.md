# Instrumentation audit

## Evidence classes

Kernel activity, model-stream enclosures, scheduler selections, and deadline budgets are different measurements.
A CUDA graph envelope includes gaps between graph nodes.
An event enclosure also includes submission gaps and stream waits.
Neither enclosure is an exclusive kernel union.
Its complement can establish a conservative interval without observed model compute.
It does not establish that a future Decode request cannot arrive in that interval.

## Retained failed controls

The first activity collector produced an empty Ouro D trace in an otherwise completed Native serving run.
The analyzer initially constructed one eight-second complement from that empty trace.
That derived interval was invalid.
The corrected analyzer records `INVALID_ACTIVITY_COVERAGE` and does not construct idle gaps from incomplete coverage.
The superseded derived table remains under `invalid_missing_coverage_gap_table_v1.csv`.
Raw activity files remain unchanged.

Activating the collector under the model's CUDA device produced complete records in an N24 smoke test.
It did not fix coverage in all N128 runs.
Therefore, device-context activation alone is not an established solution.

One Huginn N128 trace requested ten buffers but returned only eight buffers.
The reported dropped-record count was zero.
Thus, a zero dropped-record count alone did not establish complete trace coverage.
The revised screen also checks buffer returns and native graph replay counters.

## Controlled recovery tests

The recovery tests separate three changes.

1. Select the model's device during activation and final export.
2. Force the final buffer flush after serving and model-stream synchronization.
3. Replace per-kernel collection with bounded model-submission event enclosures.

The second test changes only final export behavior.
It must not flush or synchronize to export data during the measured cohort.
The third test does not claim an exclusive physical kernel timeline.
It records every model submission in a bounded window with preallocated timing events.
Its central-window complement is a conservative lower bound, not a complete physical idle distribution.

NVIDIA documents that a default flush can retain an entire buffer when one record remains incomplete.
NVIDIA recommends a forced flush before session termination.
This behavior motivates the test. It does not prove the cause of every missing trace.
See [CUPTI 12.8 Activity API](https://docs.nvidia.com/cupti/12.8/api/group__CUPTI__ACTIVITY__API.html).

## Dispatch correction

The first two `collector_device1_flush*_launches10000` configurations used the wrong dispatch field.
They completed Native N8 serving, not the intended collector self-test.
Their dispatcher success does not establish collector correctness.
The corrected `_v2` configurations use `worker=operator_micro` and the isolated collector worker.
The corrected tests verify exactly 10,000 graph envelopes and 10,000 eager kernels.
Both corrected tests passed in the first device-context comparison.

## Unchanged qualification

The local-tail screen remains an absolute five-percent screen.
It includes TPOT p95, ITL p95, ITL p99, and maximum ITL.
It includes the whole cohort, three fixed arrival windows, and the central capture region.
A faster instrumented tail does not establish equivalence when its absolute change exceeds five percent.
Unqualified traces remain diagnostic.
No failed trace becomes a Safe Slack distribution.

The three context-corrected paired starts retain their original seeds and order.
Source archives identify the collector revision for each run.
Additional event-enclosure trials use fresh paired seeds and counterbalanced order.
They do not replace failed seeds from the earlier trials.

## P pressure correction

Huginn uses synchronous owner scheduling.
Its submitted Prefill can leave the request in a transfer-related phase before GPU work completes.
The asynchronous in-flight ticket list can therefore be empty during a real Prefill service enclosure.
The revised pressure analysis retains native host service enclosures as a separate field.
It joins D-credit-clear P-slot waiting with those enclosures or pending Prefill events.
It does not label that host evidence GPU compute saturation.
GPU activity and deadline safety still require their own evidence.

## Final export and current collector build

The drained Ouro pilot returned 2/2 activity buffers and zero invalid timestamps.
Its local-tail perturbation screen still failed.
The drained Huginn pilot returned 10/10 buffers but contained one invalid timestamp record.
Both pilots completed their response cohorts before final export.
The Huginn export failure does not establish a serving-body progress stall or a model-state failure.
The cause of the invalid timestamp remains unresolved.

The collector retains invalid timestamp details separately in its current source revision.
The current source compiled and passed the `_v3_invalid_dump` synthetic plumbing test.
That test does not validate an N128 model trace or explain the earlier invalid record.
Earlier source archives and shared-library hashes remain available.

## Minimal-host controls

The minimal-host controls replace repeated live HBM queries with CPU-only owner snapshots.
They also omit optional transport observation. They retain native transfer and ACK handling.
All eight full/light/minimal/event pilot cohorts completed their output, full-work, and lifetime checks.
Neither minimal-host event pilot passed the frozen absolute local-tail screen.

The Ouro local ratios ranged from 0.9809017180258613 to 1.068204736202201.
Its cohort-duration ratio was 1.006933353856532.
The Huginn local ratios ranged from 0.6371727660081676 to 0.8803734567781655.
Its cohort-duration ratio was 0.9922173837982168.
The faster Huginn tails do not establish measurement equivalence.

Native A/A repeats measure run-to-run variation separately.
They do not retrospectively qualify failed observer traces.
The GC interval inventory excludes startup intervals and records worker role when available.
Temporal overlap between a host pause and a token delay does not establish causality.

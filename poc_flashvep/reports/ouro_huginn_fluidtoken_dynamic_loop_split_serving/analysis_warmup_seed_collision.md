# Startup seed collision in latency analysis

## Measured failure

The first N128 native Ouro run completed all 128 measured requests.
Its worker returned `COMPLETE` with return code zero.
The primary controller then failed in latency analysis:

```text
AssertionError: Duplicate device state for one role and seed
```

The run name is `f_ouro_medium_native_start1_128`.
The failure occurred after GPU execution, not during model evaluation.
The original token outputs, device records, and worker log remain unchanged.

## Cause and regression test

The initial startup requests reuse the first measured generation seeds.
The latency analyzer filtered device records by seed only.
It therefore counted startup and measured records as duplicate requests.

For seed `2026110362`, the recorded Prefill submissions had distinct request identifiers.
The startup submission occurred 11.977417 seconds before the measurement marker.
The measured submission occurred 0.077104 seconds after that marker.
This reproduces the exact analyzer failure without another GPU run.

The correction retains only submissions at or after `measurement_begin.json`.
It still requires each expected measured seed exactly once per role.
It still rejects genuine duplicate measured records.
The raw startup records remain intact.

The CPU regression tests cover both cases.
One test retains a measured request with the same seed as a startup request.
Another test rejects two measured requests with the same seed.

## Recovery scope

The corrected analyzer can reuse the completed N128 execution.
No model math, checkpoint, output, KV state, or SLO threshold changes.
The primary controller resumes from the immutable completed namespace.
The existing static controller can retain its already loaded analyzer until exit.
A one-shot supervisor reloads it only after exit if its ledger remains incomplete.
The supervisor does not stop any GPU worker or reset the campaign deadline.

This correction is not a serving speedup.
Clean-latency acceptance still requires the matched marker-overhead control.

## Accepted-boundary and payload analysis

The first N128 Loop3 Ouro run also completed all measured requests.
Its post-run work accounting then rejected a reused startup seed:

```text
AssertionError: Count one actual accepted boundary per request
```

The run name is `f_ouro_medium_loop3_start1_128`.
The accepted-boundary file contains 140 records, including startup records.
For seed `2026110363`, two records have different native request identifiers.
Their monotonic timestamps are `8466449.286863374` and `8466463.107594619` seconds.
The measurement marker is `8466462.424920902` seconds.
Only the second record belongs to the measured cohort.

The correction applies the measurement boundary to accepted partitions and payload records.
The corrected analyzer still rejects two measured boundaries for one seed.
The regression test first reproduced the rejected startup record.
Both startup-filter and genuine-duplicate tests then passed.
This CPU-only correction does not change the completed GPU execution.

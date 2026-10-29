# Evidence-I/O defect and timing correction

Status: CPU correction passed. GPU requalification and canonical retiming remain in progress.
No architecture verdict follows from the old timings.

## Observed defect

The original evidence recorder calls `flush` and `fsync` for every appended record.
These operations run on the caller's thread.
The request harness uses this recorder when a request completes.
The split and frontier connectors use the same recorder during serving.
Thus the original timing path contains synchronous evidence I/O.

| Run | Recorder location | Calls | Cumulative wall time | Evidence |
|---|---|---:|---:|---|
| `ouro_B_P2048_C8_s1` | P-side append | 768 | 7,167.440053 ms | MEASURED CPU counter delta |
| `dory_B_P2048_C8_s0` | Coordinator request append | 24 | 268.462029 ms | MEASURED CPU counter delta |

The source counters appear in `before_measured.json` and `after_measured.json`.
Coordinator counters appear in `coordinator_IO_before.json` and `coordinator_IO_after.json`.
These durations are not GPU communication times.
They are not mutually exclusive request phases.
Do not subtract them from E2E to construct a corrected latency.

Synchronous request logging can delay observation of another request's next token.
This delay can change strict request-level ITL attainment.
Therefore, the defect can affect positive and negative results.
The earlier Dory strict-SLO signal requires retiming too.

## Evidence classes

All original raw records, failures, source ZIPs, and logs remain unchanged.
Old timing runs have status `TIMING_INVALID_FOR_CLEAN_METHOD_COMPARISON`.
Their generated tokens and exact state checks remain usable as correctness evidence.
Their recorded wall times remain observations of the instrumented implementation.
They do not establish a clean method speedup or regression.

`archives/legacy_sync_derived_snapshot.tar.gz` preserves the derived tables before this correction.
The current analysis marks legacy timing rows as `MEASURED_INSTRUMENTED_EVIDENCE_IO_NOT_CLEAN_SERVING`.

## Correction

`telemetry.py` installs a process-local evidence writer before worker imports.
The producer serializes a record and places it in an unbounded queue.
A background thread batches durable writes.
Each file retains its record order.
Serialization occurs before enqueue, so later object mutation cannot alter a queued record.

The recorder drains the queue before a terminal status.
It also drains at normal interpreter exit and multiprocessing finalization.
Each writer saves its attempt ID, record counts, timing counters, and failure state.
`evidence_protocol.py` rejects incomplete or mismatched writer evidence.

One-time writer initialization saves a writer proof.
Warmup precedes timed serving in this campaign.
Producer timing counters exclude this initialization.
Background I/O can still contend for CPU or storage.
Nonblocking recording does not mean zero instrumentation cost.

The correction does not change weights, sampling, RNG, attention, KV, model depth, or scheduler policy.
The private RLT patch remains `4904a311358c5cf6467678d0e1c30af38b5a96064cae4c96ed680943d7e694c7`.
`configs/evidence_protocol_epoch.json` locks the new recorder's source hash.

## Verification

| Test | Result | Scope |
|---|---|---|
| Actual `worker.append` seam before the correction | FAIL, 1 test | Foreground `fsync` reproduced |
| Same seam after the correction | PASS, 1 test | No request-thread `fsync`, durable terminal record |
| Multiprocessing spawn durability | PASS, 400 records | 200 parent records and 200 child records, preserved order |
| Analysis and control regression suite | PASS, 21 tests | CPU only |
| GPU output and state requalification | PENDING | Separate 100-prompt natural comparisons and exact state imports |

The RED and GREEN logs remain under `logs/evidence_protocol_RED.log` and `logs/evidence_protocol_GREEN.log`.
`logs/telemetry_spawn_smoke.log` records the spawn check.

The first two completed new-protocol DP jobs passed durable writer checks.
They each saved 25 records, including warmup.
Their cumulative enqueue times were 9.453305 ms and 9.660492 ms.
These are process instrumentation counters, not measured E2E improvements.
The jobs are `dory_DP_P128_C1_s2` and `dory_DP_Mixed_P128_C1_s2`.

## Canonical replay and C calibration

Existing inference processes retain their original recorder.
No healthy worker receives a restart or signal for this correction.
Future workers load the new hook in the private environment.
The controller continues its manifest.

Before C, the campaign runs new-protocol output and state qualification.
It first repeats A's 42-job cut sweep with the new recorder.
It preserves the original sweep and original selected cuts.
`configs/clean_chosen_cuts.json` records the new independently qualified choices.
Canonical A confirmations use these choices, not recorder-confounded tuning results.
`frozen_runs.py` then replaces completed legacy timing cases with new independent starts.
`configs/frozen_serving_replacements.json` maps each logical case to its actual replacement.
The old case remains available.
Replacement timings require the locked runtime, recorder hash, and durable writer checks.
Known failed serving cases also receive preserved-name-independent retries in this replay.

A separate 27-job isolated calibration refreshes C's shared service-cost coefficients.
All C policies receive the same coefficients.
Legacy calibration remains available if a new calibration job fails.
The resulting coefficient file states whether nonblocking calibration completed.
The pending-work score is not a validated completion-time predictor.

## Acceptance

Accepted comparisons require the new recorder protocol, exact output checks, and the frozen model runtime.
Three independent starts remain necessary for any gain claim.
Heavy correctness hashing and GPU profiling remain separate from clean serving.
The old positive Dory signal does not bypass generic transport controls or reference-fidelity limits.
No GO or NO-GO performance decision is accepted yet.

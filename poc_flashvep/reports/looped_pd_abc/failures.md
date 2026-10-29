# Failures, corrections, and unresolved limits

The experiment preserves original logs, attempt archives, seeds, and outputs.
`experiment_inventory.csv` records run status.
A failed run does not contribute a completed latency measurement.

## Reference environment failures

The installed Transformers 5.16 API does not execute both checkpoint-owned reference implementations unchanged.
An initial Transformers 4.53 environment also failed compatibility checks.
The experiment retained those failures and created a separate Transformers 4.57.6 reference environment.
That environment completed 100 official HF generations for Ouro and Huginn.
It did not replace the main RLT environment.

## Huginn protocol mismatch

The original upstream EOS configuration omitted ID 65508.
The initial 100-prompt pilot continued beyond that token in 60 requests.
The original checkpoint also consumes a normal RNG draw before truncated-normal initialization.
Upstream RLT omitted that first draw.
The explicit reference-compatibility flag corrects both differences and preserves request RNG across PD.
The earlier outputs remain separate evidence, not corrected benchmark results.

## Boundary-model state failures

Upstream rejects Huginn PD and staged boundary prefill.
The prototype needed explicit prelude/coda KV handling and fixed-input carrier transfer.
Early attempts did not provide all required staged coda KV.
The corrected path fills the coda prefix before handoff and keeps the final prompt token on D.
Later byte-identity checks and bounded generation qualify that corrected internal path.
They do not establish official-reference accuracy.

## Dory fidelity remains unresolved

Dory has no upstream RLT model adapter at the pinned revision.
The prototype uses an isolated adapter and the official BF16 checkpoint.
Adapter v1 matched 86/100 official-reference output sequences.
Adapter v2 matched 80/100.
Packing projections and using native RoPE did not solve the discrepancy.

The four-prompt coda diagnostic matched full and split coda outputs on 4/4 cases.
Both paths matched the original reference on 3/4 cases.
This diagnostic does not explain every mismatch.
No official-reference lossless serving claim follows from it.

## Evidence-statistics import failure

A logging-cost hook initially omitted `import os` from `StageEngine.stats`.
It caused `huginn_A_P128_C1_s0` and `huginn_B_P128_C1_s0` to fail before timed measurement.
The same hook also failed `huginn_DP_P512_C4_s0`.
Their logs contain the original `NameError` and the subsequent stage-client failure.
The experiment corrected the import and added a CPU regression test.
The main campaign continued on the other independent configurations.

`repair_known_failure.py` retries only this identified instrumentation failure before the C phase.
It archives the original status and configuration first.
It preserves append-only logs and source ZIPs.
Retry completion remains pending until the retry status says `COMPLETE`.
This failure is not an A/B model-correctness result or a measured performance result.

The failed DP session retained a child GPU context during interpreter shutdown.
Its close path raised another error while it tried to collect missing final statistics.
The failed status preceded cleanup by 1491.796472 seconds, approximately 24.86 minutes.
This delay is not productive GPU computation or request latency.
The experiment signaled only the verified owned failed process group, 2587445.
It preserved the healthy controller.
`archives/failed_worker_cleanup_2587445.json` records that cleanup.
The independent guard now checks failed statuses and validated owned sessions.
It does not signal healthy jobs, burners, or unrelated processes.

## Extended CPU API regression

The first extended suite failed two upstream statistics-schema tests.
The telemetry hook added keys to an API that expected exactly two keys.
The corrected hook enables extra fields only when `ABC_RUN_DIR` is set.
Uninstrumented callers retain the original `requests` and `used_blocks` schema.
The corrected extended suite passed 97 tests and deselected 86 GPU cases.
The failed and corrected logs remain separate.
This correction does not establish model or GPU correctness by itself.

## Scope limits

GSM8K test and HumanEval accuracy are `NOT_MEASURED`.
The bounded pilot is not an answer-quality benchmark.
The current engine has fixed P/D roles. A ready-made hybrid-role PD coordinator was not found in the pinned source.
C2 migration and dynamic role switching are not implemented.
The 8GPU P6:D2 and P7:D1 ratios are not in the initial scaling matrix.
The experiment does not call them empirically dominated.

Some telemetry fields changed after the first exploratory serving trials started.
The model operators and prescribed recurrence remained unchanged.
The attempt source archives identify those telemetry versions.
Any positive candidate requires a frozen-build confirmation before acceptance.

The original short profile does not contain every matched G1 transfer control.
The report must not attribute profiler differences across those controls solely to B.
No unvalidated simulator, transfer-only estimate, or component-time sum counts as measured E2E speedup.

Quiet-lane isolation excludes other GPU workers and owned burners.
The controller, monitors, and lightweight CPU analysis still run.
The experiment does not claim exclusive CPU or storage access.
The parallel-pair calibration does not separate CPU, storage, and GPU contention causes.
Any positive final confirmation must avoid concurrent CPU tests and heavy analysis too.

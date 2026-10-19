# Collective-pair teardown retry: CPU review

## Result

No source blocker remains for the single bounded retry.
Native teardown completion remains untested.
Admission still requires outer-guard success and every rank's error-free local cleanup status.

The [structured review](../artifacts/hybrid_collective_pair_teardown_retry_review_v1.json) records tested hashes and fault cases.

## Preserved failure and unchanged work

The frozen v1 source retains SHA256 `36143ef03a69efe79ca470c54b943f4e9103e30d1632005bf04444e1d7e31b67`.
The first attempt remains NOT_ADMITTED.
A successful retry cannot change that status or establish its stall's cause.

The complete per-C loop matches frozen v1 at the AST level.
This comparison includes payloads, pair bodies, warmups, captures, correctness checks, trial order, and timing.
It preserves 31 trials and 16 pairs per graph.
AST identity does not establish identical measured timing.

## Diagnostic-I/O blocker and remedy

The first retry patch had a lifecycle blocker.
A trace-open, watchdog-setup, or progress-write exception escaped before cleanup.
An extracted-finally CPU test reproduced all three cases.
Each case skipped every NCCL, WORLD, and custom-AR cleanup call.

Main patched those paths.
Diagnostic setup and progress-write errors now enter the error list.
They no longer skip the tested cleanup operations.
Watchdog cancellation and trace-file close enter guaranteed finalization.

The initial retry hash was `0c617785debad9bb1a06c63ebcbf18d61e29349b2cde799c9edf5854c040df7d`.
The tested amended hash is `d2746828ac401dc0497b752260b93da07ed5f21ab68ece2cbb556b4c2ebae3ba`.

## Lifecycle tests and limits

Four unit tests pass.
Ten fake lifecycle cases execute the actual finally AST with injected fake objects.
They execute no real Torch or CUDA call.

The normal case follows NCCL subgroup → Gloo WORLD → CustomAllreduce close.
The fault cases preserve all three attempts.
They cover diagnostic setup/write, each teardown operation, watchdog cancellation, trace close, and final status writing.

The installed Python close path passes rank explicitly to buffer release.
It performs no process-group query.
Thus, its Python interface does not require a live Gloo group.
This supports the retry order, not a proven native fix.

AFTER markers mean that an attempt returned or entered its exception handler.
They do not certify success.
The local error list and outer guard remain necessary.

Cleanup timestamps use host UTC.
They are diagnostics, not communication durations or synchronized rank clocks.

The final status write can still fail after all cleanup attempts.
The fake case raises an I/O error and fails closed.
Do not admit a run with a missing local status.

These tests establish local control flow only.
They do not establish native progress or recovery from fatal CUDA errors.
The [retry contract](hybrid_collective_pair_teardown_retry_contract_v1.md) permits no workload change or further investigation after another blocked retry.

This reviewer changed no probe code, process, package, GPU state, or raw artifact.
No actual outcome or protected result entered this review.

## Actual work

The actual interval is 2026-10-19 18:31:53 UTC through 2026-10-19 18:34:29 UTC.
The active interval through this checkpoint is 156 seconds.
No prior pause interval enters this work.


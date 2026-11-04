# Rejected grouped-Core diagnostic

## Actual failure

Run `m_huginn_loop_partial_burst_state8` failed during startup.
The worker returned code 1 after `47.15784204378724` seconds.
This duration includes startup and warm-up. It is not GPU compute latency.
No measured N8 output/state verdict exists for this candidate.

The native callback rejected the proposed multi-loop prompt submission:

```text
ValueError: wavefront prefill requires one recurrent depth per scheduling step
```

The exception comes from `Scheduler._apply_wavefront_prefill_batch_progress`.
The preceding Atomic Native N8 control passed exact output and recorded-state comparison.
That control does not exercise a grouped staged-Core submission.

## CPU replay limitation

The first CPU replay used `_apply_split_prefill_progress`.
That helper supports grouped depths and produced identical logical frontiers.
The active Wavefront path calls a different progress method.
Therefore, the first replay did not test the actual caller contract.
Its original result remains preserved with its stated CPU-only scope.

The corrected replay calls `_apply_wavefront_prefill_batch_progress` directly.
It checks proposed counts 3 for Ouro R4 and 31 for Huginn R32.
The common callback rejects both counts with the same exception.
This callback-level result does not establish any GPU arithmetic difference.

## Decision

The optional candidate is `BLOCKED_NATIVE_WAVEFRONT_PROGRESS_GUARD`.
Do not remove the assertion or label this execution successful.
Do not start candidate performance tests or N32 expansion.
The remaining optional Token and Ouro GPU pilots receive `NOT_RUN_SHARED_WAVEFRONT_GUARD`.

The original serving configurations omit `contiguous_prompt_loop_burst`.
They retain the passing one-depth Wavefront implementation.
Their Token/Loop qualification and serving measurements continue independently.
The failed diagnostic does not replace or modify that runtime.

Supporting grouped Wavefront execution needs a separate lifecycle implementation.
It must preserve chunk completion, frontier order, transfer publication, and source ownership.
The current campaign does not bypass these requirements.

## Evidence

- `raw/m_huginn_loop_partial_burst_state8/dispatcher_result.json`
- `logs/m_huginn_loop_partial_burst_state8.log`
- `state/native_burst_cpu_replay.json`
- `state/native_burst_wavefront_guard_replay.json`
- `state/burst_pilot.json`

No measured speedup exists for this rejected candidate.

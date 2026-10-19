# C07 sealed source cohort: pre-run amendment

## Retained failure and limit

The first guarded P1/GPU5 probe did not reach parity.
Its [capture record](/home/esjung/MLLM-EP-ouro-deep-discovery/poc_ouro_parallelism_deep_discovery/outputs/c07_base14_tp1_row2_depth2_gpu5_fixture_v1/1.4B_C07_TP1/capture_finish.json) reports four observed forwards.
Every forward received `NOT_FOUR_UNPADDED_V2_ROWS`.
The record contains no CPU batch counts.
It therefore does not establish async admission, actual padding, or mixed request progress as the cause.
The prior output remains unchanged.

This amendment adds CPU rejection evidence and changes only untimed reference construction.
The parent permits one further P1 smoke before reassessment.
This amendment authorizes no ragged scheduler, new serving adapter, or automatic retry loop.
No GPU command or GPU execution occurred during this change.

## Rejected CPU evidence

[rejected_cpu_batch](../c07_fixture.py:208) retains an owned copy of each rejected CPU admission snapshot.
The snapshot includes native request IDs and request-state indices.
It includes real and padded row/token counts, scheduled counts, query boundaries, and computed/prefill arrays.
The record uses no device-value read.
Later batch updates cannot change its retained counts.
The capture-finish record and failed export retain these records.

The regression executed the actual rejected-forward body.
It first failed because `cpu_batch` was absent.
It now passes and checks ownership after the original CPU dictionary changes.

## Native pause, queue, and resume semantics

The helper requires DP1, PP1, four prompts, no unfinished request, and an initially unpaused scheduler.
It refuses to resume a pause that it did not acquire.
It uses the existing synchronous [call_utility](/home/esjung/.venvs/cp-composition-vllm026/lib/python3.14/site-packages/vllm/v1/engine/core_client.py:875) interface.
That method sends a utility message and waits for its result Future.
The core [utility handler](/home/esjung/.venvs/cp-composition-vllm026/lib/python3.14/site-packages/vllm/v1/engine/core.py:1534) waits for returned Futures before sending the result.

The exact construction order is:

1. Call `pause_scheduler('keep', False)` and wait for its acknowledgement.
2. Check the native paused state.
3. Call native `LLM.enqueue` with all four frozen requests and the unchanged sampling parameters.
4. Check paused state through another synchronous same-client utility call.
5. Call native `resume_scheduler` in `finally` and check the resumed state.
6. Call native `LLM.wait_for_completion` only after resume.

The native [multiprocess pause implementation](/home/esjung/.venvs/cp-composition-vllm026/lib/python3.14/site-packages/vllm/v1/engine/core.py:1764) sets `PAUSED_ALL` for `keep`.
It does not abort requests.
Passing `False` prevents cache clearing.
If needed, its Future waits for the engine's idle condition before acknowledgement.
The native [scheduler](/home/esjung/.venvs/cp-composition-vllm026/lib/python3.14/site-packages/vllm/v1/core/sched/scheduler.py:446) sets token budget zero while fully paused.
Its [add path](/home/esjung/.venvs/cp-composition-vllm026/lib/python3.14/site-packages/vllm/v1/core/sched/scheduler.py:2120) still queues incoming requests.

[LLM.enqueue](/home/esjung/.venvs/cp-composition-vllm026/lib/python3.14/site-packages/vllm/entrypoints/llm.py:476) adds completion requests through the native input processor.
It does not provide an atomic multi-request scheduler transaction.
The pause prevents scheduling between those separate add messages.
The same-client utility acknowledgement fences the preceding adds on this single-engine connection.
The core [input loop](/home/esjung/.venvs/cp-composition-vllm026/lib/python3.14/site-packages/vllm/v1/engine/core.py:1583) processes their decoding and preprocessing in connection order.
This fence is not a returned scheduler queue-count measurement.
Native preprocessing errors can still prevent a valid cohort.

Native [resume](/home/esjung/.venvs/cp-composition-vllm026/lib/python3.14/site-packages/vllm/v1/engine/core.py:852) returns the scheduler to `UNPAUSED`.
Native [wait_for_completion](/home/esjung/.venvs/cp-composition-vllm026/lib/python3.14/site-packages/vllm/entrypoints/llm.py:536) drains results through the normal engine loop.
Waiting while paused would prevent progress.
The helper therefore resumes before that call, including on enqueue failure.

## Failure handling and unchanged gates

[sealed_reference_cohort](../c07_fixture.py:213) retains ordered control events and native request IDs.
The runner writes `source_cohort_seal.json` in `finally`.
The helper attempts resume after an attempted pause, including when pause or enqueue raises.
It retains a resume error without masking an earlier error.
A dead backend can still prevent restoration.
The external guard retains timeout and teardown responsibility.

Sealed source submission is not proof of an unpadded C4 cached-decode invocation.
The exact V2 admission checks remain unchanged.
Actual padded row/token counts still block the fixture.
The code does not relabel padded work or relax progress checks.
The bitwise output/KV checks and all-rank parity gates also remain unchanged.
No physical service interval includes pause, enqueue, resume, or reference preparation.

The frozen prompts, constant allowed token, two source outputs, and complete native R4 computation remain unchanged.
The engine configuration and selected row/depth remain unchanged.
Fresh job-local caches and a new output directory remain mandatory.
The new pre-run metadata also hashes the actual V2 and native control source files.

## CPU checks and actual time

The suite passes 42 CPU tests in both interpreters.
New tests check control order, strict keep/no-clear arguments, queue fencing, restoration, and failures.
They also check prior-work and prior-pause rejection, primary-error retention, and runner ledger persistence.
Syntax checks pass.
These tests do not establish native C4 admission or GPU parity.

```text
f791f0fed369bfa2df53437066bfcc4a842b5ada9cc897b2e389afea3238912e  c07_fixture.py
c8c40dac14c8e919b7a5522722dfc5b9ac2a9e44da2efd119cd197b23502025c  c07_fixture_probe.py
23da37e20d184aabbe06c9dbefb3dbf5280c4942458feaa5240be268b443d03b  scripts/c07_one_pass_fixture.py
e8998462beebddb3f85600e88831bf5c1eb012f8977277637ad7fd30b2637cf9  tests/test_c07_fixture_cpu.py
```

The source and CPU segment ran from `2026-10-19 15:42:28 UTC` to `2026-10-19 15:49:04 UTC`.
That interval is 396 seconds.
Contract drafting and lint follow that interval.

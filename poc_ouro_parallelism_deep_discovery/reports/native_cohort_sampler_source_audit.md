# Native cohort sampler and admission source audit

## Decision and scope

The frozen singleton-token plan has no blocking issue in the inspected sampler and admission source.
This finding supports a controlled hardware workload.
It does not establish runtime correctness, task quality, throughput, or simultaneous admission.
This audit ran no GPU work.
It did not change the live runner, observer, installed packages, or raw evidence.

The audit covers [protocol v2](../artifacts/native_cohort_protocol_v2.json), [the runner](../scripts/native_cohort_control.py), and [the observer](../cohort_probe.py).
The source and time ledger is [native_cohort_sampler_source_audit.json](../artifacts/native_cohort_sampler_source_audit.json).
The preserved v1 protocol proposed 32 GiB per rank.
The parent reports no v1 execution.
This audit did not inspect live GPU results.

## Native output constraint

The runner uses `SamplingParams(temperature=0, ignore_eos=True, allowed_token_ids=[token])` at lines 66–82.
It checks every returned request against exactly `length` copies of the frozen token.
It does not install a custom teacher hook.
The runner checks `token != 0` at line 35.
Both pinned checkpoint configurations set `eos_token_id=0` at line 12.

The installed sampler first converts logits to FP32.
It then applies logits processors and native greedy selection.
The allowed-token mask replaces every forbidden logit with negative infinity.
A finite allowed logit therefore selects the singleton token when no later processor conflicts.
See `v1/sample/sampler.py:95–102`, `:240–271`, and `:390–403` under the installed source root below.

The mask precedes bad-word exclusion, non-argmax-invariant processors, penalties, and thinking-budget logic.
The frozen runner supplies no conflicting stop strings, bad words, custom processors, or non-neutral penalties.
Native defaults specify `n=1`, neutral penalties, no stop strings, and no stop-token list.
See `sampling_params.py:213–270`.
Native allowed-token validation checks an empty list and token range at `sampling_params.py:844–868`.
The runner's returned-token assertion remains necessary.
It checks the actual result rather than an assumption about finite logits.

This path does not bypass the model, four recurrent passes, or LM head.
The model forward still executes its original fixed loop at `model_executor/models/ouro.py:367–374`.
Native logits computation remains at `v1/worker/gpu_model_runner.py:4428–4429`.
This audit makes no adaptive-gate claim.

## Request ownership and asynchronous sampling

`InputBatch.add_request` creates one Boolean mask row for each request.
It masks the complete vocabulary except the allowed token.
Mask state follows row swaps and row compaction.
Removal clears the old row.
See `v1/worker/gpu_input_batch.py:429–450`, `:561–564`, `:672–679`, and `:797–800`.
The singleton constraint therefore does not depend on a global sampler-call counter.
Different request progress does not change the intended constant token.

The singleton mask alone does not require output-token history.
The inspected history condition includes penalties, bad words, history-dependent processors, and thinking-budget state.
It excludes `allowed_token_ids`.
See `gpu_input_batch.py:880–896`.
With the frozen neutral parameters, this constraint does not enable output-history synchronization at `gpu_input_batch.py:1045`.
This finding does not claim that the entire runtime has no synchronization.

The native asynchronous path retains sampled GPU tokens and a request-to-row map.
It can keep CPU output-token placeholders before results return.
See `gpu_model_runner.py:3742–3791` and `v1/core/sched/async_scheduler.py:19–74`.
A global call counter cannot infer each request's output ordinal from these paths.
The new constant-token control avoids that requirement.

## Admission and trace limits

`C` correctly means submitted cohort size.
It does not mean a fixed batch size.
The frontend adds requests separately at `entrypoints/offline_utils.py:523–570`.
The multiprocessing client sends each ADD immediately at `v1/engine/core_client.py:886–889`.
The engine drains currently available inputs, then executes a step at `v1/engine/core.py:1358–1403`.
The frontend explicitly permits execution before the next prompt completes at `offline_utils.py:504–511`.

The 32768-token budget can hold eight 4096-token prompts.
That capacity is not an admission barrier.
The scheduler considers running requests before waiting requests.
Request limits, available KV, and preemption also affect admission.
See `v1/core/sched/scheduler.py:469–509`, `:665–674`, and `:1212–1234`.
The asynchronous queue can schedule work before an older result reaches the scheduler.
See `v1/engine/core.py:617–679`.

The observer records the scheduled request IDs and token count for every observed forward.
It records native sampler tensor shape and element count.
The wrappers preserve original arguments, return values, and exceptions.
They read metadata only.
They call no tensor read, tensor copy, or synchronization operation.
See `cohort_probe.py:15–35`.
The separate trace lane begins after all headline repetitions at `native_cohort_control.py:104–111`.

The installed `Sampler.sample` returns a one-dimensional token tensor for ordinary greedy sampling.
`Sampler.forward` later expands it to `[num_requests, 1]`.
See `sampler.py:240–271` and `:138–147`.
Thus this observer can record `[C]`, not necessarily `[C, 1]`.
Its row count is the first dimension in this non-speculative path.
CUDA Graph token padding does not itself add sampler rows.
The runner selects per-request hidden states before logits computation at `gpu_model_runner.py:2210–2218` and `:4428–4429`.

The current trace has these limits:

- It does not record the sampler's ordered request IDs or output ordinals.
- It does not record discard masks, preemption counts, or resumed request IDs.
- It cannot establish that all sampler rows are useful committed outputs at the same progress.
- It does not establish the exact batch sequence of each separate clean timing repetition.
- TP ranks can report the same logical batch. Do not sum their logical request or output counts.

The native runner can sample partial-prefill rows and discard their outputs.
See `gpu_model_runner.py:2077–2082`, `:2210–2218`, and `:3688–3700`.
The frozen configuration disables chunked prefill.
The observer does not independently record zero discarded rows.
Its scheduled request IDs do establish scheduled membership.
Its sampler counts establish tensor size.
Do not strengthen either observation into equal-progress or committed-output evidence.
No live observer change is necessary for the stated finite-cohort metric.

## Structural capacity and extra work

For these checkpoints, fixed-R4 BF16 KV capacity per rank is:

`2 * L * 4 * C * (N + 128) * 16 * 128 * 2 / TP` bytes.

The factors represent K/V, layers, passes, requests, positions, KV heads, head dimension, and bytes per BF16 value.
The formula assumes full unshared depth KV and head sharding at TP1/2/4.
It reserves all `N + 128` positions.
The last generated token need not enter another model call.
This reserve is not a count of actual KV writes.
The installed model creates a separate Attention cache prefix for each pass at `ouro.py:170–188`.
The checkpoints use 16 KV heads and head dimension 128.
This is structural accounting, not observed memory use.

| Checkpoint | Layers | TP1, C8, N4096 | TP2 | TP4 |
|---|---:|---:|---:|---:|
| Base1.4B | 24 | 24.75 GiB/rank | 12.375 GiB/rank | 6.1875 GiB/rank |
| Base2.6B | 48 | 49.5 GiB/rank | 24.75 GiB/rank | 12.375 GiB/rank |

The v2 KV pool is 52 GiB per rank, or 55,834,574,848 bytes.
That pool exceeds the stated worst-case structural requirement.
It does not prove successful allocation beside weights, graphs, workspaces, or other runtime allocations.
It also does not prove zero preemption in a particular run.
Inspect runtime evidence before such claims.

The mask adds a full-vocabulary `masked_fill_` before the native greedy reduction.
The native mask stores at most `8 * 49152 = 393216` Boolean bytes per CPU or GPU mask.
That value is 0.375 MiB per mask.
Metadata refresh copies active mask rows to the GPU at `gpu_input_batch.py:898–906`.
Refresh occurs when batch state changes at `gpu_input_batch.py:814–832`.
This source does not imply a CPU mask copy on every stable decode iteration.
No mask latency or traffic measurement forms part of this audit.
Keep this output policy identical across matched control arms.

## Assertions and reporting boundary

The runner already checks output request count, constant token identity, GPU UUIDs, and four depth-cache slots per layer.
It also checks the inherited asynchronous, fixed-R4, no-chunking, and no-prefix-cache settings.
See `native_cohort_control.py:49–76`.
The inherited configuration uses the native default PP1 and no speculative configuration.
The corresponding defaults are `config/parallel.py:120` and `config/vllm.py:338`.
A future configuration change must preserve those conditions or repeat this source audit.

Use the observed request count and 128 returned tokens per request as the completed-work denominator.
Report the entire `generate` wall time as submitted finite-cohort API time.
Report actual scheduled membership separately from submitted `C`.
Report task quality and adaptive behavior as `NOT_RUN`.
Do not compare this constant-token workload directly with the earlier changing-teacher workload as a system speedup.
The repeated calibration prompts are not independent request populations.

## CPU evidence and actual time

The existing seven CPU observer tests passed during this audit.
The command was `python -m unittest poc_ouro_parallelism_deep_discovery.tests.test_native_cohort_cpu -v`.
The test runner reported `Ran 7 tests in 0.005s` and `OK`.
These tests use stubs and source inspection.
They do not execute native vLLM sampling or CUDA Graphs.

The source-audit segment began at 2026-10-19 13:56:35 UTC.
The ledger records the actual final timestamp and elapsed time.
Installed-source paths use this root:

`/home/esjung/.venvs/cp-composition-vllm026/lib/python3.14/site-packages/vllm/`

The ledger retains exact source hashes and function locators.

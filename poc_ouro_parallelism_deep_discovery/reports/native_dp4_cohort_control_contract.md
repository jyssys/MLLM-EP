# Native fixed-R4 DP4 cohort control

Status: CPU implementation complete. GPU execution and timing remain `NOT_RUN`.

## Scope and frozen configuration

This control uses four independent native TP1 replicas.
It is a generic DP4 baseline, not a candidate engine.
It does not implement adaptive CDB, request migration, or the factorial study.
The existing native TP cohort runner remains unchanged.

The [DP4 protocol](../artifacts/native_dp4_cohort_protocol_v1.json) pins the [native v2 protocol](../artifacts/native_cohort_protocol_v2.json) by SHA256.
It pins the same Base1.4B and Base2.6B revisions.
The runtime retains these settings:

- Fixed R4 and BF16.
- C=1/2/4/8 and N=1024/4096.
- 128 output tokens per request.
- Native asynchronous scheduling and CUDA Graphs.
- No chunked prefill, prefix caching, expert parallelism, or speculative decoding.
- One frozen non-EOS allowed token through the native greedy sampler.
- A 52 GiB KV pool per replica, max_num_seqs=8, and max_num_batched_tokens=32768.
- Three 16-output warm-up runs per case and five complete measured repetitions.

The parent freezes `dp4_plan.json` before it starts any replica.
That file retains the cases, full engine configurations, source hashes, protocol hashes, checkpoint revision, and token ID.
Each replica retains its own engine configuration, cache environment, ownership evidence, console, and final status.
No replica changes a model operator or installs the earlier teacher hook.

## Resource ownership and placement

One outer GPU guard owns physical GPUs 4/5/6/7.
Replica 0 uses GPU4, replica 1 uses GPU5, replica 2 uses GPU6, and replica 3 uses GPU7.
Each replica has a singleton `CUDA_VISIBLE_DEVICES` mask.
Each native engine uses TP1, PP1, and native DP1.
The parent coordinates the four independent engines.
It does not create a native cross-replica process group.

The parent rejects an absent guard contract, a different mask, or a different process-group owner.
Children retain the guard's process group and `LOOPED_KV_OUTPUT` identity.
The outer guard can therefore reclaim the complete owned session after failure.
Children use separate output directories without changing that guard identity.
Each replica has separate VLLM, AOT, Inductor, Triton, extension, and compilation cache roots.
The VLLM cache root owns the native AOT cache location.
The runtime records those roots before it imports vLLM.

Every replica must report its expected GPU UUID and exactly `4 * L` cache slots.
All four engines remain loaded across warm-up, clean repetitions, and the separate trace lane.
Idle replicas retain their model and KV pool.
They still count toward the four-GPU resource budget.

The fixed placement is `cohort_slot % 4`.

| Submitted C | Replica 0 | Replica 1 | Replica 2 | Replica 3 |
|---:|---|---|---|---|
| 1 | 0 | idle | idle | idle |
| 2 | 0 | 1 | idle | idle |
| 4 | 0 | 1 | 2 | 3 |
| 8 | 0, 4 | 1, 5 | 2, 6 | 3, 7 |

These entries are cohort slots, not assumed native batch rows.
The frozen prompt populations repeat as in the native TP cohort control.
They do not become independent populations at C8.
The native scheduler still controls admission inside each replica.

## Barrier and message contract

The parent waits for `LOADED` from all four replicas before it prepares any trial.
Each `LOADED` response follows engine initialization and ownership assertions.
Every warm-up, clean trial, and trace trial uses this sequence:

1. The parent sends `PREPARE(trial_id, case_index, length, lane)` to all four replicas.
2. Each replica prepares only its static requests and native sampling parameters.
3. Each replica sends `READY(trial_id, assigned_slots)`, including replicas with no request.
4. The parent checks all four identities, trial IDs, and slot assignments.
5. The parent starts its timer immediately before the first `RELEASE(trial_id)` send.
6. Each replica executes its native `generate` call only after the matching release.
7. Each replica sends `DONE(trial_id, result)` after output assertions and result serialization.
8. The parent stops its timer after it receives all four matching completions.

A worker rejects a repeated trial or a mismatched release.
A worker with no request does not call `generate`.
It still sends readiness and completion responses.
The parent never derives global timing from a maximum, sum, or scaled TP1 measurement.

## Measured interval and denominators

The primary metric uses the observed parent release-to-all-completed wall time.
The timer uses one parent `perf_counter_ns` clock.
It includes sequential release messages, release skew, native frontend work, completion messages, and parent control work.
It also includes output assertions and result serialization before completion delivery.
It excludes engine initialization, prompt preparation, and the readiness wait.
It excludes result-file writes after all completions.

Output TPS is `C * 128 / global_makespan_seconds`.
The parent checks all four worker denominators and the identity of every cohort slot.
It retains per-worker `generate` wall time separately.
It retains native per-request metrics separately.
Neither quantity replaces the parent makespan.

This interval differs from the single-API boundary in the native TP cohort control.
Any comparison must state both boundaries.
DP4 includes its own bounded IPC and control overhead.
It does not include request preparation before the release barrier.
Neither result is steady-state serving throughput or natural task accuracy.
Do not divide TP1 time by four to predict this control.
Do not compare constant-token and changing-teacher workloads as a system speedup.

## Clean timing and separate trace

The clean lane uses no schedule wrapper.
It completes all randomized clean repetitions before it starts trace trials.
The trace lane uses the existing `CohortProbe` metadata observer.
Each replica starts its observer before `READY` and finishes it after its native `generate` call.
The parent labels every trace interval `EXCLUDED_FROM_HEADLINE`.
Idle replicas produce an empty schedule trace.

The observer records scheduled request IDs, scheduled token counts, and sampler tensor sizes.
It does not record ordered sampler request IDs, discard masks, or output ordinals.
The [sampler source audit](native_cohort_sampler_source_audit.md) states those limits.
Do not equate submitted C with a constant native batch size.
Do not assume the separate trace exactly reproduces every clean repetition's admission order.

## Failure and teardown

The startup deadline is 1200 seconds for the complete replica set.
Each barrier or completion wait has one shared 300-second deadline.
The shutdown deadline is 60 seconds.
Unexpected identities, stale messages, channel closure, output mismatch, or worker errors fail the run.
The parent retains the failure traceback, control transcript, available worker files, and failed status.
It does not publish a successful final status after forced child cleanup.

Normal teardown sends `STOP` to every replica.
Each replica shuts down its native engine before it sends `STOPPED`.
The parent then waits for child-process exit.
Failure cleanup attempts the same stop and escalates only for explicitly created child PIDs.
The outer guard retains final responsibility for the owned session and GPU-context cleanup.
No cleanup path signals an unrelated process group.

## CPU evidence and run command

The [CPU tests](../tests/test_native_dp4_cohort_cpu.py) cover placement, barriers, stale messages, metrics, configuration freeze, and child cleanup.
They also exercise the inherited control FD and singleton mask in a real CPU subprocess.
They do not load a native engine or execute CUDA.
All 14 tests passed with the default interpreter and the dedicated campaign interpreter.
The inspected TP1/2/4 workload files use frozen token 378 for both model sizes.
The [CPU evidence ledger](../artifacts/native_dp4_cohort_cpu_validation.json) records test counts, hashes, and actual segment timestamps.

The main agent must review this implementation before GPU execution.
The following command is ready for that later review and run:

```bash
/home/esjung/.venvs/llada2-mini-reference/bin/python -m poc_ouro_parallelism_deep_discovery.scripts.gpu_session \
  --gpus 4,5,6,7 --output outputs/native_dp4_cohort_base14_v1 --timeout 5400 -- \
  /home/esjung/.venvs/ouro-deep-discovery/bin/python \
  -m poc_ouro_parallelism_deep_discovery.scripts.native_dp4_cohort_control --size 1.4B
```

For Base2.6B, use `--size 2.6B` and a new output directory such as `outputs/native_dp4_cohort_base26_v1`.
Run from `/home/esjung/MLLM-EP-ouro-deep-discovery`.
The implementer did not execute either GPU command.

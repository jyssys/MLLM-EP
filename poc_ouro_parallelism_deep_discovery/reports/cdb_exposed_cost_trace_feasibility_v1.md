# Original-gate CDB exposed-cost trace contract v1

Status: outer observation is source-feasible. Several finer separations remain `UNRESOLVED`. This is a pre-run contract, not measured hardware evidence. No observer implementation, runtime edit, or GPU run occurred.

The [frozen contract](../artifacts/cdb_exposed_cost_trace_contract_v1.json) specifies both Base sizes and both policies. The CDB revision remains `abdfae0adbdd12be8c6123ccb34eee98a38afb39`. This work concerns baseline attribution. It is not a new architecture or a C07 continuation.

## The existing baseline already uses graphs

[quality_pilot.py](../scripts/quality_pilot.py:23) calls `load_engine(...,graphs=True,batch=1,blocks=640,max_model_len=10240)`. [load_engine](../runtime.py:34) preserves BF16, the original checkpoint gate, `last_exited`, async OFF, delayed consumption OFF, and synthetic replay OFF. Decode stage graphs remain ON. Prefill graphs remain OFF.

Existing `gsm8k/test/1074` pilot records contain these recurrent graph-hit counts:

| Cell | Graph hits | Graph captures |
|---|---:|---:|
| 1.4B fixed | 336 | 0 |
| 1.4B adaptive | 249 | 1 |
| 2.6B fixed | 284 | 0 |
| 2.6B adaptive | 225 | 0 |

These are existing quality-pilot records, not new timing measurements. Their paths are `outputs/quality_base14_{fixed,adaptive}_v1/1.4B_{fixed,adaptive}/gsm8k_test_1074.json` and the corresponding `base26` paths. They establish graph use. They do not establish a capture-free interval for every old request. A new graph-ON versus graph-OFF control is not proposed.

## Safe outer observation points

All instance wrappers must call the original bound method exactly once. They must return its original result unchanged. They must preserve exceptions and restore exact instance attributes in `finally`. Do not patch a global class or Torch function.

| Observation | Exact callable and location | Clock and actual scope |
|---|---|---|
| Recurrent stage | [runner._run_recurrent_stage](/home/esjung/external/decode-first-cdb-upstream/src/looped_cdb/continuous_depth_batching/model_runner.py:896) | One compute-stream event pair around one original graph replay. |
| Containing batch | [runner.compute_recurrent_batch](/home/esjung/external/decode-first-cdb-upstream/src/looped_cdb/continuous_depth_batching/model_runner.py:646) | Host wall for metadata reuse waits, staging, H2D, hidden-bank work, graph launch, and native synchronization. |
| CPU metadata preparation | [runner._prepare_recurrent_stage_buffers](/home/esjung/external/decode-first-cdb-upstream/src/looped_cdb/continuous_depth_batching/model_runner.py:704) | Nested host wall for lists and pinned metadata tensors. Not H2D-only time. |
| Original exit decision | [engine._should_exit](/home/esjung/external/decode-first-cdb-upstream/src/looped_cdb/continuous_depth_batching/continuous_api.py:1656) | Host wall including dispatch and the unchanged policy call when reached. |
| Terminal K/V publication | [engine._flush_exit_kv_copies](/home/esjung/external/decode-first-cdb-upstream/src/looped_cdb/continuous_depth_batching/continuous_api.py:1210) | Host wall for every call. One compute-stream event pair for a nonempty backlog only. |
| Combined coda device window | [runner._run_coda_forward](/home/esjung/external/decode-first-cdb-upstream/src/looped_cdb/continuous_depth_batching/model_runner.py:1036) | Compute-stream events around staging/gather and the LM-head-plus-argmax graph. |
| Containing coda stage | [runner.stage_coda_batch](/home/esjung/external/decode-first-cdb-upstream/src/looped_cdb/continuous_depth_batching/model_runner.py:973) | Host wall including native synchronous completion and token readback on this path. |
| Coda consumption | [runner.consume_coda_result](/home/esjung/external/decode-first-cdb-upstream/src/looped_cdb/continuous_depth_batching/model_runner.py:1154) | Host token conversion and hidden-slot release. |

The recurrent event pair includes all physical layers, recurrent normalization, the original gate head, and native output-buffer copies. See [_forward_recurrent_stage](/home/esjung/external/decode-first-cdb-upstream/src/looped_cdb/continuous_depth_batching/model_runner.py:941). It excludes the surrounding hidden-bank gather and scatter. It is not a gate-free transformer-only kernel measurement. Event spans can include GPU idle time between host enqueue operations.

The CUDA graph calls the original [Ouro recurrent adapter](/home/esjung/external/decode-first-cdb-upstream/src/looped_cdb/continuous_depth_batching/adapters/ouro.py:248). Its layer order, norm, gate, and precision remain unchanged. The observer does not enter that adapter or `_forward_recurrent_stage`.

The [containing batch](../artifacts/cdb_exposed_cost_trace_contract_v1.json) already synchronizes the compute stream before same-step routing. The host exit policy then executes `torch.sigmoid(signal).float().item()` and updates the CPU CDF. See [ExitPolicy.should_exit](/home/esjung/external/decode-first-cdb-upstream/src/looped_cdb/continuous_depth_batching/exit_policy.py:150). Do not move the sigmoid, change its dtype order, or relocate its caller stream.

`ExitPolicy` is a frozen, slotted dataclass. Its method is not a safe direct instance-patch target. The bounded measurement therefore includes `engine._should_exit` dispatch. It must label forced final-step shortcuts separately. Pure policy-method wall remains unresolved. Fixed R4 also retains gate-head work and nonfinal policy updates in this baseline. This contract does not remove them.

## Publication and coda limits

`_record_exit` only queues early-exit work. It does not finish publication. [_flush_exit_kv_copies](/home/esjung/external/decode-first-cdb-upstream/src/looped_cdb/continuous_depth_batching/continuous_api.py:1210) drains that queue before the next recurrent read. It explicitly uses the existing compute stream. Its [cache operations](/home/esjung/external/decode-first-cdb-upstream/src/looped_cdb/continuous_depth_batching/cache.py:99) construct token-row indices, gather source K/V, and copy each required deeper slot.

An outer pair measures this complete publication window without changing its dependencies. It does not measure pure copy-kernel busy time. It includes row preparation and host enqueue gaps. Empty flushes receive no CUDA pair. Fixed R4 should have no early-exit publication backlog under this policy. An adaptive request can also have no early exits. Neither case permits forced exits for the observer.

Pure terminal-copy kernel time remains unresolved. A future first-copy marker could exclude initial row preparation, but that is not necessary for this contract. Per-layer events or waits are forbidden.

The original [_run_coda_forward_graphed](/home/esjung/external/decode-first-cdb-upstream/src/looped_cdb/continuous_depth_batching/model_runner.py:1064) captures the LM head and argmax together. An outer hook cannot split them without graph-body instrumentation. The native synchronous coda path also contains token readback. Isolated D2H time is not available through these outer methods. Report combined coda windows, not separate invented projection and sampling costs.

## Frozen request and clock protocol

Selection uses only calibration membership, dataset, and lexicographic ID. The first two IDs in each domain are:

- GSM8K: `gsm8k/test/106`, `gsm8k/test/1074`.
- MATH-500: `math500/test/142`, `math500/test/156`.

No answer or outcome enters this selector. Prior pilot records establish graph use, not request selection. No held-out request enters the lane.

Each size/mode cell uses a fresh engine. That engine stays warm across its four requests. Each request uses this sequence:

1. One untimed same-request warm run.
2. `CLEAN_BEFORE`.
3. `TRACE`.
4. `CLEAN_AFTER`.

Every generation has cap 128. Native EOS and original stop strings remain active. Native greedy sampling remains unchanged. The existing four-output native warmup is optional. The per-request cap replaces the earlier aggregate-token proposal.

Every measured phase must have zero recurrent and stage graph captures. Reject output-token, finish-reason, exit-depth, or NFE mismatches. Compare actual per-position depths and recurrent step counts. Compare prefill counts separately, with their original units. Retain prompt IDs, generated counts, graph hits, and coda token counts. The common clean observer is only the existing pilot-style CPU exit recorder.

Use unique `Event(enable_timing=True)` pairs on `inputs_and_outputs.compute_stream`. Record each pair explicitly on that stream. Do not change the original method's stream context. In particular, do not execute the host policy inside a new compute-stream context.

All events must exist and complete initial materialization before measurement. At cap 128, at most 508 recurrent decode passes occur. The pool reserves 512 recurrent pairs, 128 publication pairs, and 128 coda pairs. This requires at most 1536 event objects. Pool overflow invalidates the trace.

No event node enters a captured body. Every observed replay gets its own pair. Resolve elapsed times only after the request-boundary synchronization. Reuse events only after resolution and archival. Never multiply one final replay duration by a replay count.

Store CPU request, position, depth, batch ordinal, and graph-key identities before native mutation. Use only existing CPU fields and tensor metadata. No hook may read tensor contents, print, serialize, create events, or add a synchronization. Preserve every native wait. Restore wrappers before `CLEAN_AFTER`.

## Overhead, interpretation, and stop conditions

Request walls use the existing synchronize/generate/synchronize boundary. Event export occurs after that clock ends. Report each trace wall beside both clean walls. Report the trace-to-clean-mean ratio and clean-before/after drift. One triplet per request supplies no timing confidence interval.

The frozen diagnostic guard is 5%. If clean drift exceeds 5%, mark drift unresolved. If trace wall exceeds 105% of the larger clean wall, mark observer overhead unresolved. This contract predeclares these guards. They are not measured overhead claims.

Retain failed records. Do not subtract estimated overhead from spans.

Every interval remains nested or overlapping evidence. Do not add CPU and CUDA intervals. Do not label host-wall minus device-span as pure CPU overhead. Do not convert interval sums into a 100% end-to-end breakdown. No Amdahl gain or architectural speedup follows from this lane.

Stop attribution after any parity failure, measured capture, unknown stream, event overflow, hook-restoration failure, or CUDA error. Do not retry automatically or adjust semantics. This contract does not claim recovery from fatal device errors.

The contract passes metadata-selector, prompt-hash, source-hash, callable-line, and event-capacity checks. The source/CPU segment ran from `2026-10-19 16:15:16 UTC` to `2026-10-19 16:22:11 UTC`. Its elapsed time was 415 seconds. Report drafting and lint followed this segment.

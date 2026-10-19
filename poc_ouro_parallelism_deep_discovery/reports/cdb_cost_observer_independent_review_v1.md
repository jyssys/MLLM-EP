# CDB cost observer independent review v1

Outcome: the review found no remaining source blocker for the frozen B1 lane. This outcome applies to the tested hashes below. It is not GPU correctness, numerical parity, or overhead evidence. The reviewer executed no GPU command.

The review compared [the observer](../cdb_cost_observer.py), [the runner](../scripts/cdb_exposed_cost_trace.py), and the unchanged [source contract](../artifacts/cdb_exposed_cost_trace_contract_v1.json). The author resolved the findings during review. This reviewer changed no implementation or shared runtime file.

## Findings and resolutions

| Initial finding | Final source status |
|---|---|
| TRACE exceptions lost partial observations because export followed successful generation. | Resolved. The runner exports rows, event usage, and restoration status in `finally`. It does not resolve timing events after generation fails. |
| Zero captures and recurrent hits did not exclude eager prelude/coda execution. | Resolved. Each measured phase requires `stage_graph_hits == coda_batches + sum(prelude_batches.values())`. |
| Missing publication events could receive an attribution PASS despite early exit. | Resolved. Publication count must equal the number of exit depths below four. Fixed R4 requires zero publication pairs. |
| TRACE parity checks occurred after CLEAN_AFTER already ran. | Resolved. The runner checks phase parity and TRACE coverage before any later phase. |
| Nested event-pool overflow replaced its RuntimeError with AssertionError and left a nonempty stack. | Resolved. Setup and native calls now share guaranteed stack cleanup. The original overflow error survives. |
| The job plan omitted the new implementation hashes. | Resolved. The plan records observer, runner, and CPU-test SHA256 values before model loading. |

The publication and overflow failures were independently reproducible on CPU before their resolution. The current publication guard rejects the missing-event case. The current overflow path preserves `RuntimeError('Timing event pool overflow: recurrent_stage')`. It clears the observer stack and restores overrides.

## Signatures and lifecycle

[ScopedMethods](../cdb_cost_observer.py:6) saves each raw instance attribute and its prior presence. Restoration reinstates an existing override by identity. It deletes a newly installed override so the original class method becomes visible again.

The wrapper closes over the original bound method. Its instance-installed function passes the original positional and keyword arguments unchanged. Metadata adapters match the pinned CDB call forms for batch, preparation, recurrent stage, exit decision, publication, coda stage, and consumption.

On the normal path, [the wrapper](../cdb_cost_observer.py:90) calls the original exactly once. It returns the original result unchanged. An initial-marker failure aborts before the original call. It does not continue unobserved work. An original exception remains primary if the end-marker call also fails. The observer records the cleanup exception type separately.

The common `_record_exit` recorder remains present in warm, clean, and trace phases. This matches the existing pilot-style depth recorder. Additional timing wrappers exist only during TRACE. The observer restores them before CLEAN_AFTER. It does not patch the frozen exit-policy object.

## Graph, stream, and metadata checks

The observers remain outside captured bodies. They do not replace `_forward_recurrent_stage`, projection modules, norms, the gate, sampling, or cache operations. Events record explicitly on the existing compute stream. No wrapper changes the caller stream or adds a synchronization inside native execution.

The pinned engine constructor gives the engine and runner the same I/O object. See [ContinuousDepthBatchingEngine.from_model](/home/esjung/external/decode-first-cdb-upstream/src/looped_cdb/continuous_depth_batching/continuous_api.py:328). The observer checks that the pool uses the runner's stream. The native publication method uses that shared stream.

The stage-hit rule matches the actual source. [_record_prelude](/home/esjung/external/decode-first-cdb-upstream/src/looped_cdb/continuous_depth_batching/continuous_api.py:1515) increments its batch count once per nonempty prelude launch. [Prelude replay](/home/esjung/external/decode-first-cdb-upstream/src/looped_cdb/continuous_depth_batching/model_runner.py:523) increments one stage hit. [Coda replay](/home/esjung/external/decode-first-cdb-upstream/src/looped_cdb/continuous_depth_batching/model_runner.py:1109) increments one stage hit. Request counter reset preserves graph objects while resetting these counters.

The publication count rule is specific to B1, refill OFF, same-step gate, and `last_exited`. Each nonfinal-depth exit creates one backlog entry. The wave flushes it before the next read or terminal stage. This rule is not a generic mixed-batch publication rule.

Metadata rows retain Python strings, integers, booleans, and new lists/dictionaries. They do not retain mutable work-item objects or tensor contents. Position, depth, hidden-slot, and backlog metadata precede native mutation. Graph-key dimensions use tensor metadata only. Nested parent IDs describe containing host calls, not a complete serving DAG.

[EventPool](../cdb_cost_observer.py:39) preallocates 1536 distinct event objects outside request clocks. Reset refuses unresolved reuse. The 512 recurrent pairs cover the maximum 508 decode passes at cap 128. Resolution follows the request-boundary synchronization. The observer does not multiply a final replay duration by a count.

## Independent CPU checks

The current module passes all 17 CPU tests. Syntax checking passes for the observer, runner, and tests. The actual runner's `source_checks` passes against the frozen contract.

Additional independent checks used the actual runner phase-loop AST with CPU fakes:

- A TRACE exception retains partial rows and restoration status. It does not call event resolution or execute CLEAN_AFTER.
- A TRACE token-parity failure stops before CLEAN_AFTER.
- A nested pool overflow preserves its primary error, empties the stack, and restores methods.
- A failing end marker does not replace an original ValueError. The existing instance override also retains its identity.

The AST checks execute real runner control flow with fake generation and events. They do not execute a CDB model or CUDA operation.

## Remaining limits

The hardware run must still establish token, depth, NFE, and finish-reason parity. These fields do not establish hidden-state, logit, or K/V bitwise equality. No such tensor-parity result follows from the CPU checks.

The hardware run must also pass capture, coverage, restoration, and overhead guards. A single request triplet supplies no timing confidence interval. A fatal CUDA failure remains a failure, even when diagnostic export succeeds.

Device intervals remain windows, not pure kernel busy times. Recurrent spans include the gate head and output copies. Publication spans include row preparation and host enqueue gaps. Coda spans combine staging, projection, and argmax. Host intervals contain or overlap device intervals. No additive 100% breakdown or removable-cost estimate follows.

`status.success` means the runner completed. Consumers must also inspect each request's `validity` and `attribution_permitted`. An observer-overhead or drift result can remain unresolved despite successful runner completion.

The frozen four-request selector, cap 128, native EOS/stops, both-size/both-mode scope, and unchanged original gate remain intact. This review does not authorize a GPU launch. Subsequent hardware output is outside this source/CPU review.

## Tested hashes and time

| File | SHA256 |
|---|---|
| `cdb_cost_observer.py` | `a2da8e314a094ada520df0f979a2333cbf57862779b741e77bea609b36653d8e` |
| `scripts/cdb_exposed_cost_trace.py` | `03bfa05f46e69b3e48066db1acc89db9605c4c82e28c2c243a088297f99e70fd` |
| `tests/test_cdb_cost_observer_cpu.py` | `64165158bfd4768bf616d826d8e2777b2314e8ce086e3330341502007126ce1c` |
| Frozen source contract | `835b3d7a3e6290fe91db5542e843c3a4fa5deef86fd3d6bb973709058a587c60` |

The source/CPU segment ran from `2026-10-19 16:26:56 UTC` to `2026-10-19 16:32:30 UTC`. Its elapsed time was 334 seconds. Report drafting and lint followed this segment.

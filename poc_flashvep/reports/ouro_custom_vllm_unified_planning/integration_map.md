# Source integration map

## Reading key

This table maps audited source to future changes. It does not record implemented patches.
Paths in the vLLM tables are relative to revision `568afb3a13806beb53bb2e6bd518269357b237c0`.
Exact source links appear in [evidence_index.md](evidence_index.md).

Scope classes are **REUSE**, **SMALL_ADAPTER**, **SUBSTANTIAL_CHANGE**, and **HIGH_RISK**.
The sizes describe design estimates. They do not predict line counts or delivery dates.

## Native model and execution path

| File and actual symbol | Current behavior | Proposed change | Scope / phase |
| --- | --- | --- | --- |
| `vllm/model_executor/models/ouro.py`, `OuroMLP` | Fused column-parallel gate/up and row-parallel down projections | Preserve operators and loaders | REUSE / all |
| Same, `OuroAttention` | Native QKV/O projections, RoPE, one Attention module per depth | Keep native attention. Register only locally owned logical depths for PP | SMALL_ADAPTER / P2 |
| Same, `OuroDecoderLayer.forward` | Fused residual and normalization path | Expose a bounded layer range without changing operation order | SMALL_ADAPTER / P2 |
| Same, `OuroModel.__init__`, `forward` | Physical `make_layers(24)`, four local loops, unused intermediate input | Implement embed/core/norm stages and explicit loop index | SUBSTANTIAL_CHANGE / P2 |
| Same, `OuroForCausalLM` | `SupportsLoRA`, not `SupportsPP` | Add PP interface only after staged semantics pass | SMALL_ADAPTER / P2 |
| Same, `load_weights`, `AutoWeightsLoader` | Native fused parameter mapping | Preserve tied storage and local ownership. Skip nonlocal weights explicitly | SMALL_ADAPTER / P2 |
| Same, `early_exit_gate` | Constructed, loaded, never consumed in forward | Expose trained gate only in the new Dynamic path | SMALL_ADAPTER / P3 |
| `vllm/model_executor/layers/linear.py`, `RowParallelLinear` | Parallel input defaults to true. Returns output and optional bias | Configure correct gate input partition. Do not rewrite the projection primitive | REUSE + gate adapter / P3 |
| `vllm/model_executor/models/utils.py`, `make_layers` | Splits a contiguous physical layer list | Add Ouro-specific ownership plan. Do not globally reinterpret `make_layers` | SMALL_ADAPTER / P2 |
| `vllm/model_executor/models/llama.py`, `LlamaModel.forward` | Working native first/intermediate/last PP pattern | Reference for intermediate tensors, not a recurrent executor | REUSE pattern / P2 |

Calling dormant Ouro gate code without an input-sharding audit is unsafe under TP4.
The full hidden input has width 2048. A TP4 row shard expects width 512 when input is already parallel.
Setting `input_is_parallel=False` is one candidate. Test loader, reduction, and bias behavior before adoption.
Do not change generic `RowParallelLinear` defaults for other models.

## Native workers, scheduling, and graph path

| File and actual symbol | Constraint found in source | Required extension | Scope |
| --- | --- | --- | --- |
| `vllm/v1/worker/gpu_worker.py`, `Worker.execute_model` | Waits prior PP sends, receives from predecessor, sends only for nonlast rank | Linear PP may reuse this path. Cyclic visits need explicit reentry and result routing | SMALL_ADAPTER linear / HIGH_RISK cyclic |
| `vllm/v1/worker/gpu_model_runner.py`, `GPUModelRunner.execute_model` | One outstanding `execute_model_state` until `sample_tokens` | Keep recurrent microsteps inside one logical token, or add a bounded internal state machine | SUBSTANTIAL_CHANGE |
| Same, model forward and `sample_tokens` split | Nonlast intermediate return. Final rank owns logits path | Add correct coda routing from a Dynamic exit owner | SUBSTANTIAL_CHANGE |
| `vllm/v1/executor/multiproc_executor.py`, `MultiprocExecutor` | Native multiprocess RPC and async result machinery | Reuse processes and error handling. Audit cyclic dispatch and in-flight ownership | REUSE / possible HIGH_RISK extension |
| `vllm/v1/engine/core.py`, `EngineCore.step_with_batch_queue` | Native queued model and sample futures | Preserve client request lifecycle. Add internal recurrent completion, not extra token completion | REUSE + bounded adapter |
| `vllm/v1/core/sched/scheduler.py`, `Scheduler` | Tracks computed token positions and token budgets | Add only necessary recurrent scheduling metadata. Do not increment token count per loop | SUBSTANTIAL_CHANGE if refill |
| `vllm/v1/core/sched/output.py`, scheduler output types | Batch token metadata for ordinary model calls | Versioned stage/depth descriptor or internal runner state | SMALL_ADAPTER synchronous / SUBSTANTIAL refill |
| `vllm/v1/worker/gpu_input_batch.py`, `InputBatch` | Persistent packed request state | Add survivor gather/scatter and stable request-row mapping | SUBSTANTIAL_CHANGE |
| `vllm/compilation/cuda_graph.py`, `CUDAGraphWrapper` | Graphs keyed by batch descriptor and runtime mode | Retain wrapper. Make stage/depth executable identity explicit and lifetimes stable | SMALL_ADAPTER + tests |

Do not multiply only the batch descriptor by depth and assume compilation remains correct.
The executable, registered attention context, slot mappings, and captured tensor addresses must also match.
Graph scopes can use per-stage modules or distinct wrappers. Select the smaller proven option in PP2.
Dynamic survivor batches need native supported capture buckets. Record padding separately from actual executed work.

Linear PP and cyclic PP may share transport and model operators.
They cannot assume identical dispatch topology or final-rank rules.
The PP2 cyclic proof is the boundary between a bounded fork and an engine rewrite.

## Native paged KV and transport

| File and actual symbol | Reuse | New work and test |
| --- | --- | --- |
| `gpu_model_runner.py`, `get_kv_cache_spec` | Discovers registered Attention modules | Check exactly 96 global namespaces and correct owner-local subsets |
| Same, `_reshape_kv_cache_tensors`, `initialize_kv_cache_tensors` | Native shape, stride, and layer-packing logic | Derive safe row views. Test packed and unpacked layouts |
| `vllm/v1/kv_cache_interface.py` | KV specification and group types | Avoid new cache format. Add readiness outside immutable specification where possible |
| `vllm/v1/core/kv_cache_utils.py` and allocator integration | Page sizing and grouping | Verify differing local namespace counts do not break rank capacity negotiation |
| `vllm/v1/attention/backends/flash_attn.py`, `do_kv_cache_update` | Native slot-mapped cache write | Evaluate reuse for repaired rows or direct strided row copies. Measure packing cost |
| `vllm/distributed/parallel_state.py`, `GroupCoordinator.isend_tensor_dict`, `irecv_tensor_dict` | Native tensor metadata and P2P primitives | Specify send order, receive completion, buffer leases, and cancellation |
| Native TP projection collectives | Native optimized all-reduce, including custom reduction kernels when selected | Record actual backend. Do not force NCCL-only behavior to create a slower baseline |

Primary `last_exited` must not use generic full-page replacement.
Copying a source page can overwrite older target rows with the wrong recurrent history.
The existing native scatter path is a candidate, not a proven repair API.
Some layouts may require several strided copies. Include every copy and synchronization cost.

## RLT: portable semantics versus private runtime

RLT pin: `4a350344027960a48384626aa646f1765fda6d4f`.

| RLT location | Useful concept | Portability decision |
| --- | --- | --- |
| `models/ouro.py`, `recurrent_prepared` | One normalized recurrent pass and gate/coda boundaries | Port boundary semantics. Keep native vLLM operators |
| `core/scheduler.py`, `_should_exit`, `update_from_output` | Token-local recurrent state and survivor routing | Reimplement against native request state. Resolve CDB arithmetic differences first |
| `core/kv_cache_manager.py`, `prepare_recurrent` | Per-depth metadata and logical-to-physical ownership | Map to native block tables. Do not copy its private cache allocator |
| Same, `mark_finalized`, `finalize_token` | Separate allocation, validity, and publication | Preserve current-row semantics. Add real receiver completion state |
| `worker/model_runner.py`, `prepare_on_cpu`, `prepare_on_gpu` | Preparation and readback lifecycle | Reuse concepts only. Native input buffers and graphs differ |
| Same, `finalize_many` | Batched repair | Uses a private Triton kernel and cache layout. Not a drop-in reuse path |
| Same, `finalize`, `_free_runner_slot` | Stream events and safe slot release | Reference lifecycle tests. Do not copy cross-process assumptions |
| `engine/stage_engine.py` | Separate engine/control architecture | Do not import as the native engine replacement |
| `config.py`, `ouro_delayed` and async modes | Delayed decision variants | Exclude from original-policy primary comparisons |

RLT's current model uses separate `nn.Linear` Q/K/V projections and a separate attention/cache implementation.
Its package name does not establish native vLLM TP or PP integration.
The audit found no ready-to-use native Ouro Loop PP4 patch in this tree.

## CDB semantic reference

| Location | Reuse or validation |
| --- | --- |
| `src/looped_cdb/continuous_depth_batching/exit_policy.py`, `ExitPolicy.should_exit` | Canonical cumulative hazard, min-depth handling, and sigmoid dtype |
| `continuous_depth_batching/continuous_api.py`, `_run_no_refill_decode_wave` | Cohort shrink, wave coda, repair flush, and successor ordering |
| `continuous_depth_batching/cache.py` and model cache interface | Cross-check current-row `last_exited` behavior |
| `models/ouro/modeling_ouro.py` | Independent recurrent and gate semantics |

The CDB paper includes additional asynchronous mechanisms. This plan does not import all of them.
Use the original gate without prediction or delayed exit in the initial Dynamic baseline.
Respect file-specific Apache-2.0 notices in this MIT-root repository.

## Expected modification envelope

The first native qualification needs only a harness and inspection hooks.
Linear Static PP likely affects the Ouro adapter, ownership plan, and cache-registration tests.
Dynamic adds runner microsteps, survivor packing, policy state, repair, and result routing.
Cyclic physical PP can also affect worker/executor dispatch and scheduler completion.
Across the full program, roughly 8–12 existing modules may require reviewed changes, plus tests and harness modules.
This is a planning estimate, not a committed patch count.
Do not approve that entire scope before the Phase 1 and PP2 gates.

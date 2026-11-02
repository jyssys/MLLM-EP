# Reference-code applicability

Status: SOURCE_AUDIT. No reference kernel or allocator was copied.

The execution baseline is commit `f4ecea59f36f620c69c4fef09c590bf790802cd9`.
The private RLT HEAD is `6b164d3844d35ab8e753df1864f0c87dc180401a`.
Its preserved working diff has SHA256 `4904a311358c5cf6467678d0e1c30af38b5a96064cae4c96ed680943d7e694c7`.
The reference tree remains read-only.

Paths under `source/` refer to `poc_flashvep/ref_loopserve/`.
RLT paths refer to `/home/esjung/external/looped-pd-abc-rlt/`.

| Feature | Reference source | Current RLT source | Direct reuse | Difference and decision |
| --- | --- | --- | --- | --- |
| Block allocation | `source/vllm/v1/core/kv_cache_manager.py:352` | `vllm_rlt/core/kv_cache_manager.py:186` | No | Reference groups logical caches through a coordinator. RLT reserves depth-indexed block tables. Reuse explicit denial accounting, not allocator objects. |
| Physical versus policy denial | `source/vllm/v1/core/kv_cache_manager.py:366` | `vllm_rlt/pd/engine.py:350` | Concept only | Reference separates physical shortage and completion-credit guards. Record P/D KV and transfer-slot failures separately. |
| Exact block release | `source/vllm/v1/core/block_pool.py:414` | `vllm_rlt/core/kv_cache_manager.py:358` | No | Reference uses block reference counts. RLT defers free while transfer leases exist. Preserve native leases. |
| Transfer ownership | `source/vllm/v1/core/sched/scheduler.py:1434` | `vllm_rlt/pd/connector.py:335` | No | Reference requires generation-specific release receipts. RLT has native NIXL ACK ownership. Do not release before ACK. |
| P/D reservation | No compatible native 1P1D endpoint identified | `vllm_rlt/pd/engine.py:336` | No | RLT reserves both workers before P executes. D must publish legal remote destinations. Delayed D reservation is not a flag change. |
| Pending transfer slots | Reference publication and runway slots use different ownership | `vllm_rlt/pd/connector.py:56` | No | Compute slots and transfer slots differ. Test supported draining limits without changing computation. |
| Mixed batching | `source/vllm/v1/core/sched/scheduler.py:3308` | `vllm_rlt/core/scheduler.py:806` | Existing RLT feature | Keep Native Mixed as a baseline. It is not a new mechanism. Confirm real mixed rows in execution evidence. |
| Recurrent KV semantics | `source/vllm/v1/worker/ouro_runner/runtime_owner.py:471` | `vllm_rlt/core/kv_cache_manager.py:367` | No | Reference contains pooled and ragged early-exit policies. This study retains full fixed-depth BF16 caches. |
| Numerical geometry | Reference README, batch-invariant Down projection section | Existing numerical-diagnosis patches | Already qualified | Keep Ouro `down_norm` and Huginn `down_norm_adapter`. Do not replace qualified arithmetic. |
| Instrumentation | `source/vllm/v1/worker/ouro_runner/runtime_owner.py:163` | Existing ModelRunner and host-execution observers | Concept only | Record host receipts as host receipts. They do not measure kernel completion. |
| Bounded graph trace | Reference separates optional device phase traces and host snapshots | New process-local profiler launcher | No binary reuse | Launch only the selected native stage under Nsight. Capture a short CUDA-profiler window. Validate perturbation against clean starts. |

## Compatibility and license

The reference README records Python 3.10, torch 2.10/CUDA 12.8, and custom compiled vLLM extensions.
The current qualified environment uses Python 3.14.6 and torch 2.11/CUDA 12.9.
The reference source retains an Apache-2.0 license in `source/LICENSE`.
Its compiled artifacts are external assets.
Those differences prevent a drop-in runtime or ABI claim.
This campaign copies no reference source or compiled artifact.

## Interpretation

The reference provides useful ownership and accounting principles.
It does not provide a proven fix for native RLT admission or the earlier Nsight stall.
The first controls change supported budgets, pool capacity, and transfer-slot limits.
No ACK ordering, KV value, model weight, recurrent depth, or attention operation changes.

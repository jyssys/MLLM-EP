# Fixed-shape tail root-cause source audit

## Runtime

- Model: local Qwen3-VL-30B-A3B-Instruct snapshot (`9c4b90e1e4ba969fd3b5378b57d966d725f1b86c`).
- vLLM: 0.20.0 V1, BF16, eager, TP2/DP2/EP4/PP1, DBO off, prefix cache off.
- Visibility: `CUDA_VISIBLE_DEVICES=1,2,3,4`; no process was started on physical GPUs 0 or 5--7.
- Runtime logs prove `Using DeepEPHTAll2AllManager`, `DeepEPHTPrepareAndFinalize`, EP world 4, linear 32/128 expert placement, and `Using TRITON Unquantized MoE backend`.

## MoE boundary

`Qwen3MoeDecoderLayer.forward` calls self-attention, post-attention normalization, then `self.mlp` (the sparse MoE). The local hook wraps the stock `FusedMoE.apply` call without changing tensors, routing, placement, or scheduler behavior.

## DeepEP HT dependency

In local `vllm/model_executor/layers/fused_moe/prepare_finalize/deepep_ht.py`, `_do_dispatch` obtains a `previous_event`, switches from the compute stream to the communication stream, and calls `get_dispatch_layout(..., previous_event=...)` and `dispatch(..., previous_event=...)`. The receiver invokes `EventOverlap.current_stream_wait()` before consuming the dispatched buffer. `_finalize` repeats the dependency for `combine(..., previous_event=...)`, followed by a receiver-side stream wait.

This is not an explicit global host barrier. It is an implicit collective/stream dependency: a prior asynchronous DeepEP operation (or peer-dependent NVSHMEM/NCCL kernel) can keep the communication stream busy; the next dispatch CUDA span then includes the wait until that dependency is usable. The same-device CUDA-event wrapper measures that span and never subtracts timestamps across GPUs.

## Measurement patch

`online_routing_geometry/hooks/sitecustomize.py` adds read-only wrappers around DeepEP `get_dispatch_layout`, `dispatch`, `combine`, and the modular `_fused_experts` call. Each wrapper records start/end CUDA events on the executing device and joins the record with layer, rank, phase, M, route id, and a local invocation id. Because stock V1 did not expose a scheduler iteration id to this hook, `scheduler_iteration_id` is explicitly labeled `local_moe_invocation_proxy`; it is not presented as a native scheduler counter.

The optional `FLASHVEP_SYNC_BEFORE_MOE=1` branch performs a diagnostic `torch.cuda.synchronize()` before MoE only; it is not enabled in baseline and is not an optimization.

## Nsight evidence

The full-serving profile captured child worker PIDs 3647414/15/18/19 on CUDA devices 2/3/0/1 (the four visible devices, corresponding to physical GPUs 3/4/1/2). SQLite contains 424,716 CUDA kernel rows. Actual DeepEP kernels are present: `notify_dispatch`, `dispatch`, `cached_notify_combine`, `combine`, `get_dispatch_layout`, and the `deep_ep::intranode::barrier` kernel. `fused_moe_kernel`, `topkGating`, FlashAttention, and NCCL AllGather/AllReduce kernels are also present.

NVTX was not emitted in this run; `nsys` reports an empty NVTX table. Attribution therefore uses actual kernel names plus same-device stage events, not an invented NVTX mapping.

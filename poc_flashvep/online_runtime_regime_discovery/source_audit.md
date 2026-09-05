# Source/runtime audit

## Environment

The live driver uses `CUDA_VISIBLE_DEVICES=1,2,3,4`, BF16 Qwen3-VL-30B-
A3B-Instruct, vLLM 0.20.0 V1, TP2/DP2/EP4, eager mode, no prefix cache,
`enable_expert_parallel=True`, linear placement and
`all2all_backend=deepep_high_throughput`.  The runtime log from the fresh
trace is preserved under the result directory and reports `world_size=4`,
`Using DeepEPHTAll2AllManager`, `DeepEPHTPrepareAndFinalize`, and 32/128
experts per rank.

## Communication controls

In local vLLM 0.20.0, `vllm/distributed/device_communicators/all2all.py`
sets `DeepEPAll2AllManagerBase.num_sms=20` and
`DeepEPHTAll2AllManager.set_num_sms()` calls `deep_ep.Buffer.set_num_sms`.
The V1 `gpu_ubatch_wrapper.py` reads `VLLM_DBO_COMM_SMS` even with DBO
disabled and applies a temporary SM allocation around each ubatch.  DeepEP
requires an even SM count; values above the buffer's 20-SM cap are clamped.
This sprint therefore tested 8, 12, and 20 as separate engine runs.  The
knob changes communication-kernel resource allocation only; no routing or
model math is changed.

The installed `ParallelConfig` exposes `deepep_high_throughput` and
`deepep_low_latency`.  Low-latency uses the DeepEP LL manager and reports
zero communication SM usage; it was attempted only if a bounded engine run
could initialize.  Other registered backends (NIXL, FlashInfer, allgather)
were not silently substituted because their native dependencies are absent or
would change the validated execution path.

## MoE boundary

`DeepEPHTPrepareAndFinalize._do_dispatch()` obtains a prior CUDA event,
calls `get_dispatch_layout(previous_event=...)`, then `dispatch(...,
previous_event=...)`; the receiver waits on the returned `EventOverlap` before
expert execution.  `_finalize()` calls `combine(..., previous_event=...)` and
waits before copying the reduced result.  The online observer wraps the stock
`FusedMoE.apply` and places CUDA events around the complete apply interval,
recording the exact top-k route and rank/expert histograms.  It intentionally
does not invent dispatch/expert/combine sub-durations that are not separately
instrumented in the serving process.

## Online scheduler

The driver submits varied waves through the V1 `LLM.generate` request queue;
the normal scheduler performs continuous batching and chunked prefill.  The
trace records `phase`, `M`, layer, DP/EP rank, and route context.  A synthetic
`M=4096` memory-profile forward is excluded from natural summaries.

## Analysis join

Each hook record contains a route file such as
`route_00000001_dp0_l8.npz`.  `aggregate_by_route.py` normalizes the DP suffix
to `route_00000001_l8` and joins the four EP-rank records for one invocation.
This preserves rank-local CUDA durations without subtracting clocks from
different GPUs.  The older timestamp-window aggregator remains available for
comparison but is not the final primary join.

## Limitations

The current hook is a read-only observer.  It records full stock MoE CUDA
intervals, not a direct rank-synchronized dispatch/expert/combine breakdown;
phase decomposition in this report is therefore source-backed or inferred.
Different SM configurations are separate process runs with natural wave
variation and are not a matched-route causal replay.  These limitations are
explicit in the gate and prevent an adaptive-method claim.

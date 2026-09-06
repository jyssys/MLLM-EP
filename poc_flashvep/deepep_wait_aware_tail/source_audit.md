# DeepEP wait-aware tail source audit

## Runtime

The measurements use the local vLLM `0.20.0+cu129` package under
`/home/esjung/anaconda3/envs/flashvep-poc/lib/python3.12/site-packages/vllm`
and DeepEP from `/home/esjung/.venvs/flashvep-deepep-v020` (commit recorded in
the run metadata).  The serving driver uses `CUDA_VISIBLE_DEVICES=1,2,3,4`,
TP=2, DP=2, EP=4, BF16, `deepep_high_throughput`, eager execution, DBO off,
and prefix caching off.

## Exact call path

`Qwen3MoeDecoderLayer.forward` invokes the selected vLLM fused-MoE method.
The local observer wraps `FusedMoEModularMethod.apply` and
`UnquantizedFusedMoEMethod.apply`, preserving tensors and the stock call.
The DeepEP HT finalize path (`vllm/.../prepare_finalize/deepep_ht.py`) calls:

1. `dbo_get_previous_event(self.buffer.capture)` before dispatch;
2. `buffer.get_dispatch_layout(..., previous_event=previous_event)`;
3. `buffer.dispatch(..., previous_event=previous_event, async_finish=...)`;
4. `event.current_stream_wait()` before expert execution;
5. expert Triton/FusedMoE kernel;
6. `buffer.combine(..., previous_event=previous_event, async_finish=...)`;
7. a second `event.current_stream_wait()` before returning output.

`deep_ep.Buffer.dispatch` and `combine` pass the Python event's underlying
`EventHandle` to the C++ intranode implementation.  The installed
`EventHandle` exposes `current_stream_wait()` but no non-blocking `query()`;
therefore direct readiness is not available through this API.  The hook records
whether a previous event object was supplied and puts CUDA events around every
stock `EventOverlap.current_stream_wait()` call.  Those events are resolved
after the MoE call, without inserting a synchronization in the baseline.

With DBO disabled, `dbo_get_previous_event` normally returns `None` because
the vLLM ubatching context map is empty.  The giant tails observed previously
are consequently attributed to DeepEP's internal asynchronous notification /
barrier state on its communication stream (`notify_dispatch` and
`deep_ep::intranode::barrier`), rather than an exposed DBO event argument.

## Diagnostic interventions

`FLASHVEP_SYNC_BEFORE_MOE=1` is the broad diagnostic P1 and calls global
`torch.cuda.synchronize()` immediately before each MoE.  The new hook also
supports a narrower communication-stream drain before dispatch:

* `FLASHVEP_COMM_STREAM_SYNC=1` drains `Buffer.get_comm_stream()` before stock
  dispatch;
* `FLASHVEP_POLICY=online_simple` with
  `FLASHVEP_SIMPLE_PREV_DISPATCH_THRESHOLD_MS=<x>` drains that stream only when
  the previous same-process dispatch event exceeded `x` ms;
* `FLASHVEP_ORACLE_SYNC_INVOCATIONS=<comma-separated local ids>` performs the
  same relevant-stream drain for offline-selected invocations (or global sync
  if `FLASHVEP_ORACLE_INTERVENTION=global`).

These are bounded measurement interventions, not a production scheduler.

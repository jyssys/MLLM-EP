# ASAP / vLLM / DeepEP source audit

## ASAP

The paper is *A Disaggregated and Asynchronous Inference System for MoE
Prefill* ([arXiv:2606.22541](https://arxiv.org/abs/2606.22541)).  Its
characterization attributes the synchronous-baseline loss to DP attention
progress differences before a shared EP MoE stage, with request-length
variance creating a barrier/bubble that increases TTFT and reduces throughput.
The author page ([sc2682cornell/sc2682cornell.github.io](https://github.com/sc2682cornell/sc2682cornell.github.io))
links the paper but no public official ASAP implementation was found in the
bounded repository search: `ASAP_CODE_STATUS: NOT_PUBLICLY_FOUND`.

## Local runtime

The experiment used vLLM `0.20.0`, PyTorch `2.11.0+cu129`, Triton `3.6.0`,
BF16 Qwen3-VL-30B-A3B-Instruct, four H100s exposed as physical GPUs
`1,2,3,4`, DBO off, prefix caching off, eager execution, and DeepEP high
throughput.  Runtime proof files record the actual groups and PIDs.

| topology | TP | DP | EP | TP group | DP group | EP group | sequence-parallel MoE |
|---|---:|---:|---:|---|---|---|---|
| A | 2 | 2 | 4 | 2 ranks | 2 ranks | 4 ranks | true |
| B | 1 | 4 | 4 | 1 rank | 4 ranks | 4 ranks | false |

The local `ParallelConfig.use_sequence_parallel_moe` predicate enables
sequence-parallel MoE only when TP>1 and DP>1, so this is a topology confound
that is recorded rather than hidden.

## DeepEP HT synchronization semantics

The installed source is
`/home/esjung/anaconda3/envs/flashvep-poc/lib/python3.12/site-packages/vllm/model_executor/layers/fused_moe/prepare_finalize/deepep_ht.py`.
`_do_dispatch` captures a compute-stream `previous_event`, switches to the
communication stream, calls `get_dispatch_layout(previous_event=...)` and
`buffer.dispatch(previous_event=...)`, then switches back to compute.  The
receiver calls `event.current_stream_wait()` before consuming dispatched
tokens.  Finalization similarly calls `buffer.combine(previous_event=...)`
and waits on the returned event before copying the combined output.  There is
no explicit Python/global barrier in `DeepEPHTPrepareAndFinalize`; the
relevant synchronization is the DeepEP collective plus CUDA stream/event
dependencies.

vLLM logs also state “Asynchronous scheduling is enabled” and “Disabling NCCL
for DP synchronization when using async scheduling.”  Therefore the tested
runtime is not the same as a blocking global-NCCL-barrier baseline.  This is a
key reason a paper-style wait cannot be inferred from a duration spread alone.

## Measurement caveat and validation

CUDA event timestamps from different GPUs were never subtracted as an
absolute global clock.  The direct `EventOverlap.current_stream_wait()` span
is an asynchronous wait enqueue duration, not a complete wall-clock wait.
The closest complete collective-span proxies available without changing
execution semantics are the host `prepare_host_ms`, dispatch CUDA span, and
the `ep_entry_to_done_ms` span.  A calibrated diagnostic GPU delay measured
approximately 0.515, 1.020, and 2.030 ms for requested 0.5, 1, and 2 ms, and
caused monotonic increases in peer prepare/dispatch spans in the layer-24
trace.  This validates sensitivity of the instrumentation while not proving
that natural runs have a global barrier.

The bounded Nsight Systems capture is preserved at
`../asap_sync_phenomenon_reproduction_20260901_nsys_capture/nsys_b256.nsys-rep`.
It contains DeepEP dispatch/combine kernels and CUDA API activity; no custom
NVTX ranges were added and the small capture did not provide a reliable
per-rank idle attribution.

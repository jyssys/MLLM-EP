# Phase 1b expert timing audit

Date: 2026-08-03 KST

## Conclusion

`0.447 ms is plausible but measures only pure local GEMM`.

More precisely, the prior TP4 value is a max-over-ranks CUDA-event span around
`FusedMoEKernelModularImpl._fused_experts`. It is a fused execution of all
assignments mapped to the rank's 32 local experts, not the time of one expert
and not the complete MoE phase. It excludes the router, dispatch, prepare,
DPEP/TP combine, and the full decoder layer.

## Prior TP4 instrumentation boundary

- Start and end CUDA events were recorded on the worker's current CUDA stream,
  immediately before and after `_fused_experts`.
- Events were not synchronized at every stage. The end event was synchronized
  while records were flushed at worker shutdown, so `0.447 ms` is not merely a
  Python launch/enqueue duration.
- The previous TP4/DP1 path was
  `MoEPrepareAndFinalizeNoDPEPModular`: there was no DPEP dispatch, expert input
  was replicated by TP semantics, and the output used TP all-reduce.
- The prior local-load table was reconstructed by slicing a replicated global
  routing histogram. It was useful for ownership accounting but was not an
  observed post-DPEP tensor.

## Phase 1b in-path cross-check

The final TP2/DP2/EP4 selected-layer profile used the same `_fused_experts`
boundary. Across all selected iteration/layer samples, `T_expert_max` was:

| Statistic | Value |
|---|---:|
| median | 0.448672 ms |
| p90 | 0.478192 ms |
| mean | 0.454631 ms |
| stddev | 0.030436 ms |

Per-layer medians were 0.449872, 0.444656, 0.451568, 0.447696, and
0.445664 ms for layers 0, 12, 24, 36, and 47. This independently reproduces
the scale of the prior 0.447 ms result under a real DPEP path.

## Isolated fused-expert microbenchmark

The microbenchmark captured the already-dispatched layer-24 inputs, performed
20 warm-ups, then measured 100 CUDA-event iterations without saving weights or
activations. Its `[802, 2048]` gathered input contains the real TP-padded
request (800 tokens) and the idle DP rank's two dummy tokens.

| rank / GPU | local experts | assignments | max expert batch | median | p10 | p90 | estimated TFLOP/s |
|---|---|---:|---:|---:|---:|---:|---:|
| 0 / GPU 4 | 0-31 | 1,194 | 353 | 0.247264 ms | 0.233997 | 0.271286 | 45.57 |
| 1 / GPU 5 | 32-63 | 1,725 | 327 | 0.253856 ms | 0.239789 | 0.279299 | 64.13 |
| 2 / GPU 6 | 64-95 | 1,528 | 322 | 0.260400 ms | 0.248854 | 0.281517 | 55.38 |
| 3 / GPU 7 | 96-127 | 1,969 | 363 | 0.249840 ms | 0.239574 | 0.264803 | 74.37 |

The microbenchmark is intentionally faster than the in-path critical expert
span: it reuses warm inputs and omits dispatch, inter-stage handoff, and normal
request scheduling. The measured 45.6-74.4 estimated TFLOP/s and hundreds of
assignments split across 32 small local experts are physically credible for a
fragmented Triton BF16 MoE workload. They do not imply that the full MoE layer
finishes in 0.25-0.45 ms.

## Scope and bias assessment

- Correct: the event duration is a completed GPU span on one rank, and the
  reported maximum captures the critical local-expert rank.
- Limited: it measures one fused local-expert invocation, not a single expert
  and not dispatch-to-combine latency.
- Expected warm-cache bias in the isolated microbenchmark does not invalidate
  the in-path 0.447 ms-scale cross-check.
- The earlier detailed and lean whole-request profilers had +59.66% and
  +25.42% overhead. That makes their request latency unsuitable for a small
  speedup claim, but it does not turn the shutdown-synchronized per-stage CUDA
  event into an asynchronous launch-only measurement.

Raw evidence is in
`poc_flashvep/results/phase1b_tp2dp2_vision896/audit.jsonl`,
`stage_events_v2.jsonl`, and `analysis_final.json`.

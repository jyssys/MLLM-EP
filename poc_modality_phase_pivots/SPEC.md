# Working contract

- Base repository HEAD is recorded before measurement; previous artifacts are
  read-only inputs and no prior result directory is overwritten.
- Every GPU command must see exactly `CUDA_VISIBLE_DEVICES=4,5,6,7`.
- Track 1 measures unsplit, full-size Qwen3-VL language Attention against real
  DeepEP dispatch, expert, combine, and whole-MoE execution. The primary
  statistic is `eta=(Tx+Ty-Txy)/min(Tx,Ty)` using rank-critical CUDA-event
  durations and medians, not maxima.
- Track 2 uses natural request/layer tasks. It reports perfect, measured, and
  simple-heuristic schedule oracles separately and never calls a layer proxy an
  observed TTFT gain.
- Track 3 does not switch A2A backend. It tests current-source-supported
  communication SM allocation and aggregation of whole request execution
  units. Token-axis fragmentation and CPU offload are forbidden.
- Clean and instrumented requests are interleaved. Same-device events only;
  no absolute timestamp subtraction across GPUs.
- Correctness is checked with greedy tokens and saved final logits. Any
  incorrect path is excluded from positive performance evidence.
- Each track receives an independent GO/HOLD/NO-GO decision. No implementation
  is promoted unless its stated request-level threshold is met.

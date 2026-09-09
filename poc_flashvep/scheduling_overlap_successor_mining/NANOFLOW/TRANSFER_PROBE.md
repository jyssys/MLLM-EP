# NanoFlow transfer: what is actually executed

The official H100 Qwen1.5-MoE-A2.7B EP4 path is live: 24 layers, 60 routed
experts, top4, FP16, shared experts and output all-reduce. This is not a
synthetic isolated GEMM, but also not DeepEP or Qwen3-VL. Full reproduction
scope and checkpoint pins are in REPRODUCTION.md and CODE_AUDIT.md.

The bounded adapter exposes native FFN nano splitting, dependency execution,
buffer planning and stream choice. Live CUDA/NVTX traces verify overlap. It
does not supply missing MoE auto-search profiling hooks, so manually selected
plans must not be described as the original paper's searched-optimal baseline.

Large pure-prefill inputs at M4096/8192 use real FineWeb text, four actual
requests, three restarts and two exact-shape warmup cohorts. All first-token
cross-plan checks pass. An expanded decode portfolio tests B4/B64/B256 next.
These are fixed cohorts, not shape-changing native continuous serving.

Fresh Qwen3-VL real-image operation/request traces are separately available;
see `../COMMON/LARGE_MATCHED_TRANSFER.md`. The visible ~14–16% entire-TTFT share
for four-chart requests is not a measured nano-plan oracle. No full native VL
port is started without material, correctness-valid request-level plan regret.

# Qwen3-VL EP8 critical-rank coalescing source audit

## Configuration

- checkpoint: `Qwen3-VL-30B-A3B-Instruct` snapshot `9c4b90e1e4ba969fd3b5378b57d966d725f1b86c`
- architecture: `Qwen3VLMoeForConditionalGeneration`
- nested text config: 48 decoder layers, hidden size 2048, 128 routed experts,
  top-8 (`num_experts_per_tok=8`), routed MoE on every decoder layer
- topology: TP2 / DP4 / EP8 / PP1; 16 experts per EP rank
- placement: linear `expert_id // 16`; BF16; DeepEP high-throughput;
  TritonExperts; EPLB/DBO/prefix caching disabled; eager execution

## Runtime path and instrumentation

The experiment uses the existing vLLM V1 Qwen3-VL path.  A local
`sitecustomize.py` installs read-only hooks in each child worker.  The router
hook wraps `BaseRouter.select_experts` and records the returned top-k logical
expert IDs and weights.  A decoder-layer context resolves the layer index.
The shared DeepEP hook wraps `_prepare`, `_fused_experts`, and `_finalize` and
records per-EP-rank CUDA-event durations for dispatch, routed expert execution,
and combine.  No route, placement, scheduler, or model tensor is modified.

The router sees sequence-parallel TP shards, so each file stores exact global
token positions reconstructed from the TP rank and the immutable prompt token
manifest.  The analysis concatenates the two DP0 TP shards into one canonical
invocation.  Hidden vectors are sampled only for Vision rows (up to 128 per TP
shard) at layers 16, 24, and 40 to bound storage.

## Measurement-mode caveat

The vLLM 0.20 DP4 multimodal path can leave an empty DP engine waiting in its
shared-memory broadcast.  To make the EP8 collective participate, the same
real image request is submitted to all four DP engines in this bounded
measurement mode.  It is not a serving policy.  Canonical route metrics use
one DP0 copy; raw timing includes the four replicated copies.  Rank ratios are
therefore shape diagnostics, not a four-request throughput claim.

## What is not captured

The trace does not contain alternate route executions or actual Qwen3-VL EP8
expert outputs for each sampled hidden vector.  Consequently hidden cosine is
only a redundancy candidate signal.  Coalescing results are trace-driven
oracle/count-cost estimates, not measured coalescing speedups or a quality
claim.

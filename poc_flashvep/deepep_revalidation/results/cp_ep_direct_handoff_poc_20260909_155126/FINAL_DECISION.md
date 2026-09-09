# Final Decision

## Status

`NO_GO` for both proposed directions in the tested four-H100 Qwen3-VL regime.

- Exact Streaming Handoff: `NO_GO_ACTUAL_GAIN_AND_CORRECTNESS`.
- Exact Direct CP-to-Expert Transport: `ALGEBRAICALLY_BLOCKED_ROUTE_DEPENDENCY`.

## Why the ideal oracle did not transfer

The serial layer exposes a nominal 37.5–49.3% overlap window because attention and MoE have comparable durations. Realizing it, however, requires replacing one efficient full-token Router/DeepEP/grouped-GEMM invocation with 2–128 micro-invocations. On clean timing, streaming changes same-chunk latency by only −0.00% to −0.78% and is 67–604% slower than the whole serial layer for the representative 256/1024 blocks. The detailed full sweep shows the same sign and regime. Fixed per-invocation layout, dispatch, combine, launch, and small grouped-GEMM costs dominate; attention and EP also contend for the same H100/NVLink resources.

The decomposition additionally fails strict exact integrated routing: BF16 block accumulation changes top-k for 0.5–3.4% of assignments depending on layer.

Direct transport cannot know the expert owner before exact dense output projection, residual, RMSNorm, and router evaluation. Any exact formulation retains the CP hidden assembly/reduction and EP hidden transport. Its valid structural-byte and layer-time oracle is 0%, before additional metadata and reductions.

## Scope

These are actual layer-level GPU results with Qwen checkpoint weights and real Qwen3-VL hidden distributions. They are not request-level native serving results. Full-model integration and Kimi were correctly skipped because the Qwen layer prototype failed the >=10% promotion gate and correctness requirements.

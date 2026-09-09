# Exact boundary of the optimized MoE search reproduction

Official `dev-h100` pin: `915790ea862d1ddd52a8871282c1eb5be88f1391`.
This is **not** a claim that NanoFlow cannot run MoE: the supported Qwen EP4
forward, native splits, graph execution and CUDA stream overlap all run.

The unavailable milestone is a faithful, fully profiled/search-optimized MoE
plan under changing MLLM workloads:

- `nanoflow/operations/fused_moe/fused_moe.py:113` derives `FusedMoE` directly
  from `Operations`. Its actual methods implement shape, weights and forward,
  but none of the profile/database methods below.
- `operations/operation_base.py:103` and `:110` raise `NotImplementedError` for
  profile DB creation/storage. `:163` and `:166` have empty update/run hooks.
  The profile loop records events around that hook, so simply suppressing the
  database exception would silently time no expert work. That is not a valid
  profile or an optimized-baseline result.
- `nanoflow/auto_search/search.py:649` instantiates the dense Llama3-70B
  AllReduce pipeline and consumes its operation-specific profile tables.
  Reusing them for Qwen MoE would change the workload and miss real expert cost.
- The Qwen factory's categories, optional-copy interfaces and phase lifecycle
  differ from the dense auto-search model. Bounded compatibility fixes expose
  existing execution, but do not complete those missing profiling semantics.

The separate official Meta-Llama access probe returned HTTP401. That is an
access limitation, not the cause of the missing Qwen MoE hooks. No gated mirror
or authentication workaround was used. Restricted Gurobi is a possible larger
solver limitation; it is not labelled an observed Qwen solve failure when no
faithful MoE solve was attempted.

## Alternate evidence actually obtained

1. Official ungated Qwen1.5-MoE EP4 plus independent HF numerical reference.
2. Native fixed-plan splitter/executor with graph and separate-stream controls.
3. Same-prefix, same-plan and independent-HF near-tie diagnostics.
4. CUDA/NVTX confirmation of operation overlap, separately from clean latency.
5. Three-restart, exact-first-token large-prefill finite-plan envelope; expanded
   decode-volume control; independent real-image Qwen3-VL operation traces.

These alternatives support bounded screening, not a failure of the original
searched optimum. Completing real expert profiling and calibrated interference
costs would be baseline-port work before any successor experiment. It should
not be disguised as a new scheduling method or as a universal zero-headroom
conclusion from the measured manual portfolio.

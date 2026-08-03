# Debate Note: TP4/EP4 Phase 1 Gate

Date: 2026-08-03 (Asia/Seoul)
Decision: **Phase 2 NO-GO / HOLD**

## Experiment

- Physical GPUs: 4,5,6,7 only (logical ranks 0-3).
- Qwen3-VL-30B-A3B-Instruct exact snapshot
  `9c4b90e1e4ba969fd3b5378b57d966d725f1b86c`.
- BF16, TP=4, effective EP=4, DP=1, PP=1, Triton Unquantized MoE,
  FlashAttention 3, eager execution, prefix caching disabled.
- Fixed batch-size-1 prefill, one 896x896 gray image, fixed prompt, greedy
  `max_tokens=1`.
- Five warm-ups and twenty measured iterations per pass.

The current processor identifies `<|image_pad|>` as token ID 151655. The exact
799-token classification is visual indices 4-787 (784 tokens), vision start
at index 3, vision end at index 788, and 15 remaining text/special indices.
Every iteration produced token 1986 (`This`) and route shape `[799,48,8]`.

`batch_size=1` is the request batch, not the fused-MoE token batch. Each
prefill layer processes 799 tokens and top-k 8 creates 6,392 routed
assignments. Across iteration/layer samples, a rank receives a median 1,616.5
local assignments; the busiest rank has median 1,989 assignments. The busiest
single expert has median 416.5 assigned tokens and an observed maximum of 587.

## Baseline And Profiler Overhead

| Pass | Median | p90 | Mean | Stddev | Overhead |
|---|---:|---:|---:|---:|---:|
| No profiler | 75.881 ms | 87.258 ms | 77.752 ms | 4.568 ms | reference |
| Detailed 14-stage | 121.152 ms | 164.009 ms | 128.861 ms | 17.162 ms | +59.66% |
| Lean layer/router | 95.173 ms | 96.888 ms | 95.878 ms | 2.695 ms | +25.42% |

The detailed trace contains 53,760 valid records with full 20x48x4x14
coverage and no error/null duration. Synchronization happens only at worker
shutdown. Nevertheless, annotation/event launch overhead is material because
the original H100 layer path is short and launch-sensitive.

Mixing the lean outer-layer durations with detailed component durations is
invalid and gives 0.792x. That value is retained only as an overhead warning,
not as a speedup estimate. The reported oracle result uses one internally
consistent detailed pass. Its inflated outer `T_layer` makes it favorable to
the hypothesis, so a failure is conservative with respect to the 1.15 gate.

## Stage Breakdown

Median across all 960 measured iteration-layer samples:

| Stage/value | Median |
|---|---:|
| `T_layer` | 1.998 ms |
| `T_attention` (full attention block) | 0.908 ms |
| QKV projection | 0.078 ms |
| attention core | 0.191 ms |
| attention output projection | 0.817 ms |
| `T_norm_router` | 0.212 ms |
| `T_dispatch` | 0.000 ms (collective structurally absent) |
| `T_expert_max` | 0.447 ms |
| `T_combine` | 0.808 ms (TP all-reduce) |
| complete fused MoE call | 1.057 ms |
| expert fraction | 22.08% |
| exposed non-expert fraction | 56.17% |
| `T_optimistic` | 1.925 ms |
| per-sample oracle speedup | 1.035x |

The QKV/core/output medians are separately measured subranges; their sum is
not substituted for the full attention block because Q/K normalization and
RoPE remain inside the latter boundary. Residual+RMSNorm ranges retain the
runtime's fused semantics.

`T_expert_max` is not the latency of the slowest individual expert. It is the
maximum across four ranks of `_fused_experts`, one Triton fused-kernel span
covering all routed assignments to that rank's 32 local experts. Individual
expert start/end times are not separable in this backend and were not claimed.
The 0.908 ms attention block includes the row-parallel output projection and
its TP all-reduce; that output-projection range alone has a 0.817 ms median.
The FlashAttention core itself has a 0.191 ms median.

Across the per-request sums of all 48 layers:

- median `T_layer`: 96.930 ms;
- median `T_optimistic`: 93.159 ms;
- median oracle speedup: **1.0349x**;
- p90 oracle speedup: **1.0785x**;
- median exposed non-expert fraction: **55.92%**.

No layer has a per-layer median oracle speedup >=1.15x. Layer 0 is the slowest
by median detailed `T_layer` at 2.444 ms. Logical rank 2 / physical GPU 6 is
the most frequent whole-layer critical rank (297/960), while logical rank 3 /
physical GPU 7 is the most frequent expert critical rank (492/960).

## Gate

| Requirement | Result |
|---|---|
| Oracle layer speedup >=1.15x | **Fail:** 1.0349x median, 1.0785x p90 |
| Exposed non-expert fraction >=15% | **Pass:** 55.92% |
| Combined Phase 2 gate | **Fail** |

`T_optimistic` is a measured-component estimate using the specification's
oracle formula. It is not measured overlap, and no first-tile timing exists.

## Decision

Phase 2 was not executed. No token tiles, trace-driven scheduler, offline MoE
replay, custom CUDA/Triton kernel, or live overlap path was added. Advancing
despite this result would violate the explicit Phase 1 gate, and the current
runtime path has no dispatch collective to overlap in the first place.

Primary evidence:

- `poc_flashvep/results/tp4_phase1_vision896/stages.jsonl`
- `poc_flashvep/results/baseline/summary_tp4_ep4_vision896.csv`
- `poc_flashvep/results/baseline/layer_breakdown_tp4_ep4_vision896.csv`
- `poc_flashvep/results/baseline/gate_tp4_ep4_vision896.json`
- `poc_flashvep/results/tp4_phase1_vision896/{baseline,profile,lean}_requests.json`

# True-EP2 runtime audit

## Verdict

The measured substrate is true expert parallelism, not replicated experts or
tensor parallelism mislabeled as EP.  It runs TP1/DP2/EP2 over physical GPUs 6
and 7, with DeepEP high-throughput dispatch/combine and Triton expert kernels.

## Pinned environment

| Item | Value |
|---|---|
| Model | `inclusionAI/LLaDA-MoE-7B-A1B-Instruct`, local fused checkpoint |
| Model dtype | BF16 expert weights and activations; router linear evaluated in FP32 |
| dInfer source | `/home/esjung/dInfer-dllm-ep`, `1ffeb961cd258bede74fcf5ca8a416ae6d57b18f` |
| Minimal substrate patch | expert-weight loading follows `get_ep_group()` rather than the TP group |
| vLLM | 0.10.2 |
| DeepEP | 1.2.1+73b6ea4 |
| PyTorch / CUDA | 2.8.0+cu128 / CUDA 12.8 |
| Backend | `deepep_high_throughput` |
| Manager | `DeepEPHTAll2AllManager` |
| MoE path | `FusedMoEModularKernel -> DeepEPHTPrepareAndFinalize -> TritonExperts` |
| Parallelism | TP=1, PP=1, DP=2, EP=2 |
| Physical GPUs | 6 (`GPU-e3f3998e-...`) and 7 (`GPU-4cc26b88-...`) only |
| Interconnect | NV18/NVLink between GPUs 6 and 7 |

The local dInfer model creates a replicated router and calls vLLM's
`FusedMoE.forward_impl` at
`python/dinfer/model/modeling_fused_olmoe.py:667-700`.  The minimal prior patch
at lines 1122-1133 shards expert weights by the EP group.  No model/router
semantics were changed for this study.

## Ownership proof

| Rank | Physical GPU | EP rank | Global experts | Count | Local expert bytes |
|---:|---:|---:|---|---:|---:|
| 0 | 6 | 0 | 0--31 | 32 | 402,653,184 |
| 1 | 7 | 1 | 32--63 | 32 | 402,653,184 |

There are 64 global experts, top-k=8, and one expert occupies 12,582,912 bytes.
The two expert maps are disjoint and exhaustive.

## Communication/execution proof

The fresh profiler capture contains all of the following GPU kernels:

- `deep_ep::layout::get_dispatch_layout`;
- `deep_ep::intranode::notify_dispatch<2>`;
- `deep_ep::intranode::dispatch<2,...>`;
- Triton fused-expert kernels;
- `deep_ep::intranode::cached_notify_combine<2>`;
- `deep_ep::intranode::combine<...,2,...>`;
- `_moe_C::moe_sum`.

For M=64 on rank 0, same-device event spans were dispatch 0.664 ms, expert
0.576 ms, combine 0.142 ms.  Rank outputs were bit-identical
(`max_abs=0`, relative-L2=0).  This is direct evidence of remote token movement,
local-sharded expert execution, and combine—not inference from a configuration
flag.

## Real request stage share

The cache-off gen=64 request executes 19 denoising forwards in the instrumented
run.  Critical-rank sums and their share of the independently measured clean
request median (620.63 ms) were:

| Stage | Sum (ms) | Clean-request share |
|---|---:|---:|
| Router linear | 16.71 | 2.69% |
| Dispatch | 102.22 | 16.47% |
| Expert | 138.10 | 22.25% |
| Combine | 22.25 | 3.58% |
| Whole MoE wrapper | 333.71 | 53.77% |

These shares establish that MoE is economically relevant, but do not say that
the entire MoE span is removable.

## Timing validity and observer tax

Three clean engine restarts were 620.63, 615.33, and 673.00 ms (median 620.63
ms, CV 5.01%).  The same-device stage-event run was 596.98 ms, or -3.81% versus
the clean median.  A negative observer tax is physically meaningless and shows
that restart noise dominates at this sample count; stage boundaries are used
for attribution, not for a claim that instrumentation accelerates execution.

The all-layer expository tensor capture took 839.34 ms, +35.24% over its clean
request denominator.  It is explicitly excluded from latency evidence.  No
cross-GPU absolute timestamp subtraction is used.

## Current physical-work behavior

The studied dInfer path uses `cache_factory=None`.  Each model forward presents
the full 64-token prompt plus 64-token generation block (physical M=128) to all
16 MoE layers, even as the number of live masks shrinks.  Therefore stock dInfer
rebuilds the route and executes 128×8 routed assignments per layer per
iteration.  This is the physical baseline against which the work gap is
measured.

Raw evidence: `results/ep2_residual_20260911_191336/runtime_audit/` and
`stage_timing/`.

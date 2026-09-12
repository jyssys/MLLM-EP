# Environment and execution truth

## Scope

- Date: 2026-09-13 KST.
- Project branch: `flashvep/dllm-moe-ep-discovery-poc`, base `238c4521fd8142324ca709c1dd1c8c5a7be802dc`.
- Instrumented dInfer worktree: `/home/esjung/external/dinfer-llada2-flash-discovery`, commit `9132ce9a` on base `1ffeb961cd258bede74fcf5ca8a416ae6d57b18f`.
- Model revision directory: `/home/esjung/models/LLaDA2.0-flash-744c3f8`.
- Model config: BF16, hidden 4096, 32 layers, 256 routed experts, top-8, one shared expert, expert intermediate 1024, 8 routing groups/top-4 groups.
- Software: Python 3.11.15, PyTorch 2.8.0+cu128, NCCL 2.27.3, SGLang 0.5.3.post1, Transformers 4.57.0, DeepEP `1.2.1+73b6ea4` (extension SHA-256 `15796552ab60caa16b3b4c515032c18ebca6925953f83ddbfa015afebd67ed0e`).

## GPU boundary

Only physical GPUs 0--3 were used:

| physical GPU | UUID | model | connectivity |
|---:|---|---|---|
| 0 | `GPU-f217c8a0-1142-20f4-d84b-af29f3a47a0d` | H100 80GB HBM3 | NV18 to peers 1--3 |
| 1 | `GPU-a77f3471-67d4-20b0-9fab-e502d4de5adb` | H100 80GB HBM3 | NV18 to peers 0,2,3 |
| 2 | `GPU-24200107-8a7f-de46-1bc8-b81f8d3af13e` | H100 80GB HBM3 | NV18 to peers 0,1,3 |
| 3 | `GPU-17488c15-2d4c-5d9e-d503-29b0d959a8a8` | H100 80GB HBM3 | NV18 to peers 0--2 |

GPU 4--7 processes were neither used nor stopped. The task-owned burn on 0--3 was stopped for measurement and restarted after the last GPU run.

## True-EP4 path

The executable path is dense TP4 plus routed EP4, DP1. The manual dInfer runner partitions source token rows evenly across the four ranks before routing. Each rank routes its local source rows, DeepEP normal mode dispatches selected branches to the owning rank, the SGLang BF16 fused-MoE runner executes 64 complete local experts, DeepEP performs the reverse combine, and an exact TP-group gather restores source-token order. Shared experts remain local and exact.

This is established by source and runtime evidence, not by the flag alone:

- `num_local_experts = 256 / 4 = 64` in the DeepEP dispatcher;
- contiguous owner mapping `expert_id // 64`;
- four TP/EP ranks initialized in every run;
- per-source rank dispatch counts, received owner rows, owner-local expert IDs, dispatch and reverse-combine CUDA events in every traced layer;
- nonzero remote fraction around 0.75 and mean destination fanout around 3.06--3.08.

The runtime warns that normal-mode DeepEP uses 20 communication SMs. It was held fixed rather than retuned. CUDA graphs are disabled because the active DeepEP mode is normal.

## Strongest baseline

The preceding static sweep established submitted batch 32 and `mini_batch_size=32` as the strongest feasible static EP4 setting: five-run median 6.273 s versus 6.890 s for mini16. This PoC freshly repeated mini32 three times on each of GSM8K and HumanEval; it did not benchmark against a weak default.

## Evidence boundary

- Clean runs alone provide request latency, throughput, NFE and quality.
- Shape, temporal and per-stage traces are observer-heavy diagnostics. Their CUDA events are same-device durations; cross-GPU absolute timestamps are never subtracted.
- Per-layer rank rows are collapsed with the maximum rank for critical-stage attribution, never summed as request latency.
- Component totals divided by clean E2E are explicitly upper-bound attribution, not a claim that the components can jointly be removed.

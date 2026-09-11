# Runtime audit

Status: **TRUE EP4 VERIFIED AFTER A MINIMAL SUBSTRATE PATCH**.

## Pinned sources

- Parent project HEAD: `f0dc8372ec0d1d2e10347c82e630b84430c5dc5f`
- dInfer upstream HEAD: `1ffeb961cd258bede74fcf5ca8a416ae6d57b18f`
- dInfer source checkout: `/home/esjung/dInfer-dllm-ep`
- Isolated environment: `/home/esjung/anaconda3/envs/dinfer-ep-poc`

## Stock path is not dispatch-capable EP4

The official benchmark sets `enable_expert_parallel=True` while launching the
four GPUs as tensor-parallel ranks. Live inspection found:

| Property | Stock four-GPU path |
|---|---|
| Topology | TP4 / DP1 / nominal EP4 |
| Expert ownership | 16 experts/rank: 0–15, 16–31, 32–47, 48–63 |
| Local expert bytes | 201,326,592 bytes/rank |
| vLLM all-to-all manager | `None` |
| Dispatch calls / layer | 0 |
| Combine calls / layer | 0 |
| Observed cross-rank operation | TP reduction after local contribution |

The rank-0 profiler contained 32 `vllm::all_reduce` calls and no broadcast
dispatch. Therefore the flag and sharded weights alone were insufficient proof
of remote token dispatch/combine, and the stock command does not meet this PoC's
definition of true EP4.

## Minimum dispatch-capable substrate

The saved patch makes two source-level corrections:

1. Expert tensor slicing follows `get_ep_group()` instead of the TP group.
2. The global monkeypatch that redirected every vLLM TP all-reduce into the
   process world is removed; it would otherwise corrupt TP1/DP4 independence.

The runner also creates the real four-rank NCCL world before installing the
DP-aware vLLM configuration. Installing it in the reverse order made vLLM
interpret the four processes as external DP and construct virtual ranks in a
world of 16.

The verified result is:

| Property | Patched path |
|---|---|
| Topology | TP1 / DP4 / EP4 |
| FusedMoE topology | TP1 / DP4 / EP4 |
| Expert ownership | exactly 16 contiguous experts/rank |
| One BF16 expert | 12,582,912 bytes (12 MiB) |
| All-to-all backend | vLLM `naive` |
| Runtime manager | `NaiveAll2AllManager` |
| Dispatch / combine | 16 / 16 calls for 16 layers |
| Rank-0 dispatch evidence | 128 NCCL broadcast calls, 1.956 ms aggregate CUDA |
| Rank-0 combine evidence | 16 NCCL all-reduces, 0.446 ms aggregate CUDA |
| Expert evidence | 16 `vllm::inplace_fused_experts` calls |

Thus this is a true remote-token communication path. It is intentionally the
portable vLLM naive manager, not DeepEP or a production throughput result.

## Correctness boundary

- Every route record conserves work: expert counts and rank counts both sum to
  `M × top_k`; all 16 layers are present for every logical iteration.
- Clean versus traced generation matched exactly for 90/90 executions, and all
  three restarts produced identical rank-0 strings.
- The complete expert-assignment vectors also match exactly for 3,792/3,792
  logical records across all three traced restarts.
- The minimum patch does not alter routing or expert arithmetic.

Cross-topology numerical equivalence is weaker. Against a one-GPU reference,
true EP4 has layer-1 relative L2 0.159%, but iterative BF16 differences amplify
to 46.1% at layer 16; final-logit cosine is 0.9776. Stock TP4 also drifts
(50.5% layer-16 relative L2, final-logit cosine 0.9109). This is not evidence
that a future live policy is output-equivalent. The present study therefore
uses the modified path only for route structure, stage timing, and an early
economic kill gate; it makes no production-correctness or quality claim.

Raw per-rank audits and profiler traces are under `runtime_audit/` in the result
root. The exact dInfer delta is preserved in `patches/dinfer_true_ep4.patch`.

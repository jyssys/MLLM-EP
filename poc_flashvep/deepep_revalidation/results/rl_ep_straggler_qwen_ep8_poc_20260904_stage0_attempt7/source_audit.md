# Qwen3-30B-A3B EP8 source/config audit

## Model

- checkpoint: `/home/esjung/.cache/huggingface/hub/models--Qwen--Qwen3-30B-A3B/snapshots/ad44e777bcd18fa416d9da3bd8f70d33ebb85d39`
- architecture: `Qwen3MoeForCausalLM` / `qwen3_moe`
- hidden size / layers: `2048` / `48`
- routed experts / top-k: `128` / `8`
- experts per EP rank: `16` at EP8
- dtype: `torch.bfloat16`
- sparse path: `decoder_sparse_step=1`, `mlp_only_layers=[]` (all 48 decoder layers use routed MoE)

## Runtime proof

The run log shows eight NCCL workers with `world_size=8`, and every backend
proof in `backend_proof/` reports `ep_world_size=8`,
`DeepEPHTAll2AllManager`, `DeepEPHTPrepareAndFinalize`, and `TritonExperts`.
The four driver processes were DP ranks 0–3 and each used TP2, giving
TP2/DP4/EP8/PP1. `CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7` was preserved in
all rank proof files. EPLB and DBO were disabled; placement was linear.

## Measurement hook

The experiment-local child-worker hook wraps the existing
`FusedMoEKernelModularImpl._prepare`, `_fused_experts`, and `_finalize`
calls. It resolves CUDA events once after all waves. `dispatch`, `expert`,
and `combine` are therefore real per-rank CUDA durations; rank values are
never subtracted as cross-device absolute timestamps. The hook records the
local 16-expert assignment histogram and leaves hidden states, top-k routing,
weights, placement, and scheduler decisions unchanged.

## Limitation for gated actions

The raw capture contains exact local expert counts but not token-level expert
IDs or alternate-route outcomes. Capacity-Aware-MoE and EPLB are consequently
source-audited only. `capacity_action_proxy.csv` is a clearly labelled
count-only sensitivity diagnostic, not a correctness-preserving GPU result.

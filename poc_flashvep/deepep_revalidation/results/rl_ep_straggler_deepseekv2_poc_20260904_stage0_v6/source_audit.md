# DeepSeek-V2-Lite source/config audit

## Model configuration

- model type: `deepseek_v2` (`DeepseekV2ForCausalLM`)
- hidden size: `2048`
- decoder layers: `27`
- routed experts: `64`
- top-k per token: `6`
- shared experts: `2`
- MoE frequency: every layer (`moe_layer_freq=1`)
- first dense replacement: layer 0 (`first_k_dense_replace=1`); measured routed layers 1–26
- dtype: `torch.bfloat16`

## vLLM path inspected

Installed vLLM `0.20.0 (run log / installed package)` source at
`/home/esjung/anaconda3/envs/flashvep-poc/lib/python3.12/site-packages/vllm`.
`DeepseekV2DecoderLayer.forward` (deepseek_v2.py:1043+) executes self-attention,
post-attention RMSNorm, then `self.mlp`; routed layers construct
`DeepseekV2MoE`, whose `FusedMoE` uses `top_k=config.num_experts_per_tok`.
The experiment-local hook wraps the existing modular `_prepare`,
`_fused_experts`, and `_finalize` calls with CUDA events and reads
`expert_num_tokens_cpu`; it does not alter routes, weights, placement, or
scheduler behavior.

## Backend proof

All four EP rank proof files report `DeepEPHTPrepareAndFinalize`,
`DeepEPHTAll2AllManager`, `TritonExperts`, `ep_world_size=4`, and
`visible_devices=1,2,3,4`. The runtime metadata records TP2/DP2/EP4,
BF16, DBO off, prefix cache off, and linear placement.

## Capacity/EPLB references

Public references were inspected at the commits in
`capacity_eplb_reference_manifest.json`. Capacity-Aware-MoE's patch applies
capacity-factor token selection; EPLB's `rebalance_experts` packs weighted
experts and can replicate them. Neither was executed because the
preregistered Stage-0 natural-straggler gate failed.

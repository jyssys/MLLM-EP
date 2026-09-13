# Environment and model truth

Status: substrate verified; measurement campaign complete.

## Revisions

- Project worktree base: `92c37786dd3b889ec07e8ba5314d7a31c9d573fb`
- Isolated dInfer base: `6cce0074d7dacf7f91f8cba2df2635761125d344`
- Model: `/home/esjung/models/LLaDA2.0-flash-744c3f8`
- Model config SHA256: `ac35e9dc8313f49b2600da2a8b3ec26c8e21f8e208ab07`
- Driver: 570.211.01
- Accelerator: 4 × NVIDIA H100 80GB HBM3

## Physical/logical mapping

Every task launch sets `CUDA_VISIBLE_DEVICES=4,5,6,7`; therefore task-local CUDA
devices 0/1/2/3 map to physical devices 4/5/6/7.  UUIDs and NVLink topology are
captured in `environment/`.  Launch guards refuse a run if any compute process
already occupies physical GPU 4–7.

## Downloaded model truth

The downloaded revision has 32 transformer layers, hidden size 4096, 32 query
heads, 4 KV heads, BF16 weights, 256 routed experts, top-k 8, one shared expert,
and MoE intermediate size 1024.

## Runtime topology

The retained production-like bridge is dense TP4 plus routed EP4, DP1, PP1.
Each TP/EP rank owns a contiguous shard of 64 routed experts.  Dense activations
are partitioned into source-token rows before DeepEP dispatch; owner ranks run
the fused expert path, DeepEP returns routed outputs, and an exact gather restores
the dense TP representation.  Fresh per-rank route/receive evidence is collected
by the shape/timing trace campaign rather than inferred from `ep_size=4` alone.

## Strong baseline contract

- DeepEP mode: normal/high-throughput
- Fused expert backend: SGLang/Triton/DeepGEMM-compatible checkpoint path
- Temperature: 0
- Threshold decoder: 0.9, with 1.0 one-token-per-iteration effort control
- Prefix KV cache
- CUDA graph: disabled because this manually bridged dynamic path has no validated
  graph capture
- DeepEP maximum source rows: 1024 per EP rank, not 1024 global rows

The previous B32 static baseline (`submitted batch=32`, `mini=32`) is reproduced
before dynamic comparisons.  All B values are subsequently compared at their own
best feasible mini size, plus fixed-mini and matched-global-M controls.

## Discovered benchmark assumptions

Two stock benchmark behaviors invalidated a naive variable-B experiment:

1. config 42 silently overwrote CLI `block_length` with 32;
2. generation extent was floor-aligned to a global constant bucket of 32.

The isolated dInfer worktree removes the override and supports an explicit common
`target_total_length`.  Native behavior is retained as a separate deployment-like
view; matched-extent results are the causal variable-B comparison.

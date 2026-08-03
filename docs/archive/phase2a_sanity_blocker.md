# Phase 2-A Sanity Check Blocker

Date: 2026-06-25

## Command

```bash
PYTHONPATH=. CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --standalone --nproc_per_node=8 scripts/phase2a_sanity_forward.py --use-deepspeed --output outputs/phase2a_sanity_deepspeed.json
```

## Result

The dummy multimodal prefill forward completed on all 8 H100 GPUs without OOM
or NCCL/communication failure.

Output file:

- `outputs/phase2a_sanity_deepspeed.json`

Key observed values:

- world size: 8
- DeepSpeed engine class: `InferenceEngine`
- output logits shape per rank: `[1, 79, 151936]`
- dummy prefill token mix: 64 vision tokens, 15 text tokens
- peak memory per rank: `57.94 GB`

## Blocker

The run does not satisfy the required Phase 2-A EP sanity condition.

The spec requires Qwen3-VL-30B-A3B-Instruct to run as attention-DP plus
8-way MoE expert parallelism, with 128 experts placed sequentially as 16 experts
per GPU. That setup should shard MoE expert weights across ranks. Instead, the
observed peak memory is about `57.94 GB` on every rank, which is consistent with
each rank holding a full model copy rather than an 8-way expert-parallel shard.

DeepSpeed accepted the inference config (`ep_size=8`, `moe_experts=[128]`) and
wrapped the model as `InferenceEngine`, but this did not convert the HF
`Qwen3VLMoeTextSparseMoeBlock` packed expert implementation into real
DeepSpeed-MoE EP dispatch/sharding.

## Decision

Per the Phase 2-A instruction, measurement/profiling did not proceed. The
following items remain untouched:

- `hooks/register_hooks.py` real hook implementation
- `measure/ep_load.py`
- motivation figures
- calibration profiling outputs
- `docs/phase2a_report.md`

## Likely Cause

The downloaded Hugging Face Qwen3-VL-MoE model uses an eager packed-expert MoE
block. DeepSpeed inference wrapping alone does not repartition that custom HF
MoE block into 8-way expert-parallel experts. A real Phase 2-A run needs either:

- a Qwen3-VL-MoE DeepSpeed-MoE adapter that replaces/loads the HF sparse block
  as true EP modules, or
- a checkpoint-loading path that shards the packed expert tensors into
  rank-local expert modules before forward.

No placement, merge, cap, all-to-all intervention, de-RoPE, or accuracy/speed
measurement was attempted.


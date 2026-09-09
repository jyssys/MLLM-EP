# BatchGen paper vs current main

## Versions audited

- OSDI artifact pin: `df221143d0520ea37d2dd35b23f2915ae5f92678`
- Current main: `ca2aaca8a24fa110bb48b86f1876fe25b5cf3267`
- Paper text: OSDI 2026 open-access version

## Paper contract

The paper distinguishes two types of yield point:

1. Intra-forward yields are selected once, statically, per model from compute
   characteristics.
2. Inter-forward yields react dynamically to runtime state.

For an MoE block it evaluates a combined attention+MoE unit, a split
attention/MoE unit, and per-expert coroutines.  It adopts the split unit for
expert batch formation and retains the combined unit for stragglers.  Its
static planner fixes batch sizes, buffer allocations and yield points, and its
cost model profiles a representative layer based on transformer homogeneity.

The paper's overhead table reports GPU-resident combine as zero, hidden-state
checkpointing below 5 microseconds in its evaluated setting, and a much larger
cost when state must be offloaded.  Those values are evidence for its system,
not blindly reused measurements for Qwen3-VL.

Most importantly, vision-language models are future work.  The paper itself
suggests that a yield after the encoder could let the runtime regroup lighter
language stages.  Therefore “add one encoder yield” is explicitly anticipated,
not an adequate successor contribution.

## Current main contract

Current main has expanded substantially, including Kimi-K3-related model and
test scaffolding.  It does not silently support the missing multimodal path:
`batchgen/models/moonshotai/kimi_k3/model.py` defines a hard failure for media
placeholder tokens because the MoonViT tower/projector is not implemented.
The tests assert this failure.  No Qwen3-VL implementation was found.

## What would have been new

A paper-worthy successor would need all of the following:

- actual vision, attention and MoE compatibility objectives that conflict;
- adaptive yield placement or grouping that best static encoder and MoE yields
  cannot recover;
- material direct BCT headroom after checkpoint/memory costs;
- an advantage over independent MLLM stage batching and over BatchGen's
  attention→MoE regrouping.

The fresh GPU oracle fails the headroom and independent-value conditions, so a
native port was not justified.

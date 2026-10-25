# Native architecture and ownership

| Model | Learned parameters | Physical MoE sites | Recurrent sites | Passes | Logical MoE executions |
|---|---:|---:|---:|---:|---:|
| IFM MoE Ouro | 8,780,889,600 | 28 | 28 | 4 | 112 |
| IFM MoE Huginn | 8,783,251,968 | 28 | 12 | 8 | 112 |
| Foil-1 | 553,730,048 | 3 | 1 | 16 | 18 |

These counts come from instantiated native modules and observed forward calls.
No checkpoint or recurrence configuration was substituted.

## IFM models

Both IFM models use width 1,536 and 25 routed experts per physical MoE site.
Each token selects two experts.
The expert intermediate width is 2,432.
The router uses sigmoid scores and normalizes the selected scores.
There are no shared experts in these checkpoints.
The attention geometry has 24 query heads and 6 KV heads, each with width 64.

Ouro repeats all 28 physical layers four times.
Huginn has 8 prelude layers, a 12-layer core, and 8 coda layers.
Its core repeats eight times.
Its native evaluation uses the published deterministic normal-state initialization and input injection.
It is not the dense Huginn checkpoint from earlier studies.

The native loader disables its fused training blocks as its official inference procedure specifies.
It loads the original strict BF16 state without random reinitialization.
FA4 uses the same official attention dispatch function.
The reports retain the separate backend numerical limitation.

## Foil-1

Foil-1 has one recurrent MoE site with 64 experts and top-2 routing.
Its 16 recurrent passes use untied attention modules.
The expert weights and router remain shared at that site.
Its head and tail each contain a nonrecurrent eight-expert MoE block.
Neither outer block enters temporal recurrence statistics.

The width is 1,024.
The attention has 16 heads.
The published FFN width is 3,072.
Each expert uses width 1,536 because the implementation divides by top-k.
The attention scale is `1 / head_dim`, not `1 / sqrt(head_dim)`.
The experiment preserves that published operation.

The 563.2M model-state count needs qualification.
The instantiated model has 553,730,048 learned parameters.
It also has 9,437,184 registered RoPE buffer elements.
The persistent state therefore contains 563,167,232 elements.
This explains the rounded 563.2M count without a checkpoint substitution.

Foil-1 supports a 4,096-token context.
The actual tested prefixes and generation caps remain within that limit.
The experiment does not extend RoPE positions beyond the supported range.

## Single-device ownership

Every worker has one visible CUDA device.
Every learned parameter must reside on `cuda:0` within that worker.
The CUDA mask identifies one authorized physical GPU.
Later workers also compare their own NVML PID mapping with the recorded GPU UUID.
No model uses CPU offload or sharded weights.
The eight workers are independent experiments, not a distributed serving engine.

## Cache behavior

Both official generation paths recompute the entire prefix.
Neither supplies a generation KV cache in this study.
No KV-memory quartering or cached-decode claim is applicable.

C2 stores expert outputs, not attention KV.
Its lookup key contains the physical site, token position, and expert identity.
The cache only connects adjacent recurrent passes inside one full forward.
Every new full-prefix forward clears it.

Native parameter pointers and tensor versions must remain unchanged after each completed run.
The run completion records include this integrity check.

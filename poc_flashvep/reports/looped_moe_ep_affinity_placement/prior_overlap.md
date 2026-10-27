# Prior work and the novelty boundary

## Scope

This study tests recurrent expert affinity on three released checkpoints.
It does not repeat expert skipping, output reuse, cross-loop coalescing, or loop-conditioned replication.
The previous results remain unchanged.

## Primary sources

| Source | Relevant mechanism | Required control in this study |
| --- | --- | --- |
| [Expert Coupling in MoE Pretraining](https://arxiv.org/abs/2610.09372) | Co-activation placement, destination deduplication, and inter-layer token shuffling | These are generic MoE controls, not recurrent novelty. |
| [ExFlow](https://arxiv.org/abs/2401.08383) | Context-coherent execution uses conditional inter-layer expert affinity | Inter-loop predictability alone does not establish a new execution mechanism. |
| [Semantic Parallelism](https://proceedings.iclr.cc/paper_files/paper/2026/hash/f0552f14388d95b19740dee809f5cad1-Abstract-Conference.html) | Co-activation placement and token/request scheduling | Compare placement under the same memory and workload limits. |
| [IFM native source](https://github.com/ifm-ai/xllm-loop) | Native recurrent models, routing, and publication evaluation | Keep the native router, attention, recurrence, and benchmark scorer. |
| [Foil paper](https://arxiv.org/abs/2609.35751) | Tied recurrent experts and untied recurrent attention | Keep the distinct boundary pools and the repeated core pool separate. |

The first source appeared on 2026-10-07.
It directly overlaps co-activation placement and routing prediction.
Its training results do not establish an inference speedup on these checkpoints.
This study does not transfer its reported speedups.

## Prior local experiments

The committed single-GPU study is `looped_moe_headroom/`.
Its commit is `532cd9023ce900272a5403464d54e9da45a9fe72`.
It found no justified expert-skipping or output-reuse prototype.

The committed EP study is `looped_moe_ep_coalescing_replication/`.
Its commit is `a5b37f72245969782a77c93a1b0f0bb700c3494f`.
It measured TRUE EP4 and EP8 on all three checkpoints.
Its finite coalescing search gave 1.000× over its best generic window control.
Its loop-conditioned replication model added less than 1% over matched generic replication.
Those are bounded negative results for the tested mechanisms.
They do not prove that recurrent EP has no optimization opportunities.

The previous IFM experiments used FA4.
The official publication evaluator rejects FA4.
This study built FA3 separately and reproduced full native benchmark anchors.
It does not relabel the earlier engineering probes as publication-quality evaluation.

## Generic execution control

The research EP bridge replicates dense hidden states on every rank.
The exact current router also runs on every rank.
The generic control chooses the current top-1 expert owner as the dispatch source.
It selects an existing copy of the hidden state.
It does not move attention ownership or KV state.
It uses no recurrence index or future routing information.

This control is legal only under the recorded replicated-dense execution contract.
It is not free token migration for an attention-sharded serving engine.
Its measured result must remain separate from a loop-aware optimization claim.

## Placement and payload

The original bridge sends one BF16 activation for each top-2 edge.
Two selected experts on one rank still receive two rows.
Co-placement alone therefore does not deduplicate payloads.
A destination-deduplicating dispatcher needs an additional implementation and correctness test.
The oracle must charge that implementation or identify it as unimplemented.

An expert transition connects routing choices, not necessarily physical state transfers.
The bridge combines expert outputs and restores replicated dense states after every MoE invocation.
It has no direct expert-to-expert state channel between recurrent passes.
Transition affinity can therefore be predictive without reducing the current communication path.

## Evidence labels

Native benchmark scores and completed TRUE EP timing runs are measured evidence.
Co-activation and transition statistics describe recorded router outputs.
Placement models are analytical or trace-based evidence.
An unimplemented dispatcher is not a measured E2E speedup.

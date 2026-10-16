# Prior boundary

This campaign validates a released model and measures native serving.
It does not introduce a scheduling algorithm, KV compression method, or distributed execution engine.
Fixed R4 prevents an adaptive-depth speedup claim.

| Prior | Established direction | Boundary for this campaign |
|---|---|---|
| [Ouro](https://arxiv.org/html/2510.25741v2) | Pretrained recurrent language model and reasoning evaluation | The checkpoint and its published evaluation are the subject, not a new contribution |
| [FlashLoop](https://arxiv.org/abs/2609.29812v2) | Sparse cross-loop updates and compressed KV residuals | No reuse, approximation, or KV residual compression runs here |
| [Continuous Depth Batching](https://arxiv.org/abs/2608.09444v2) | Loop-level scheduling for adaptive computation | Fixed R4 and ordinary native scheduling remain the baseline |
| [vllm-rlt](https://github.com/ThinkFlowLab/vllm-rlt) | Recurrent-loop batching and depth-aware cache management | The campaign does not rename loop-level batching as a new framework |
| [LoopSpec](https://github.com/kaist-flexml-lab/loopspec) | Pipelined self-speculative decoding | No draft, acceptance verifier, or speculative future token runs here |
| [WaveFront](https://arxiv.org/abs/2510.14961) | Parallel sampling for recurrent-depth models | This campaign preserves ordinary autoregressive output dependencies |
| Generic TP and paged KV | Head/weight sharding and cache allocation | TP×DP capacity and throughput gains are existing mechanisms |

The prior audit supports only these mechanism boundaries.
No listed prior is reproduced as a measured speed baseline.
No performance number from another hardware system supplies this campaign's baseline.

A new method requires a measured, removable critical-path or capacity cost.
It must also require a recurrence-specific mechanism beyond ordinary depth or paged KV.
Routing divergence, a 192-slot cache, or imperfect TP scaling alone is insufficient.
Quality, systems evidence, and novelty remain separate verdicts.

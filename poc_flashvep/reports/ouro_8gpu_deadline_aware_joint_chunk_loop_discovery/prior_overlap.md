# Prior overlap

RLT already implements Atomic, Split, and Mixed prefill scheduling.
The pinned code separates phase selection, execution descriptors, recurrent state, and KV ownership.
Loop yielding and mixed-depth packing are not new contributions here.
The source pin is [RLT 4a350344](https://github.com/InferMatrixLab/vllm-rlt/tree/4a350344027960a48384626aa646f1765fda6d4f).

[SLOWeave](https://arxiv.org/html/2609.07883v1) selects the largest feasible prefill chunk before the earliest active decode deadline.
Its model assumes monotone iteration cost.
It also discusses prediction margins, oldest-first prefill selection, and overload.
Our proposed C-only control is an independent RLT adaptation of that principle.
It is not an official port or a reproduction of the paper's performance results.

One RLT recurrent batch does not complete one Ouro output token.
Static R4 can require more recurrent passes and coda/sampling.
The new feasibility analysis must account for that remaining path.
It must test whether legal C/G choices provide useful additional operating regions.
A second scheduling variable alone does not establish novelty.

[Continuous Depth Batching](https://arxiv.org/abs/2608.09444) schedules loop-aware work for depth-adaptive inference.
This study retains fixed model depth.
Adaptive scheduling quantum selection is not adaptive model early exit.

Potential novelty remains unproved.
It requires a causal online benefit beyond strong adaptive C-only and unmodified RLT Mixed.
Staging, fairness, Graph paths, and predictor overhead must not explain the entire gain.
The prior fixed-quantum CONDITIONAL verdicts remain unchanged.

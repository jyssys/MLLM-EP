# Prior boundary

This study asks whether legal dynamic C selection adds value beyond equally informed dynamic G selection.
It does not claim a new model-parallel primitive or a new multiobjective optimization principle.

[SLOWeave](https://arxiv.org/html/2609.07883v1) selects the largest predicted chunk that fits active decode deadlines.
Its guarantee assumes an accurate cost model and feasible decode-only execution.
Its experiments include simulation and iteration-level GPU execution.
Our fixed-budget exposure proxy does not reproduce that complete-next-token policy or its guarantee.
Our workload names are inspired by its regimes. Our complete-response every-gap metric is stricter than its within-request P99 rule.

[P-PAS](https://arxiv.org/html/2608.15171v2) changes the global token budget from current prefill/decode pressure.
Our primary comparison freezes the time budget to isolate the C/G action surface.
We do not claim a P-PAS implementation, reproduction, or measured victory.

The pinned [vLLM-RLT source](https://github.com/InferMatrixLab/vllm-rlt/tree/4a350344027960a48384626aa646f1765fda6d4f) already supplies Split/Mixed execution.
The exact local checkout and G2/G3 patch hashes appear in `model_source_locks.json`.
The public GitHub commit-page fetch returned a cache error during a later reference check.
The verified local checkout, source snapshots, and hashes remain the runtime evidence.
Native scheduler, paged KV, graph execution, and model kernels are reused.
The new code changes legal action scoring and measurement wrappers only.

The parent result at `821b11653695d0a470bd88bfcc997ae1bbcf5bc9` remains NO_GO_INCREMENTAL.
Its identical144/144 plans are starting evidence, not an error that this study must overturn.
Objective diversity, two-dimensional enumeration, and scalarization do not establish novelty.
Novelty requires useful, quality-matched executed C changes beyond tuned single-axis and Mixed controls.

# Prior work and novelty boundary

The current experiment does not claim novelty from enumeration or binary variables.
Its candidate contribution requires measured benefit from the recurrent control axis after strong matched controls.
No such online benefit is established yet.

| Prior | Execution axis | Control input | Co-batching / yield | Evaluation distinction |
| --- | --- | --- | --- | --- |
| [Sarathi-Serve, OSDI 2024](https://www.usenix.org/conference/osdi24/presentation/agrawal) | Token chunks | Chunk/batch scheduling under latency objectives | Prefill/decode co-scheduling | Paper capacity results are not Ouro measurements |
| [Layered Prefill, MLSys 2026](https://proceedings.mlsys.org/paper_files/paper/2026/hash/c0f460c6d63599ea870ba9db63dc96a9-Abstract-Conference.html) | Physical layer groups | TTFT/TBT trade-off | Layer-axis interleaving | Its MoE expert-reload mechanism does not describe dense shared-weight Ouro |
| [FlowPrefill v1](https://arxiv.org/abs/2602.16603v1) | Operator boundaries | Priority and arrival/completion events | Separates preemption granularity from scheduling frequency | Generic fine-grained yielding already exists |
| [P-PAS v2, 2026-10-09](https://arxiv.org/abs/2608.15171v2) | Global token scheduling budget | Current prefill/decode pressure | Adapts maximum batched-token budget | Budget adaptation alone is not a new contribution |
| [SLOWeave](https://arxiv.org/abs/2609.07883) | Token C | Next-token deadline | Deadline-constrained adaptive chunking | An RLT adaptation is not an official reproduction |
| [CDB](https://arxiv.org/abs/2608.09444) and [pinned RLT](https://github.com/InferMatrixLab/vllm-rlt/tree/4a350344027960a48384626aa646f1765fda6d4f) | Recurrent execution and token batches | Loop queues, depth state, runtime policy | Existing Split G1 and Mixed | This PoC retains Static R4, not adaptive model exit |
| Current candidate | Token C and legal recurrent G | Measured service budget and optional past-output feedback | Completion-aware fairness is a separate ablation | Requires incremental natural-serving benefit over equally informed C-only and Mixed |

The external paper pages were checked before this design.
The local pinned RLT scheduler and executor were inspected directly.
The descriptions above concern the cited versions and inspected code.
Optional P-PAS-style or SLOWeave-style controls must disclose any algorithmic differences.
No external paper's speedup is used as an Ouro result.

The full-R4 completion objective distinguishes useful completed prompt work from partial recurrence progress.
That objective is a measurement requirement, not itself proof of a new method.
Staging, graph paths, fairness, cost information, and budget rules must match across the primary contrast.

# Prior overlap and claim boundary

| Primary source | Existing contribution | ELB evidence required beyond it |
| --- | --- | --- |
| [CDB paper](https://arxiv.org/html/2608.09444v2) | Recurrent batch reform, boundary scheduling, asynchronous preparation, and depth-aware KV | A measured reason to keep a batch across multiple static recurrent passes |
| [CDB code](https://github.com/kschwethelm/continuous-depth-batching/tree/abdfae0adbdd12be8c6123ccb34eee98a38afb39) | Local recurrent ready queue, refill accounting, and compatible depth batching | An incremental grouping/reform policy, not another name for refill |
| [Pinned RLT](https://github.com/InferMatrixLab/vllm-rlt/tree/4a350344027960a48384626aa646f1765fda6d4f) | Native Mixed, immutable execution spans, multi-loop runner primitive, CUDA Graphs, and events | Safe grouped-decode scheduler progression plus an advantage over its strong native controls |
| [FlowPrefill](https://github.com/ChiachiHsieh/FlowPrefill) | Event-driven scheduling and separation of scheduling frequency from preemption granularity | A loop-specific benefit beyond generic event-driven D1 selection |
| [SLOWeave](https://arxiv.org/abs/2609.07883) | Deadline-aware adaptive prefill chunk sizing | Decode batch-lease economics, not another prefill deadline controller |
| [Ouro](https://ouro-llm.github.io/) | A pretrained shared recurrent stack | Physical execution benefit. Repeated weights alone do not establish novelty |

The checked CDB code is a research implementation, not the RLT serving engine used in this experiment.
Its local recurrent queue does not constitute cross-GPU request placement.
This study keeps the model depth fixed at four.
It does not mix an adaptive model exit policy with an adaptive scheduling quantum.

RLT already supplies the multi-loop execution primitive.
The new patch adds decode group selection and both scheduler progress updates.
That patch alone is an implementation result, not a new scheduling contribution.
ELB would require a physical KEEP/REFORM crossover and an advantage over tuned fixed D and native Mixed.
Generic priority or event scheduling that matches ELB would invalidate a loop-specific incremental claim.

The earlier prefill C/G verdicts remain unchanged.
The new experiment does not infer success from a larger action space or fewer scheduler calls.

# Baseline and dependency audit

The runtime uses pretrained Ouro-1.4B Base, BF16, and Static R4.
The original model weights and gate remain unchanged.
The experiment disables model early exit with threshold1 and maximum depth4.
Every prompt and forwarded output token traverses all four recurrent passes.

## Source findings

| Mechanism | Pinned source | Existing behavior |
| --- | --- | --- |
| Phase ordering | `core/scheduling_policy.py:59` | Refill selects prelude, coda, recurrent, and prompt work |
| Mixed packing | `core/scheduling_policy.py:159` | Native prompt/decode recurrence co-packing. Not new ELB functionality |
| Selection | `core/scheduler.py:599` | Sequence, token, capacity, and readiness constraints precede execution |
| Execution descriptors | `request.py:137`, `worker/executor.py:16` | Immutable spans carry position, depth, and borrowed Request ownership |
| Group primitive | `worker/executor.py:167` | ModelRuntime already iterates supplied depth-specific metadata |
| Native progression | `core/scheduler.py:836,960` upstream | Sync and async callbacks advance one loop. Both need adaptation |
| GPU preparation | `worker/model_runner.py:813,933` | Host metadata precedes GPU upload and native producer waits |
| Producer ordering | `worker/model_runner.py:1147,1161` | The consumer stream waits for every selected request producer event |
| Graph lifetime | `worker/cuda_graph.py:105` | Native per-call graph metadata, input preparation, and reuse events remain active |
| Async state | `worker/async_state.py:1` | Event-owned routing storage and stable per-request slots |
| CODA | `engine/stage_engine.py:724` | Native bounded output placeholders, sampling, readback, and stop handling |

Line numbers refer to the pinned upstream source unless explicitly marked as the patch.
The patch modifies config, scheduler progression, and execution descriptor construction.
It does not change model math, attention kernels, KV layout, graphs, transport, or sampling.

## Static policy controls

Native refill D1 and native Mixed D1 retain their original selection logic.
Grouped decode changes only pure RECURRENT batches.
Mixed batches retain D1, even when configured decode D is larger.
The common legal group is the minimum remaining depth across every selected member.
PRELUDE and CODA never inherit a recurrent group.
The patch rejects adaptive-gate grouped execution.

Native async scheduling requires `ouro_delayed`, `random_lookahead`, or trace mode at construction.
The fixed-R4 async lane selects `ouro_delayed` with threshold1.
No early-exit signal can change its depth.
The first smoke used the unsupported constructor combination. Its failure remains preserved.

## Publication and lifetime

Host `loops_done` describes issued work, not completed KV writes.
The runner retains its native per-request completion event after a grouped submission.
The next recurrent or CODA consumer waits for that event.
Intermediate depth metadata uses separate KV namespaces.
Static R4 has 24 physical layers and 96 logical depth/layer namespaces.
No missing intermediate KV write is replaced with a final-depth alias.

The scheduler advances a selected row by its actual group length in both update paths.
Sampling still occurs only in CODA after the full recurrent path.
Cancellation and request-ID reuse retain native object-identity checks.
Graph and input storage retain native bounded submission ownership.
These source properties require GPU correctness tests. They are not performance evidence.

## Prior boundary

[CDB](https://arxiv.org/html/2608.09444v2) already reforms recurrent batches and manages depth-indexed KV.
[RLT](https://github.com/InferMatrixLab/vllm-rlt/tree/4a350344027960a48384626aa646f1765fda6d4f) already supports native Mixed and asynchronous submission.
[FlowPrefill](https://github.com/ChiachiHsieh/FlowPrefill) already separates scheduling frequency from preemption boundaries.
[SLOWeave](https://arxiv.org/abs/2609.07883) adapts prefill chunks using decode deadlines.
This study tests decode lease length against the benefit of newly eligible batch members.
Grouping alone or generic event scheduling alone does not establish a new method.
An incremental ELB claim requires physical crossover and an advantage over tuned fixed D and native Mixed.

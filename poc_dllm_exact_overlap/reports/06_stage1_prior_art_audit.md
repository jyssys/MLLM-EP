# Stage 1 adversarial prior-art audit

| work | exact relevant mechanism | collision assessment |
|---|---|---|
| [DeepEP](https://github.com/deepseek-ai/DeepEP) | async dispatch/combine events permit independent compute before wait | the primitive used by this PoC; removing a wait is not novel |
| [SGLang TBO operation graph](https://raw.githubusercontent.com/sgl-project/sglang/main/python/sglang/srt/batch_overlap/operations_strategy.py) | splits dispatch/combine and yields between microbatches; in one path places shared expert between combine halves | direct adjacency for complete-wave pipeline and shared overlap |
| [COMET, MLSys 2025](https://proceedings.mlsys.org/paper_files/paper/2025/hash/e27ea0cd50b798ff8942caf9203f0992-Abstract-Conference.html) | fine-grained computation/communication overlap for MoE | direct collision with generic “comm + GEMM overlap” |
| [StreamEP](https://github.com/evolutionaryscale/StreamEP) | dispatch releases expert-major tiles as they arrive; combine sends per token as compute completes | direct collision with completed-tile dispatch/expert/combine streaming |
| [Fine-grained tile signaling](https://arxiv.org/abs/2607.19539) | persistent producer/consumer overlaps expert tiles and return all-to-all | direct collision with same-wave early combine |
| [X-Stage](https://arxiv.org/abs/2607.23264) | schedules post-issue communication progress; interleaves expert waves | direct adjacency/collision with expert-wave pipeline |
| [Analytical resource management for overlap](https://arxiv.org/abs/2609.07536) | chooses resource split using routed tiles and occupancy over COMET | very recent collision with resource-adaptive generic overlap |

The strongest measured Stage 1 result is therefore both too small at the proper
request boundary and crowded by exact mechanisms that are more fine-grained than
this PoC. A successor would need a dLLM-specific dependency/resource fact that
outperforms generic TBO/streaming; Stage 2 tests that possibility.

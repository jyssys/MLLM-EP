# Prior overlap

This campaign tests a physical hypothesis. It does not introduce a new looped architecture, sparse method, or speculative decoder.

| Prior | Relevant mechanism | Boundary for this experiment |
|---|---|---|
| [Ouro](https://huggingface.co/ByteDance/Ouro-2.6B-Thinking) | A pretrained looped model reuses a shared decoder stack. | We measure its loaded implementation. We do not claim weight sharing or recurrent depth as new. |
| [FlashLoop](https://arxiv.org/abs/2609.29812) | Token-sparse updates, sparse attention, and KV residual quantization reduce cross-loop redundancy. | This experiment preserves dense execution, BF16, and independent loop KV slots. Those sparse and compression mechanisms are outside scope. |
| [LoopSpec](https://arxiv.org/abs/2609.17184) | Intermediate recurrent states propose tokens. Pipelined verification preserves decoding correctness. | We force the same token trajectory for hardware controls. We do not implement draft or verification schedules. |
| [WaveFront](https://arxiv.org/abs/2609.23033) | Mixed-depth token states share recurrent calls in self-speculative decoding. | Its parallelism includes speculation and correction. Independent execution of dependent loops is not an exact free optimization here. |
| [Parallel Loop Transformer](https://arxiv.org/abs/2510.24824) | A new architecture changes cross-loop dependencies and shares global KV with gated local attention. | It is not an execution-only replacement for the unchanged Ouro checkpoint. |
| [Parallel Track Transformers](https://arxiv.org/abs/2602.07306) | A new architecture reduces cross-device synchronization dependencies. | Generic TP synchronization cost does not establish a recurrence-specific mechanism. Architecture changes require their own quality evaluation. |
| [vllm-rlt](https://github.com/ThinkFlowLab/vllm-rlt/tree/9d601ceaf9af63121c742df5f584467e23630a0b) | A separate loop-aware engine supports depth-aware batching and paged KV execution. | It is not the native vLLM installation used here. Its scheduler features do not prove a missing same-token TP4 latency optimization. |
| [Megatron tensor parallelism](https://arxiv.org/abs/1909.08053) | Column and row partitioning introduce ordinary per-layer reductions. | More executed layers naturally produce more reductions. This count is not a new loop-specific bottleneck. |

The vllm-rlt audit pins upstream commit `9d601ceaf9af63121c742df5f584467e23630a0b`.
Its README distinguishes native and experimental model paths. This experiment does not assume feature parity with that engine.

The decisive control is native Ouro against the same-storage explicit tape.
Both genuine Qwen checkpoints provide context. Their different depth, GQA, vocabulary, and kernels prevent causal attribution from raw speed alone.

No prior speedup number enters our matched baseline.
No absence of a feature in one engine establishes academic novelty.
If the matched GPU schedule eliminates the apparent recurrence penalty, this experiment closes that proposed mechanism.

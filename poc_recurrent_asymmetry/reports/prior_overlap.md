# Prior overlap

The source check date is 2026-10-17.
The installed code and pinned checkpoint determine this study's execution semantics.
Paper results do not substitute for measurements on the assigned GPUs.

| Primary source | Relevant boundary | This study's result |
|---|---|---|
| [Ouro](https://arxiv.org/abs/2510.25741), v6 | Pretrained tied recurrent computation | Retains all four passes and 96 depth-indexed KV slots |
| [Ring Attention](https://arxiv.org/abs/2310.01889), v4 | Exact sequence-distributed attention and communication overlap | The tested exact softmax merge is generic CP, not new recurrence-specific attention |
| [TeraPipe](https://proceedings.mlr.press/v139/li21y.html) | Token-level causal pipelining in training | Causal chunk concurrency is prior art. Training speedups do not establish serving gains |
| [Continuous Depth Batching](https://arxiv.org/abs/2608.09444), v2 | Cross-request batching between variable-depth loop steps | This study keeps fixed R4 and does not implement dynamic-depth batching |
| [Enabling Dynamic Computation in Looped LMs](https://arxiv.org/abs/2610.09013) | Early-exit execution and cache-policy interaction | The exact baseline retains independent per-depth caches and does not infer skipped work from exit scores |
| [Looped Latent Attention](https://arxiv.org/abs/2607.15456), v2 | Trained cache compression and reconstruction across loop depth | Cache approximation, distillation, and shared-depth compression remain outside this study |
| [WaveFront Decoding](https://arxiv.org/abs/2609.23033), v2 | Self-speculative decoding with verification | This study analyzes exact known-token prefill, without draft or correction stages |

The checkpoint config source is [the pinned model configuration](https://huggingface.co/ByteDance/Ouro-1.4B-Thinking/blob/3aaa2224253a92ca45cf2e3d427c360e1ef9c93d/config.json).
The local immutable snapshot supplies its measured hash and actual values.
The browser could not retrieve that pinned config page during the final check.
Local code inspection therefore remains the authority for those fields.

Neither weight sharing nor a two-dimensional dependency drawing establishes new parallelism by itself.
The exact DAG maps to an ordinary unrolled causal transformer with pass-boundary normalization.
The tested CP operator likewise uses established distributed softmax semantics.
No tested mechanism supplies an additional loop-specific physical latency gain.

The prior PARPP approximation result remains closed.
No stale, delta, or predictive activation policy runs in this study.
The study does not reframe generic cache concatenation or CUDA Graph improvements as a new recurrent architecture.

# Architecture contract

## Scope

Use only physical GPUs 4, 5, 6, and 7.
Logical ranks follow the visible-device order.
The full mask maps logical ranks 0–3 to physical GPUs 4–7.
TP1 uses GPU 4. TP2 uses GPUs 4–5. TP4 uses all four GPUs.
Unused assigned GPUs retain their own burner workers.
No process on another physical GPU is stopped.

The authoritative contract is the unmodified `authoritative_spec.md` snapshot.
This campaign characterizes native execution before any placement method.
No training, quantization, cache compression, speculative decoding, or stale KV enters the primary path.

## Ouro

Checkpoint: `ByteDance/Ouro-2.6B-Thinking`.
Revision: `f1edd81e7ac41355db670500ceaf204e0f73af68`.
The public card says 24 layers. The actual released weights and config contain 48 independent physical layers.
The same 48 modules execute four times. Logical attention depth is 192.
There is no separate prelude/core/coda stack beyond embedding, shared stack, normalization, and output head.
Each layer has 16 Q heads and 16 KV heads of dimension 128.
The primary uses BF16 and fixed four-step execution.
Adaptive exit is not compared with this fixed-depth baseline.

The installed native vLLM version is `0.26.0+cu129`.
It uses PyTorch `2.11.0+cu129` and Transformers `5.16.1`.
The native model implementation is `vllm.model_executor.models.ouro`.
Its standard column/row projections shard weights and heads.
The primary uses FlashAttention3, native compilation, and native CUDA Graphs.
The Python worker extension reads ownership and memory outside request timing.
It does not replace model execution or require pickle RPC.

The config permits 65,536 positions. The Thinking card describes 32K SFT.
This campaign limits total request positions to 32K.
The largest prompt contains 32,704 tokens, followed by 64 output tokens.
That point is a 32K-capacity request, not a 32,768-token prompt.

## Huginn

Checkpoint: `tomg-group-umd/huginn-0125`.
Revision: `bb6621b65e90b6a4b9b29ef88dc83866d450470c`.
Physical blocks are two prelude, four shared recurrent, and two coda blocks.
The checkpoint default recurrence is 32. The primary bounded reference uses the documented 16-step example.
Its logical depth is 68, not 192. Cross-model absolute timings are not equal-compute comparisons.
The model has 55 Q heads and 55 KV heads of dimension 96.
Its original trained block size is 4096.
The longest tested performance prompt is 4032 tokens plus 64 output tokens.

The official plugin is experimental and documents TP1.
Its attention code floor-divides 55 heads, without a padding or unequal-partition implementation.
Native vLLM's projection/config checks require exact head divisibility.
The plugin also imports an attention API absent from installed vLLM 0.26.
No new executor or 55-head remapping is implemented.
The native Hugging Face model runs as the real second pretrained model.
That reference is not an optimized physical multi-GPU serving result.

Huginn retains its official state initialization with `init_scale=1`.
Each request resets seed 41015. Fixed recurrence removes the random depth sampler.
The official `HuginnDynamicCache` remains mandatory.
Any `compress-s16` probe is a separate, semantics-changing secondary policy.

## Claim limits

A blocked Huginn TP path permits only one-family physical multi-GPU characterization.
It cannot satisfy the two-family generality gate.
An Ouro1.4B fallback would not provide an independent family.
The existing xLLM release would require another engine/cache integration.
We do not implement that integration to manufacture model coverage.

Clean request wall, kernel service, and partial critical-path bounds remain separate.
Native TP collectives are measured where supported. They are not MoE EP dispatch/combine.
TP2×DP2 or TP1×DP4 throughput requires actual concurrent replicas before any measured claim.
Dependent recurrent steps are never counted as independent pipeline microbatches.

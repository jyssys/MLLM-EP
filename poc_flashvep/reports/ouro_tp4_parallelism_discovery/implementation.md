# Execution and ownership

| Arm | Dense weight ownership | Full KV ownership | Execution unit |
| --- | --- | --- | --- |
| Layer PP4 | Six physical layers per GPU, auxiliary weights replicated | 24 slots per GPU across four loops, all 16 heads | One request, six layers |
| Loop PP4 | All 24 physical layers on every GPU | 24 slots per GPU for one loop, all 16 heads | One request, one complete loop |
| Custom TP4 | QKV/head and SwiGLU shards, auxiliary weights replicated | All 96 slots per GPU, four heads per slot | Four ranks jointly execute one request |
| Native vLLM TP4 | Native TP layers and vocabulary shards | Native paged cache, 96 slots, four local heads | Engine-managed |

Layer PP4 visits GPU4→5→6→7 four times. It needs 15 hidden-state handoffs per full forward.
Loop PP4 visits that path once. It needs three hidden-state handoffs.
Both return the selected scalar token from GPU7 to GPU4 before the next token.
Both preserve the original loop-end norm and gate evaluation.
Neither custom arm implements adaptive exit.

## True TP4

`tp_model.py` reuses the official HF decoder algebra.
It replaces only projection modules and cache storage.
Each Q/K/V weight shard has shape `[512,2048]`.
Each attention-output weight shard has shape `[2048,512]`.
Each gate/up weight shard has shape `[1408,2048]`.
Each MLP-down weight shard has shape `[2048,1408]`.
Attention-output and MLP-down use actual NCCL SUM All-Reduce.
Norms, residuals, embedding, output head, and the unused fixed-depth gate remain replicated.
The custom TP model therefore uses more auxiliary memory than vocabulary-sharded native vLLM.

Weights, KV, and persistent hidden tensors use BF16.
The final diagnostic variant retains GEMM accumulators and row reductions in FP32 before a BF16 cast.
Its collective operands are FP32, not BF16. This is a disclosed implementation difference.
The earlier BF16-reduction variant remains in the execution logs and correctness results.

Each physical layer retains its original parameters across all four loops.
KV index is `24 * loop + physical_layer`.
Each rank owns one contiguous four-head slice at every index.
Prefill uses the entire prompt. Decode uses the exact current causal key length.
Graph scratch never exposes future KV positions to attention.
Owner-local cache staging and graph-input copies remain inside the execution path.
The implementation does not claim a KV-update optimization.

## Common multiprocess PP control

`mp_pp.py` creates one GPU-affine process and one FIFO queue per rank.
CPU queues carry request ID, step, route visit, tensor kind, and readiness metadata.
Startup exchanges persistent CUDA IPC buffer handles.
The receiver copies the hidden tensor directly from the previous GPU into a reusable local buffer.
Serving messages do not carry a hidden tensor through host memory.
Each worker waits for its own CUDA completion event before it publishes the buffer.
The receiving worker can then use the published tensor safely.
No global barrier occurs between stages or loops.
Startup, finalization, and IPC lifetime management use a Gloo group.
Gloo is not a model-compute collective in these PP paths.

The common graph path prepares position-only RoPE constants before admission.
Both placements use the same BF16 decoder, exact-length SDPA, scratch policy, and local publication rule.
Each request owns a separate KV namespace and separate peer export buffers.
No stage merges requests into one tensor batch.
The final scalar argmax still requires host control.
Per-boundary event waits, queues, and IPC copies remain. This is not an optimized production engine.

## Batching boundary

The custom TP serving prototype uses batch-one round-robin admission.
It does not merge requests or simulate PP overlap.
The qualification guard disables its serving benchmark because full C2 qualification failed.
Native vLLM uses different kernels, paged KV, vocabulary sharding, and scheduler code.
The external run uses eager numerical probes only. It is not an optimized serving result.

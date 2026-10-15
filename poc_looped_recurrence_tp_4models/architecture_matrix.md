# Four-family architecture matrix

| Property | Native Ouro | Tied-Unrolled Ouro | Qwen3-1.7B | Qwen2.5-3B |
|---|---|---|---|---|
| Checkpoint | ByteDance/Ouro-2.6B-Thinking | Same Ouro checkpoint | Qwen/Qwen3-1.7B | Qwen/Qwen2.5-3B |
| Revision | f1edd81e7ac41355db670500ceaf204e0f73af68 | Same | 70d244cc86ccca08cf5af4e1e306ecf908b1ad5e | 3aab1f1954e9cc14eb9509a215f9e5ca08227a9b |
| Physical layers | 48 | Same 48 shared modules | 28 | 36 |
| Logical layer executions | 48R, R=1/2/4 | Same 48R | 28 | 36 |
| Hidden size | 2048 | Same | 2048 | 2048 |
| FFN width | 5632 | Same | 6144 | 11008 |
| Global Q heads | 16 | Same | 16 | 16 |
| Global KV heads | 16, MHA | Same | 8, GQA | 2, GQA |
| Head dimension | 128 | Same | 128 | 128 |
| Q/K head normalization | No | Same | RMSNorm on Q and K | No |
| QKV projection bias | No | Same | No | Yes |
| Vocabulary | 49152 | Same | 151936 | 151936 |
| Logical KV slots | 48R | Same independent 48R | 28 | 36 |
| Runtime context cap | 32768 | Same | 32768, below checkpoint 40960 | 32768 |
| Runtime | Native vLLM V2 | Same process-local explicit forward | Same vLLM V2 | Same vLLM V2 |
| Precision | BF16 | BF16 | BF16 | BF16 |
| License metadata | Apache-2.0 | Same | Apache-2.0 | qwen-research |
| Causal role | Main target | Same-math execution control | External pretrained AR context | External pretrained AR context |

The pinned configuration files define these dimensions.
The actual loaded module inspection checks heads, physical layers, parameter storage, and cache slots.
See `results/source_and_model_audit.json` and `results/physical_ownership.csv`.

## Physical partition

| TP | Physical GPUs | Logical CUDA ranks | Ouro local Q/KV heads | Qwen3 local Q/KV heads | Qwen2.5 local Q/KV heads |
|---|---|---|---|---|---|
| 1 | 4 | 0 | 16 / 16 | 16 / 8 | 16 / 2 |
| 2 | 4,5 | 0,1 | 8 / 8 | 8 / 4 | 8 / 1 |
| 4 | 4,5,6,7 | 0,1,2,3 | 4 / 4 | 4 / 2 | 4 / 1 |

At Qwen2.5 TP4, ranks 0–1 contain the same K/V projection shard. Ranks 2–3 contain the other K/V projection shard.
Actual K/V weight hashes establish replication. The global model still has two distinct KV heads.
Its total physical KV storage across TP4 is twice the unreplicated logical KV requirement.
This generic GQA partition differs from Ouro MHA. It is not a recurrence effect.

## Memory definitions

`unique_weight_bytes` sums unique loaded parameter storages within each rank.
The tied control shares physical layer storages across passes. It does not materialize 48R separate weight copies.
`KV_pool_bytes` measures actual preallocated cache storage. The HBM utilization policy sizes this pool.
`logical_KV_bytes_per_rank` counts written positions for the request and the actual local KV heads.
Peak allocated HBM includes weights, the KV pool, graph memory, and transient tensors.
These three memory quantities are not interchangeable.

## Confounds

The Qwen checkpoints do not depth-match Ouro R4.
Their GQA, FFN width, output head, vocabulary, pretrained task behavior, and operator shapes differ.
Qwen3 also normalizes Q and K heads. Qwen2.5 uses a QKV projection bias.
Equal sequence lengths do not remove those differences.
The primary causal control is native versus tied Ouro at fixed TP and R.

Ouro R1 and R2 alter the effective network. Their faster execution does not preserve pretrained R4 quality by assumption.
TP8 is unavailable. The campaign neither executes nor projects TP8 results.

Primary config sources:
[Ouro](https://huggingface.co/ByteDance/Ouro-2.6B-Thinking/blob/f1edd81e7ac41355db670500ceaf204e0f73af68/config.json),
[Qwen3](https://huggingface.co/Qwen/Qwen3-1.7B/blob/70d244cc86ccca08cf5af4e1e306ecf908b1ad5e/config.json),
[Qwen2.5](https://huggingface.co/Qwen/Qwen2.5-3B/blob/3aab1f1954e9cc14eb9509a215f9e5ca08227a9b/config.json).

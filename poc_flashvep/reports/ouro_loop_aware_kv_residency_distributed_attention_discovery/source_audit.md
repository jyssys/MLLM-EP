# Source audit

This study uses the unmodified RLT checkout at `4a350344027960a48384626aa646f1765fda6d4f`.
The checkout is `/home/esjung/external/ouro-kv-residency-rlt`.
The worker rejects any tracked source diff in that checkout.
Experimental wrappers reside only in `poc_flashvep/ouro_kv_residency/`.

## Model and cache

The Base checkpoint revision is `574fa66cb8bf5abdc979642d01cf2b79b16bfab1`.
The model contains 24 shared physical layers and four recurrent passes.
Each pass uses independent KV state for every physical layer.
The model uses 16 query heads, 16 KV heads, and head dimension 128.
All layers use full causal attention.
The configured position limit is 65,536.
The experiment keeps BF16 weights and KV values.

| Source at the pinned checkout | Finding |
|---|---|
| `vllm_rlt/models/ouro.py:156` | `OuroAttention` computes Q/K/V, applies RoPE, writes KV, and calls cached attention. |
| `vllm_rlt/core/kv_cache_manager.py:80` | One physical page pool serves all requests and recurrence depths. |
| `vllm_rlt/core/kv_cache_manager.py:143` | Storage axes are block, physical layer, token within page, KV head, head dimension. |
| `vllm_rlt/core/kv_cache_manager.py:710` | Attention receives one physical layer view and depth-specific page tables. |
| `vllm_rlt/kernels/flash_attention.py:98` | Decode calls optimized paged FA4 with one query and `num_splits=1`. |
| `vllm_rlt/worker/cuda_graph.py` | Captured attention consumes the same cache through device metadata. |

The physical pool contains every recurrence depth.
Its 54 GiB allocation must not be multiplied by four again.
The native cache stays in HBM.
There is no native per-token CPU KV offload to remove.
Static R4 has no skipped depths and needs no last-exited repair copy.

## Previous evidence

The previous memory-asymmetry report remains unchanged.
Its source is `/home/esjung/MLLM-EP-ouro-request-pp/poc_recurrent_asymmetry/reports/profiling.md`.
That report measured a 7.35% attention interval in a separate eager Thinking-model lane at 16K.
Its denominator was an 89.089 ms forward.
The same report identified growing-cache `torch.cat` traffic.
RLT already replaces that append path with paged writes.
Neither old number is a new Base-model native measurement.

The previous ELB study remains unchanged at commit `399c87bc69eaa4fbee5ee2815a9e2aeda0c95d33`.
It did not establish an incremental serving benefit over native Async/Mixed.
The present study does not repeat its decode-grouping policy.

## Supported execution scope

The pinned RLT Ouro model uses unsharded `nn.Linear` operators.
Its model path does not implement tensor-parallel parameter sharding or NCCL collectives.
The matched TP2/4/8 baseline is therefore `UNSUPPORTED_TP_BASELINE_IN_PINNED_RLT`.
This limitation does not mean that other vLLM versions cannot run Ouro with TP.
This study does not construct another serving engine to fill that gap.

Distributed experiments use separate processes and CUDA contexts.
They shard context positions for a single physical layer and recurrence depth.
They do not assign recurrent model loops to different GPUs.

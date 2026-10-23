# Reproducible environment

| Component | Actual pinned configuration |
| --- | --- |
| Model | ByteDance/Ouro-1.4B Base |
| Model/tokenizer revision | `574fa66cb8bf5abdc979642d01cf2b79b16bfab1` |
| RLT revision | `4a350344027960a48384626aa646f1765fda6d4f` |
| Runtime worktree | `/home/esjung/external/ouro-elb-rlt` |
| GPU Python | `/home/esjung/.venvs/ouro-chunk-loop-prefill/bin/python` |
| Python |3.14.6 |
| Torch |2.11.0+cu129 |
| Transformers |5.16.1 |
| Attention |FlashAttention4, flash-attn-4 4.0.0b30 |
| Triton |3.6.0 |
| NumPy |2.4.6 |
| Hardware |Eight physical NVIDIA H10080GB HBM3 devices |
| Model execution |24 shared physical layers, Static R4, BF16 |
| Logical cache namespaces |24 layers×4 depths=96 |
| Native resource budget |18,432 shared physical KV blocks, block size16 |
| Prompt quantum |C512, native split Gp1. Mixed retains native G1 co-packing |
| Token budget |2,048 batched tokens |
| Decode group |D1/D2/D4. D3 is a numerical diagnostic |
| Graphs |Native enabled, maximum graph batch equals configured N, maximum32 graph entries |
| Scheduling |Refill or native Mixed. Preemption and speculation disabled |
| Model exit |Threshold1, min/max depth4. No adaptive model exit |

Native async construction uses `ouro_delayed` only because the engine requires a supported async exit-mode name.
Threshold1 disables early exit, so every token still executes R4.
The synchronous lane uses the native `ouro` mode.
Neither lane trains or replaces the original gate.

The shared KV pool occupies27GiB for K and27GiB for V:54GiB total.
That pool already includes all four depth planes. It is not54GiB per depth.
The loaded model reports1,434,652,673 parameters and2,869,305,346 unique weight-storage bytes.
These are resident-storage facts, not HBM traffic.
The measured timing grid's PyTorch peak allocated range is56.913–58.123GiB.
Actual byte values remain in each timing record.
NVML memory readings are sampled telemetry, not an exact allocation peak.
Weight size, resident allocation, and actual HBM traffic are different quantities.
This study does not infer HBM traffic from parameter size.

P8192/N16 exceeds the common KV reservation budget and is omitted.
The experiment does not change cache capacity or evict selected rows to manufacture that condition.
All tested prompt-plus-output positions remain inside the actual model context limit.
The loaded RLT model config reports65,536 supported positions, hidden width2,048, and16 Q/KV heads.

Exact source hashes, patch contents, physical UUIDs, and per-attempt provenance remain in the JSON locks and raw identities.
The historical prefill worktree and the paused native-vLLM Loop PP project remain unchanged.

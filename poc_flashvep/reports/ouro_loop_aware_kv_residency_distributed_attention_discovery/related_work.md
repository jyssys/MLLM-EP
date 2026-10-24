# Related work and scope

The sources below support mechanism descriptions, not reproduced speedups.
The study does not reuse a paper's accuracy claim as a byte-exact KV guarantee.

| Primary source | Mechanism | Changes original KV or computation? | Training | Relation to this study |
|---|---|---|---|---|
| [FlashLoop](https://arxiv.org/html/2609.29812v2), [public code](https://github.com/Superone77/FlashLoop) | Sparse token updates, sparse attention updates, and quantized cross-loop KV residuals | Yes. Reconstructed KV and omitted updates differ from dense BF16 R4. | Training-free | Excluded. Similar activations do not authorize exact KV aliasing. |
| [Hybrid Latent Attention](https://arxiv.org/html/2610.07940v1) | Learned latent KV for older tokens and an exact recent window | Yes. Older tokens use latent scores and values. | Additional maps require uptraining. | Excluded. Frozen original weights do not imply training-free integration. |
| [Ring Attention](https://arxiv.org/abs/2310.01889), [public code](https://github.com/haoliuhl/ringattention) | Context blocks move through a ring with blockwise attention. | Exact attention formulation with floating-point order differences | No new model training is required for the distribution mechanism. | Generic CP prior. Context sharding alone is not a loop-specific contribution. |
| [FlashAttention](https://github.com/Dao-AILab/flash-attention) | IO-aware exact attention and stable softmax statistics | Numerical reduction order can differ. | No | Reused local FA4 kernel. No new CUDA kernel. |
| [Flash-Decoding](https://crfm.stanford.edu/2023/10/12/flashdecoding.html) | Sequence-parallel local attention followed by logsumexp merge | Exact in real arithmetic with floating-point order differences | No | Strong generic single-GPU control. Batch-expanded FA4 reuses this principle. |
| [vLLM-RLT](https://github.com/InferMatrixLab/vllm-rlt/tree/4a350344027960a48384626aa646f1765fda6d4f) | Paged depth-indexed KV, CUDA Graphs, Async execution, and Mixed scheduling | The registered Static R4 path retains four independent KV planes. | No | Strongest matched single-GPU baseline. |
| [Looped Latent Attention](https://arxiv.org/abs/2607.15456) | SVD-initialized cross-loop codec with distillation | Yes. Reconstructed KV is not the original BF16 tensor. | Post-training | Excluded. Exact storage accounting does not imply exact KV values. |
| [ResidualQuant](https://arxiv.org/abs/2610.10381) | Final-loop reference with low-bit residuals, scaling, and rotations | Yes | Quantized representation | Excluded. No INT2 or mixed-precision KV in this study. |
| [MELT](https://arxiv.org/abs/2605.07721) | One learned gated cache shared across loops | Yes | Transition training and attention-aligned distillation | Excluded. One learned cache is not four original KV namespaces. |

## Distinct claims

Exact BF16 context sharding can reduce per-rank KV ownership.
It does not reduce aggregate valid KV payload.
Distributed softmax merge is an established exact formulation in real arithmetic.
Its communication and floating-point error require measurement.

A generic physical-layer-major layout is an allocator control.
It is not a new recurrent algorithm.
Deterministic loop order can identify the next KV namespace.
That fact only helps prefetch when an actual cold tier exists.
Copying an already resident HBM namespace introduces extra traffic.

The proposed name LAKRA identifies a research question only.
A new contribution requires improvement over the identical depth-agnostic operator and allocator.
Capacity scaling alone does not satisfy that requirement.

## Generic local-parallelism control

Pinned FA4 rejects its internal SplitKV option on SM90.
That limitation does not make sequence-parallel attention a new idea.
The study therefore adds a local generic control after the initial CP measurements.
It exposes independent key segments as additional FA4 batch rows.
One optimized FA4 call computes their partial outputs.
A stable logsumexp merge restores the full attention result.
All original BF16 K/V values remain unchanged.
This control tests whether distributed gains reflect missing local sequence parallelism.
It is not a new loop-specific contribution.

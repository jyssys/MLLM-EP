# Exact dependency contract

Every recurrent pass consumes the previous pass's hidden state.
Every physical layer consumes the previous layer's output.
Attention at token position t uses only initialized KV positions 0 through t.
Each logical depth and physical layer has a distinct KV namespace.
Shared weights do not make these KV values identical.

The operator fixture contains Q after original RoPE.
It contains original BF16 K and V after the current token's cache write.
Its key range ends at the query's causal position.
The sharded operator divides this visible range into disjoint subsets.
It neither predicts hidden states nor reads future KV.

Each rank computes normalized local output and FP32 logsumexp.
For rank j, let z_j denote its logsumexp and a_j its normalized output.
The merged output is the sum of `softmax(z)_j * a_j` across ranks.
This equals the original attention in real arithmetic.
BF16 rounding and reduction order can change the numerical result.
The prospective contract sets the allowed error before GPU operator tests.

The current operator has one query per request.
All retained keys are causally visible.
Local `causal=False` therefore retains exactly the original valid key set.
It does not mean that the model uses noncausal attention.
This special case does not validate multi-query prefill masks.

Each rank owns context shards from every sampled recurrence depth.
No rank owns a recurrent model stage.
The study does not create loop-level Pipeline Parallelism.

## Residency and lifetime

All four depth caches remain necessary for the next generated token.
Finishing a pass does not release its historical KV.
A transient inactive depth is not a dead KV namespace.
An exact tiering method must restore that namespace before the next corresponding attention.
The transfer test charges actual bytes and completion waits.

The layer-major control changes physical storage strides only.
It preserves the cache manager's public axes and depth-specific page tables.
It does not reduce the logical KV payload.

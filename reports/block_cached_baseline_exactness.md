# Block-cached baseline exactness

Completed-prefix states are mathematically independent of future blocks under the block mask. At a fixed sequence shape, repeated refinements are bit-exact at every cached hidden/KV/router/expert/post-layer boundary. Extending to a new block changes the BF16 SDPA execution shape and can create numerical-order drift, so B1 conservatively performs one full-prefix refresh at block open and caches it for the remaining refinements.

| Boundary | same-shape bit-exact rows | same-shape P95 max rel-L2 | block-open bit-exact rows | block-open P95 max rel-L2 |
|---|---|---|---|---|
| layer_input | 100.000000% | 0.000e+00 | 83.7404% | 5.107e-01 |
| key_projection | 100.000000% | 0.000e+00 | 83.7404% | 3.058e-01 |
| value_projection | 100.000000% | 0.000e+00 | 83.7404% | 5.583e-01 |
| router_topk | 100.000000% | 0.000e+00 | 90.8014% | 0.000e+00 |
| router_weights | 100.000000% | 0.000e+00 | 79.6473% | 3.475e-01 |
| routed_expert_output | 100.000000% | 0.000e+00 | 80.5027% | 1.192e+00 |
| post_layer | 100.000000% | 0.000e+00 | 81.7075% | 5.525e-01 |
| logits | 0.000000% | 0.000e+00 | 0.0000% | 6.791e-01 |

Classification: cached same-shape boundaries are bit-exact. The logits row is elementwise 99.99999% exact but contains a sparse non-equality per row and is not a cache boundary. Block-open differences are BF16 kernel-shape/numerical-order effects that can become semantically meaningful in deep layers.

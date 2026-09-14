# Active versus Dormant routing and expert dynamics

Lag-1 dynamics use actual route rows at representative layers 1, 8, 16, 24,
and 31.  H=1 is the primary definition.

| task/state | exact top-k % | top-k Jaccard | exact destination % | destination Jaccard | hidden cosine p50 | hidden rel-L2 p50 | routed-output cosine p50 | routed-output rel-L2 p50 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| GSM8K ACTIVE | 18.31 | 0.592 | 60.91 | 0.858 | 0.945 | 0.336 | 0.882 | 0.515 |
| GSM8K DORMANT | 10.96 | 0.511 | 53.51 | 0.827 | 0.895 | 0.464 | 0.781 | 0.689 |
| HumanEval ACTIVE | 19.13 | 0.599 | 61.37 | 0.860 | 0.951 | 0.316 | 0.895 | 0.486 |
| HumanEval DORMANT | 9.70 | 0.499 | 52.96 | 0.826 | 0.886 | 0.482 | 0.774 | 0.695 |

The central premise is falsified in the useful direction: future-dormant rows
are **less** stable than near-frontier ACTIVE rows in route identity,
destination identity, hidden value, and routed expert output.  Destination
sets remain broadly overlapping, but exact-set stability is only about 53%,
insufficient for an exact cache protocol.  Routed-output relative error is
roughly 0.69 before multi-layer propagation.

This also explains the economic failure of route-change safety: a fresh router
invalidates most dormant cache entries.  “Not accepted soon” is not equivalent
to “representation is temporally quiescent.”

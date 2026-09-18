# DeltaEP predictability

Across all 19 layers the live-MASK adjacent STAY fraction is 68.109% (router mass 72.340%). For the five vector-audited layers, adjacent inputs are closer than wrong-token, nonadjacent, and zero predictors, but exact BF16 equality is not high enough for a strong simple lossless codec.

| Boundary | cosine P50/P90 | relative-L2 P50/P90 | exact BF16 words P50 | |delta| P90-vector P50 |
|---|---|---|---|---|
| dispatch | 0.9292 / 0.9955 | 0.3784 / 0.7914 | 0.537% | 0.3282 |
| combine | 0.8151 / 0.9898 | 0.6250 / 1.2948 | 0.293% | 0.3724 |

These are activation/output vector similarities, not permission to reuse expert output. The previous branch-reuse result remains unsafe.

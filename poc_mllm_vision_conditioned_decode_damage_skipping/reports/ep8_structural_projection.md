
# EP8 structural projection

Status: **STRUCTURAL_EP8_PROJECTION_ONLY**

The 128 experts are mapped linearly to 8 ranks (16/rank). This is a bounded
held-out functional-path sample, not a latency model and not a promoted runtime policy.

| Policy score used for 20% offline mask | path reduction | remote paths removed | logical bytes removed | rank max/mean |
|---|---:|---:|---:|---:|
| F1_MODES_STYLE_GENERIC | 19.53% | 21 | 172,032 | 1.812→1.864 |
| F5_VISION_SEPARATED | 19.53% | 21 | 172,032 | 1.812→1.942 |

The byte field is a BF16 hidden+contribution logical payload proxy and excludes
DeepEP padding/metadata. EP8 latency, TPOT, E2E, and speedup are explicitly N/A.

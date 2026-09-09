# Correctness Validation

## Attention primitive

The exact query-block attention replay was compared with the unchunked reference at 1,024 tokens:

- cosine: 0.9999997616
- relative L2: 0.0770%
- maximum absolute error: 0.00390625

Thus the causal attention block construction itself satisfies the requested numerical target.

## Integrated Router/DeepEP/MoE path

The BF16 block decomposition changes accumulation order. Small last-bit attention/output-projection differences are amplified by router top-k boundaries. Across 30-repetition layer sweeps:

| Layer | Length | Block | Minimum cosine | Maximum relative L2 | Minimum route agreement |
|---:|---:|---:|---:|---:|---:|
| 4 | 8K | 1024 | 0.999611 | 2.791% | 99.469% |
| 4 | 32K | 1024 | 0.999722 | 2.360% | 99.539% |
| 24 | 8K | 1024 | 0.999637 | 2.693% | 99.036% |
| 24 | 32K | 1024 | 0.999703 | 2.437% | 99.072% |
| 47 | 8K | 1024 | approximately 0.9996 | approximately 2.3% | approximately 97.0% |
| 47 | 32K | 1024 | approximately 0.9997 | approximately 2.2% | approximately 96.6% |

Requested integrated thresholds were cosine >=0.9999, relative L2 <=1%, and route agreement >=99.9% (preferably 100%). The integrated candidate fails the latter two thresholds. Streaming and same-chunk/no-overlap outputs agree with each other; this is numerical router sensitivity, not asynchronous corruption.

All timing results are therefore diagnostic layer costs, not correctness-qualified end-to-end speedups.

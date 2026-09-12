# Refinement Workload Shape Trace

The clean-best mini32 run contains 65 physical waves. Shape traces cover layers
1, 16, and 31 and are collapsed once per logical wave; rank times use the
critical-rank maximum rather than a sum.

| phase | waves | ready requests, median | physical M, median | decision-live ratio | active experts | rows/active expert | tiny-expert fraction (<=4) | fanout | remote fraction |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| early | 2 | 32 | 1024 | 0.904 | 175.8 | 50.68 | 0.317 | 2.972 | 0.753 |
| middle | 27 | 28 | 896 | 0.445 | 206.3 | 35.39 | 0.229 | 2.999 | 0.755 |
| late | 36 | 5.5 | 176 | 0.181 | 167.2 | 8.73 | 0.480 | 3.013 | 0.753 |

Logical liveness collapses strongly, but each scheduled request still contributes
32 physical rows. The observed physical-M contraction is therefore primarily the
shrinking ready pool as requests complete, not live-row compaction within a request.
Remote fraction and rank fanout are nearly phase invariant. Late waves do become
more fragmented because the underfilled ready pool gives fewer rows per expert.

Representative-layer critical-rank medians for mini32 are:

| phase | Attention | Router | Dispatch | Expert | Combine | MLP | Block |
|---|---:|---:|---:|---:|---:|---:|---:|
| early | 1.498 | 11.569 | 0.936 | 1.050 | 0.261 | 26.354 | 28.335 |
| middle | 0.451 | 10.639 | 0.826 | 1.041 | 0.202 | 23.862 | 24.623 |
| late | 0.589 | 2.927 | 0.525 | 0.816 | 0.144 | 8.495 | 9.426 |

These are observer-heavy sampled-layer diagnostics, not clean request BCT.
Evidence: [`WAVE_SHAPES.csv`](../WAVE_SHAPES.csv),
[`LAYER_SHAPES.csv`](../LAYER_SHAPES.csv), and
[`EP_STAGE_SUMMARY.csv`](../EP_STAGE_SUMMARY.csv).

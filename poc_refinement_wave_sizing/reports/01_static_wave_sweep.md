# Static Wave-Size Sweep

Submitted batch is fixed at 32. Each request contributes at most 32 physical rows
per refinement forward, so the full-wave physical M is `32 × mini_batch_size`.

| mini | full-wave M | restarts | clean BCT median (s) | slowdown vs best | throughput (tokens/s) | peak HBM/rank (GiB) |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 32 | 3 | 81.610 | 1200.88% | 32.645 | 56.864 |
| 2 | 64 | 3 | 39.261 | 525.83% | 67.861 | 56.864 |
| 4 | 128 | 3 | 21.038 | 235.36% | 126.702 | 57.870 |
| 8 | 256 | 3 | 10.783 | 71.88% | 247.164 | 61.657 |
| 16 | 512 | 5 | 6.890 | 9.82% | 386.915 | 67.968 |
| **32** | **1024** | **5** | **6.273** | **0%** | **424.145** | **79.116** |

The best static EP4 setting is **mini32**. It is feasible, but it leaves only a
small HBM margin on an 80-GB H100. Mini16 is the only nearby point and is still
9.82% slower in median BCT.

The bounded GSM8K score was 5/32 for mini1/2/4/16/32 and 6/32 for mini8.
This is not a quality improvement claim: batching changes BF16 reduction order and
one borderline answer. Crucially, the best-static mini32 does not degrade the task
metric relative to the other static controls. No dynamic policy was run.

Primary evidence: [`STATIC_WAVE_SUMMARY.csv`](../STATIC_WAVE_SUMMARY.csv),
[`STATIC_WAVE_POINTS.csv`](../STATIC_WAVE_POINTS.csv), and
[`QUALITY_RESULTS.csv`](../QUALITY_RESULTS.csv).

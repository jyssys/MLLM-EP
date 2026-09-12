# Batch and concurrency scaling

## Submitted-request sweep

All points use 32 requests, generation budget 32, the same fixed
`mini_batch_size=4`, three engine restarts, and paired randomized topology
order. Positive numbers favor EP4.

| Submitted batch | TP4 median BCT (s) | EP4 median BCT (s) | paired median EP wall gain |
|---:|---:|---:|---:|
| 1 | 17.661 | 22.948 | -29.94% |
| 2 | 13.205 | 22.832 | -17.50% |
| 4 | 9.720 | 11.477 | -10.38% |
| 8 | 13.056 | 11.058 | +1.81% |
| 16 | 13.722 | 13.143 | +3.39% |
| 32 | 25.650 | 20.290 | +20.75% |

The apparent small-to-large crossover is not statistically stable. At batch
32 the three paired gains were +21.37%, +20.75%, and -25.91%; batches 8 and 16
also changed sign. Maximum-only evidence is therefore rejected.

## Physical model-forward microbatch sweep

Submitted batch is fixed at 16; one model forward contains 32 tokens per
request. `M` is the maximum physical token-row count in a full microbatch.

| Topology | mini batch | nominal M | restarts | median BCT (s) | median tokens/s |
|---|---:|---:|---:|---:|---:|
| EP4 | 1 | 32 | 1 | 83.608 | 19.68 |
| EP4 | 2 | 64 | 1 | 26.604 | 61.84 |
| EP4 | 4 | 128 | 3 | 13.115 | 125.47 |
| EP4 | 8 | 256 | 3 | **6.904** | **238.41** |
| EP4 | 16 | 512 | 3 | 8.040 | 204.73 |
| TP4 | 1 | 32 | 1 | 48.184 | 34.14 |
| TP4 | 2 | 64 | 1 | 21.963 | 74.91 |
| TP4 | 4 | 128 | 3 | 12.233 | 134.53 |
| TP4 | 8 | 256 | 3 | 7.610 | 216.29 |
| TP4 | 16 | 512 | 3 | **5.641** | **291.86** |

This is the largest actionable effect: changing EP4's existing dInfer knob
from 4 to 8 reduces BCT by 47.36% and raises tokens/s by 90.02%. TP4 prefers
16. At matched mini batch 8, EP4 is 9.28% faster in BCT; at 16 it is 42.53%
slower. Thus the optimal granularity differs by topology, but this is a static
configuration correction, not a new method.

## Interpretation

There is a real startup/fragmentation-to-payload transition, followed by an
oversized-wave penalty. However, the best observed point is recoverable with
the existing `mini_batch_size` option. All successor oracles therefore use the
best static topology-specific point, not the weak default.

# AR-ness × EP regime map

The strongest physical observation is a clear size-driven sparse-kernel
transition. In the fixed n=8/mini8 GSM8K trace, increasing B from 8 to 128:

| metric | B8 | B128 |
|---|---:|---:|
| median physical rows | 48 | 704 |
| active experts | 101.9 | 191.9 |
| rows / active expert | 3.67 | 29.76 |
| experts with <=4 rows | 75.4% | 29.3% |
| median forward wall | 91.1 ms | 185.4 ms |
| critical dispatch | 1.335 ms | 9.145 ms |
| critical expert kernel | 0.536 ms | 1.005 ms |
| BF16 dispatch bytes | 2.36 MB | 34.90 MB |
| mean rank fanout | 3.045 | 3.075 |

HumanEval shows the same direction. Small B is startup/tiny-GEMM dominated;
large B amortizes expert compute but becomes payload/dispatch and dense-row
waste dominated.

## What did not change

Rank fanout remains close to three and remote fraction remains close to 0.75.
The transition is therefore not a new communication-topology regime. It is
primarily a physical-M, packing, and payload regime transition.

## Liveness context

For GSM8K, modeled dead-row share rises from 54.1% at B8 to 61.3% at B128.
This is measured mask state mapped to row counts, not measured Epoch speedup.
Any benefit obtained by merely eliminating those rows directly collides with
Epoch-like liveness compaction.

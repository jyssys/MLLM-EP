# Online Serving Relevance

This trace is offline and queue-free. It cannot support a serving-speedup claim.
It does show the ready-pool behavior that a future online scheduler would face:

| phase | waves | ready pool p10 / median / p90 | physical M median |
|---|---:|---:|---:|
| early | 2 | 32 / 32 / 32 | 1024 |
| middle | 27 | 24.6 / 28 / 31 | 896 |
| late | 36 | 1 / 5.5 / 17 | 176 |

Continuous batching decides which requests are admitted to the ready pool. RAWS
would then decide how to partition that already-ready pool. In the closed-set data,
partitioning is almost never useful; the large apparent late-phase benefit comes
from comparing different ready-pool populations.

An online system might replenish late waves with new requests, but any benefit must
be evaluated as execution savings minus admission/batching delay. That is a new
queueing experiment, not support for RAWS in the present PoC.

Evidence: [`ONLINE_READY_POOL_ANALYSIS.csv`](../ONLINE_READY_POOL_ANALYSIS.csv).

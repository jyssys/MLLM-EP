# Mixed-domain locality headroom revisit

| wave | mapping | oracle | same-domain | P_full | P_separated |
|---|---|---:|---:|---:|---:|
| B4 | EP2 | 8.01% | 3.51% | 4.26% | 4.40% |
| B4 | EP4 | 8.01% | 3.51% | 4.26% | 4.40% |
| B4 | EP8 | 8.01% | 3.51% | 4.26% | 4.40% |
| B8 | EP2 | 6.52% | 2.78% | 3.79% | 3.73% |
| B8 | EP4 | 6.52% | 2.78% | 3.79% | 3.73% |
| B8 | EP8 | 6.52% | 2.78% | 3.79% | 3.73% |
| B16 | EP2 | 4.10% | 1.51% | 1.93% | 2.41% |
| B16 | EP4 | 4.10% | 1.51% | 1.93% | 2.41% |
| B16 | EP8 | 4.10% | 1.51% | 1.93% | 2.41% |

Primary B8/EP4 future-truth oracle headroom is **6.52%**. It uses actual future routes with greedy marginal-union grouping plus bounded pair-swap refinement, so it is non-deployable and not a proof of the globally optimal partition. Scheduler status: **STOPPED_BY_GATE**. No runtime scheduler was implemented when the 10% gate was not met. The EP8 rows are structural only.

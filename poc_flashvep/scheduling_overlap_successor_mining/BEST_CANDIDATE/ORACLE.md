# Direct request oracle disposition

| Tested native space | Additional mean-request E2E envelope |
|---|---:|
| Dense existing policies | 0.247% observed mix / 0.642% equal workload |
| Qwen3 three policies, broad four-workload pool | 0.587% / 2.351% |
| Qwen3 five knobs, full warmup/common KV, steady + bursty | 0% / 0% |

The last screen selects PP-only/chunk2048 for both workloads, including all three
held-out restart blocks. ALP paired reductions are -21.54% steady and -19.24%
bursty; an existing static option eliminates the measured excess.

Nine-SLO fixed-arrival attained-goodput envelopes peak at 12.31% in the broad
MoE screen and 6.97% in the fully warmed screen. Neither is a capacity-rate
sweep or a new method. Threshold selection is not the headline.

Exact best-chunk, arbitrary/joint partition and native PP×EP **direct request**
oracles are NOT ESTABLISHED. Independent batch minima and a 23–26% stage proxy
cannot substitute. Feasible successor headroom is unknown, not zero; no
qualifying >=10–15% non-trivial residual has been demonstrated.

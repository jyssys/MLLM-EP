# Live DP4 baselines

Status: **NOT_RUN beyond Stage-0 independent quality workers**.

Four GPUs execute quality shards with the same base checkpoint.
Two workers run fixed R4 and two run the adaptive operating point.
They process separate question shards independently.
This arrangement speeds quality evaluation, but it is not a coordinated four-replica serving experiment.

The study does not interpret different quality-worker finish times as a rank-imbalance witness.
Those workers have different policies and generation workloads.
They do not share an eligible waiting-request pool or a matched arrival process.

| Required live policy | Status | P95 E2E / ITL / goodput |
|---|---|---|
| Round-robin DP4+CDB | NOT_RUN, Stage-0 gate | UNMEASURED |
| Queue-aware DP4+CDB | NOT_RUN, Stage-0 gate | UNMEASURED |
| Admissible length-aware DP4+CDB | NOT_RUN, Stage-0 gate | UNMEASURED |
| Strong measured-cost-aware DP4+CDB | NOT_RUN, Stage-0 gate | UNMEASURED |

The planned system-wide concurrency values are 1, 2, 4, 8, 16, 32, and 64.
No result silently treats those values as requests per GPU.
Burst, closed-loop rolling, and Poisson open-loop comparisons remain `NOT_RUN`.

No measured witness establishes that one rank is idle while another has eligible runnable work.
No true-future-depth request assignment runs in a live engine.
No active request migrates after prefill.
There is no KV-transfer omission hidden inside a measured placement result.

These missing measurements do not imply zero DP headroom.
They prevent a supported positive or negative scheduling-headroom conclusion.

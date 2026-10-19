# Refill and hindsight interaction

The two switches act at different scheduling levels.
The native CDB scheduler is `LOCAL_ONLY`.
The outer controller assigns a GPU before prefill.
No runtime component migrates an active request or its KV.
See `source_scope_audit.md` for source locations and queue semantics.

| C | Hindsight, no-refill T(A1)/T(A2) | Hindsight, refill T(A3)/T(A4) | Refill alone T(A1)/T(A3) | Refill under hindsight T(A2)/T(A4) |
|---|---|---|---|---|
| 4 | 1.046× [1.000, 1.117] | 0.968× [0.955, 1.008] | 1.041× [0.989, 1.100] | 0.963× [0.944, 0.997] |
| 8 | 1.017× [0.977, 1.045] | 0.974× [0.898, 1.023] | 0.592× [0.565, 0.651] | 0.568× [0.565, 0.613] |
| 16 | 1.029× [1.000, 1.052] | 1.015× [1.010, 1.097] | 0.420× [0.381, 0.420] | 0.414× [0.385, 0.420] |
| 32 | 0.997× [0.985, 1.050] | 1.158× [1.106, 1.165] | 0.239× [0.230, 0.245] | 0.277× [0.262, 0.277] |
| 64 | 0.997× [0.997, 1.038] | 1.038× [0.986, 1.098] | 0.183× [0.177, 0.185] | 0.190× [0.173, 0.197] |

Each cell is a speedup with its paired 95% interval.
A value below 1.000× means slower execution.

| C | Mean log interaction | 95% CI |
|---|---|---|
| 4 | +0.058639 | [+0.046156, +0.080777] |
| 8 | +0.042830 | [-0.002428, +0.074826] |
| 16 | +0.001985 | [-0.023057, +0.022269] |
| 32 | -0.119815 | [-0.141233, -0.095217] |
| 64 | -0.023448 | [-0.049812, +0.002915] |

The interaction is `log[T(A1)/T(A2)] − log[T(A3)/T(A4)]`.
A positive value means greater hindsight benefit without refill.
The calculation uses paired raw repeat times, not averages of unrelated ratios.

## Mechanism and limitations

The selected cache requires slot-homogeneous recurrent kernels.
Refill interleaves token chains but cannot combine arbitrary recurrent depths into one kernel batch.
The native coda-priority and front-reentry rules also change local batch sizes.
Thus a better depth-work balance need not give a matching GPU-time balance.

The trace scatter and batch histograms show the observed difference.
They do not measure actual expert or weight HBM traffic.
The synchronous selected runtime is not proof about the fastest possible CDB implementation.
The campaign does not validate the optional asynchronous path.

## Actual rank work and time at C64

| Arm | Rank | Reference loop work | Actual loop work | Output tokens | Core stream, ms | All measured stage intervals, ms | Rank completion, s |
|---|---|---|---|---|---|---|---|
| A0 | 0 | 47717 | 48114 | 16043 | 22080.67 | 24921.15 | 26.878 |
| A1 | 0 | 12702 | 12740 | 4096 | 5854.51 | 6823.02 | 7.597 |
| A1 | 1 | 11732 | 12053 | 4034 | 5747.27 | 6413.22 | 6.861 |
| A1 | 2 | 11173 | 11187 | 3817 | 5602.56 | 6682.99 | 7.362 |
| A1 | 3 | 12110 | 12211 | 4096 | 5842.48 | 6555.52 | 7.040 |
| A2 | 0 | 11990 | 12201 | 4096 | 6094.50 | 7157.15 | 7.890 |
| A2 | 1 | 12063 | 12270 | 4096 | 5678.29 | 6360.28 | 6.817 |
| A2 | 2 | 11601 | 11610 | 3847 | 5898.55 | 6941.70 | 7.650 |
| A2 | 3 | 12063 | 12049 | 4004 | 5497.58 | 6203.62 | 6.683 |
| A3 | 0 | 12702 | 12701 | 4096 | 31282.20 | 35409.79 | 37.631 |
| A3 | 1 | 11732 | 11998 | 4034 | 24579.81 | 26922.09 | 28.012 |
| A3 | 2 | 11173 | 10745 | 3723 | 30712.10 | 35468.32 | 37.579 |
| A3 | 3 | 12110 | 12162 | 4096 | 34146.20 | 37588.06 | 38.964 |
| A4 | 0 | 11990 | 12180 | 4096 | 32849.93 | 37436.62 | 39.645 |
| A4 | 1 | 12063 | 12120 | 4096 | 33320.58 | 36539.86 | 37.870 |
| A4 | 2 | 11601 | 11623 | 3859 | 29549.66 | 33700.87 | 35.783 |
| A4 | 3 | 12063 | 12054 | 4004 | 26596.75 | 29236.23 | 30.399 |

These are trace-lane measurements of the observed trace workload.
Reference work comes from the earlier frozen requests, not the new counterfactual execution.
Thus reference work can differ from realized work after a batching change.
Stream-interval sums are not pure kernel time or exclusive cohort critical-path shares.

## C64 completion-critical rank breakdown

| Arm | Completion-critical rank | Prefill, ms | Core, ms | Exit-KV copy, ms | Prelude, ms | Coda, ms |
|---|---|---|---|---|---|---|
| A0 | 0 | 585.49 | 22080.67 | 1849.79 | 176.51 | 228.69 |
| A1 | 0 | 336.90 | 5854.51 | 487.07 | 65.04 | 79.49 |
| A2 | 0 | 217.53 | 6094.50 | 701.59 | 65.18 | 78.35 |
| A3 | 3 | 129.42 | 34146.20 | 2337.85 | 410.84 | 563.75 |
| A4 | 0 | 211.04 | 32849.93 | 3159.24 | 532.96 | 683.45 |

Every component in one row comes from the same completion-critical rank.
The table never adds maxima from different ranks.

## Comparison with v3

The historical 12.77% estimate concerned modeled P95 latency.
The historical 4.41% concerned mean measured P95 latency, approximately 1.046×.
Neither number was the primary cohort-makespan statistic used here.
`historical_v3_recomputed.json` recalculates both old statistics from preserved evidence.
The named sensitivity reuses both historical C64 placements.

The runtime keeps the new shared 10,240-token model-length limit, rather than the old 8192 limit.
New actual token and depth traces can also differ from the historical frozen reference.
These differences prevent a claim of identical experiment reproduction.

## Matched historical and arrival-order sensitivities

| Control | C | No-refill makespan speedup | Refill makespan speedup | No-refill mean-P95 speedup | Refill mean-P95 speedup |
|---|---|---|---|---|---|
| sensitivity | 64 | 0.616× [0.613, 0.803] | 1.032× [1.030, 1.047] | 1.085× [0.981, 1.287] | 1.061× [1.054, 1.073] |
| order_control | 32 | 1.018× [1.016, 1.090] | 0.962× [0.960, 0.963] | 1.042× [1.016, 1.090] | 0.970× [0.969, 0.971] |
| order_control | 64 | 0.994× [0.991, 0.996] | 1.056× [1.055, 1.065] | 0.994× [0.991, 0.996] | 1.059× [1.054, 1.069] |

The historical sensitivity reuses both the old generic and old hindsight assignments.
The shuffled-order sensitivity uses one seed, not a search for the most favorable order.
Its matched 4GPU ratios do not use an unmatched A0 denominator.
The mean-P95 ratios have their own paired bootstrap.

These sensitivity intervals resample repeats only. The old v3 intervals resampled both requests and repeats.
They do not transform a makespan confidence interval into a P95 confidence interval.

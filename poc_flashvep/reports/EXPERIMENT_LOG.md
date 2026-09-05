# Online runtime-regime discovery experiment log

## Prior autonomous discovery history (preserved)

The prior branch recorded H1/H4/H5/H6/H10/H13/H15/H16 controlled runs,
including the A2/A32 order-dependent sign flip and the fanout null result.
Those rows remain in the previous branch commit `b55d19a84deecf3951894e5383786397b7dde2b4`;
the current sprint adds the live rows below without reusing them as new data.

| Prior window | Key observation |
|---|---|
| 01:26–02:50 | Fragmentation/fanout effects were non-monotonic and first-use/order dependent; no stable routing-feature signal. |

| Time | Experiment | New live GPU data | Result | Follow-up |
|---|---|---|---|---|
| 2026-09-05 19:18–19:20 | Fresh HT SMS20 online wave, concurrency 8, 6 waves | 49,344 natural rank/layer records (M≤2048) | DeepEP HT verified; mixed prefill/decode trace captured | SMS sensitivity and tails |
| 2026-09-05 19:20–19:23 | Fresh HT SMS8 online wave, concurrency 8, 5 waves | 37,056 natural records | Runtime completed; same stock route path | Compare SMS20/8 |
| 2026-09-05 19:23–19:25 | Fresh HT SMS12 online wave, concurrency 8, 5 waves | 37,056 natural records | Runtime completed; same stock route path | Compare SMS12 |
| 2026-09-05 19:25–19:27 | DeepEP LL bounded attempt, MBT 8192 | engine startup failure | `nvshmem_qp_depth` assertion in `low_latency_dispatch` | Retry at 1024 |
| 2026-09-05 19:28–19:30 | DeepEP LL bounded attempt, MBT 1024 | engine startup failure during dummy profile | same LL qp-depth assertion | Backend marked blocked |
| 2026-09-05 19:30+ | HT SMS20 burst16, MBT 16384 | 30,912 valid rank/layer records | completed; high-load tail/phase coverage | Aggregated in route join |

The per-run elapsed time includes model initialization; live GPU execution is
the request/wave portion only and is reported conservatively in the final
report.  No method or routing change was implemented.

## Final recovery and analysis update (2026-09-05 20:50–22:22)

| Experiment | New live GPU data | Result |
|---|---:|---|
| HT SMS20, concurrency 8, 200 waves | 1,241,280 valid rank rows | completed; long steady trace |
| HT SMS12, concurrency 8, 200 waves | 1,241,280 valid rank rows | completed |
| HT SMS8, concurrency 8, 180 waves | 1,118,400 valid rank rows | completed |
| HT SMS20, concurrency 2, 180 waves | 1,118,400 valid rank rows | completed low-load control |

The final route-joined aggregate includes 6,105,264 valid rank/layer rows and
1,524,288 complete four-rank invocations after `M<=2048` filtering.  The
compact model fit uses a deterministic stride-5 sample of 1,219,430 rows;
the raw gzip traces are line-for-line preserved.  Route joining was added to
avoid treating rank-local CUDA timestamps as a global clock.  The final
SMS lower envelope is 0.11% median / 1.71% p90 / 2.10% max in prefill and
0.25% in decode.  No adaptive method was implemented.

# Virtual EP8 limits

- EP8_WORKLOAD_STATUS: STRUCTURAL_VERIFIED
- EP8_COMPUTE_STATUS: LOCAL_KERNEL_CALIBRATED
- EP8_COMM_STATUS: EXTRAPOLATED
- EP8_FULL_LATENCY_STATUS: UNVALIDATED_EXTRAPOLATION
- Virtual ownership: TP2/DP4/EP8, 16 experts/rank, all 128 experts covered once.
- Eight-rank source/destination traffic is reconstructed from real routes.
- No 8-rank collective was physically measured. Sensitivity bounds are not a
  statistical confidence interval, and four GPUs were never oversubscribed as
  fake EP8.

Projected routed-stage request medians (not E2E):

- endpoint optimistic: 34.470 ms
- calibrated base: 36.019 ms
- concurrency stressed: 42.810 ms

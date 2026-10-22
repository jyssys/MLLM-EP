# Figure index

| Figure | Evidence and scope |
| --- | --- |
| [Warm confirmation trade-off](figures/warm_confirmation_tradeoff.png) | Ten paired cohorts per policy, three starts. Means of cohort statistics. No pooled-token quantile claim |
| [FA4 discovery grid](figures/fa4_grid_v1_TTFT_ITL.png) | Single discovery cohort for each cell. Cold and configuration-order effects can remain |
| [Triton discovery grid](figures/static_grid_v1_TTFT_ITL.png) | Secondary backend. Same exploratory grid contract |
| [Triton grid metrics](figures/grid_metrics.png) | TTFT, p99, and peak allocated HBM for the 8K/concurrency8 discovery cell |
| [Physical Nsight timeline](figures/nsys_physical_timeline.png) | Actual GPU kernels and separate host submission ranges. Heavy cold diagnostic only |
| [Atomic event timeline](figures/fa4_trace_v1_atomic_C1024_P2048_N4_T2_timeline.png) | Model-call CUDA-event intervals. They can include waits and launch gaps |
| [G2 event timeline](figures/fa4_trace_v1_grouped_C1024_P2048_N4_T2_timeline.png) | Same event semantics and diagnostic shape |
| [Mixed event timeline](figures/fa4_trace_v1_mixed_C1024_P2048_N4_T2_timeline.png) | Stock mixed co-batching. Do not attribute batch-packing differences to G alone |

The server retains intermediate per-process plots and rejected cold-order plots.
Those plots are excluded from the Git figure set because they do not summarize the final confirmation.
No raw measurement was removed.

The CSV and raw archives remain the numeric authority.
Do not infer confidence intervals from the spacing of plotted mean points.

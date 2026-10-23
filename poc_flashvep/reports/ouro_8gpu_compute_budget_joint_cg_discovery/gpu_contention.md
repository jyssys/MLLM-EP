# GPU contention calibration

All 64 registered device/start jobs completed.
Each GPU has two independent starts for each of the 1/2/4/8-worker conditions.
Each start has three warmed cohorts at P≈2K/N4 and P≈8K/N8.
These are bounded natural calibration runs on legacy inputs, not new held-out serving results.

| Prompt bin | N | Workers | Median-ITL drift range across GPUs, % | Drained throughput drift range, % |
| --- | ---: | ---: | ---: | ---: |
| ≈2048 | 4 | 2 | [-0.740, 0.758] | [-0.937, 1.566] |
| ≈2048 | 4 | 4 | [-0.578, 0.122] | [-0.311, 1.033] |
| ≈2048 | 4 | 8 | [-0.582, 1.331] | [-1.699, 0.896] |
| ≈8192 | 8 | 2 | [-0.703, 0.479] | [-2.284, 4.542] |
| ≈8192 | 8 | 4 | [-0.802, 0.031] | [-0.959, 3.870] |
| ≈8192 | 8 | 8 | [-0.255, 1.063] | [-4.200, 0.475] |

Positive ITL drift means slower token gaps. Negative throughput drift means lower drained output rate.
The paired CSV preserves same-GPU/start comparisons and 95% bootstrap intervals.
Only two process starts support each device/concurrency cell.
The 3% threshold is a review trigger, not a proof of neutral interference.

The first context has a warmed group-start barrier.
The second context can have start skew across workers.
Telemetry retains power, clocks, load, memory, and disk counters.
The present analysis does not establish the physical cause of drift.
CPU control, driver contention, graph preparation, and power effects need separate attribution.

Numerical diagnostics can continue under their explicit non-performance scope.
Clean serving comparisons require matched worker load and GPU rotation.
No adaptive policy gain or sustainable capacity follows from this calibration.

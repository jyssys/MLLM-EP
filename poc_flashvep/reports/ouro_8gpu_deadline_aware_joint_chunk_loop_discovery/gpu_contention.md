# GPU contention calibration

All 64 registered device/start jobs completed.
Each GPU has two independent starts for each of the 1/2/4/8-worker conditions.
Each start has three warmed cohorts at P≈2K/N4 and P≈8K/N8.
These are bounded natural calibration runs on legacy inputs, not new held-out serving results.

| Prompt bin | N | Workers | Median-ITL drift range across GPUs, % | Drained throughput drift range, % |
| --- | ---: | ---: | ---: | ---: |
| ≈2048 | 4 | 2 | [-0.344, 0.065] | [-0.748, 0.446] |
| ≈2048 | 4 | 4 | [-0.398, 7.397] | [-13.035, 0.191] |
| ≈2048 | 4 | 8 | [-0.689, 15.407] | [-26.280, 0.267] |
| ≈8192 | 8 | 2 | [-0.950, 0.265] | [-0.305, 4.830] |
| ≈8192 | 8 | 4 | [-0.607, 0.402] | [-1.436, 3.403] |
| ≈8192 | 8 | 8 | [-0.533, 4.267] | [-18.374, 1.230] |

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

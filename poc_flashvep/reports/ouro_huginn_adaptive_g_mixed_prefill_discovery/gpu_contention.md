# Shared-resource contention control

The calibration uses Native Mixed P256/C128 with two background requests.
Every process completes its warm-up before a shared start barrier.
The table uses GPU0 across 1/2/4/8 concurrent workers.
Each worker count has one process start and three technical cohorts.
This is a narrow interference control, not a universal zero-interference claim.

| Concurrent workers | Technical cohorts | Median cohort wall ms | Ratio to solo GPU0 |
| --- | --- | --- | --- |
| 1 | 3 | 898.873 | 1.000000 |
| 2 | 3 | 901.996 | 1.003475 |
| 4 | 3 | 904.229 | 1.005958 |
| 8 | 3 | 900.241 | 1.001522 |

The eight-worker cross-GPU median-wall CV is 0.007857.
The primary confirmation runs one model process at a time in a reserved quiet lane.
It crosses physical GPUs and rotates policy order across three starts per model/width.
The exploratory grid uses parallel independent workers and remains a separate lane.
Its outliers do not replace the quiet paired results.
The telemetry records sampled clocks, power, memory, CPU load, and host I/O.
Those samples do not directly measure exclusive GPU idle time.

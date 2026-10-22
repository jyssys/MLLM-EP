# Independent GPU workers and fairness

The latest user amendment permits four independent single-GPU experiment workers.
It supersedes the initial GPU4-first resource sequence.
It does not change the English specification's scientific gates.
The experiment uses only physical GPUs4,5,6,7.
It does not implement TP, PP, or cluster-wide DP serving.

| Physical GPU | Expected UUID | Role |
| --- | --- | --- |
| 4 | `GPU-6076e2f2-5b63-3761-5586-56ceb7df8139` | Independent Single-GPU worker |
| 5 | `GPU-a1a1cfcf-93a1-3544-9a5e-e58144b68730` | Independent Single-GPU worker |
| 6 | `GPU-e3f3998e-0f1a-e94a-b97c-4abb0e8c2c28` | Independent Single-GPU worker |
| 7 | `GPU-4cc26b88-19fc-1988-f9e0-17858aa7a99b` | Independent Single-GPU worker |

Each child checks its UUID and exactly one visible CUDA device.
Each GPU has separate CUDA, Triton, and FA4 cache directories.
Each job retains its own process identity, model context, warm-up, logs, and raw results.
GPU0–3 receive no experiment CUDA context.

## Queue and resume

The dispatcher maintains a shared CPU queue of independent jobs.
A free worker receives the next job.
The job is a complete same-GPU policy block, not one permanently assigned policy.
Policy order changes across blocks.
Completed jobs and completed grid/traffic cells do not repeat after resume.
Preserved failures require review and a new corrective ID.

Finite job tails can still leave a worker idle after the queue empties.
Low-rate open-loop idle time is intentional measured behavior.
A burner must not occupy that gap on a measured device.
SLO threshold sweeps reuse the same traces on CPU instead of repeating GPU generation.

## Measured solo/concurrent control

The control uses warmed P2033/N4 static-R4 decode without an interfering prefill.
The GPU4 solo reference is the median of three cohort ITL medians: 27.189323 ms.
The concurrent table gives the mean of two cohort ITL medians per GPU.

| Mode | GPU | Observed median-ITL summary, ms |
| --- | ---: | ---: |
| Solo reference | 4 | 27.189323 |
| Four workers concurrently | 4 | 30.502408 |
| Four workers concurrently | 5 | 27.234770 |
| Four workers concurrently | 6 | 27.203744 |
| Four workers concurrently | 7 | 30.523234 |

GPU4's concurrent value is about 12.19% above its own solo reference.
This is one control episode, not a general interference constant.
GPU5–7 lack corresponding solo episodes in this control.
Their differences cannot be attributed solely to concurrent execution.
The study does not identify a NUMA, clock, power, or CPU cause from these values.

## Analysis consequences

Policy comparisons pair the same GPU/process block and trial.
The bootstrap resamples those blocks rather than treating all token intervals as independent.
The actual GPU assignment is retained in every measured case.
The queue is not a perfectly balanced Latin-square design for every offered rate.
Absolute rate curves therefore retain device and shared-host uncertainty.

The Poisson job allocation illustrates this limit.
Each listed process block still runs all eight policies on the same device.

| Configured req/s | Physical GPUs in startup order 0/1/2/3 |
| --- | --- |
| 0.125 | 4 / 7 / 7 / 7 |
| 0.25 | 4 / 5 / 5 / 6 |
| 0.5 | 4 / 5 / 6 / 6 |
| 1.0 | 4 / 5 / 6 / 7 |

Within-rate paired differences control the shared device/block better than cross-rate absolute comparisons.
They do not remove every temporal or shared-host effect.

`GPU_policy_crossover.csv` gives the fixed-grid allocations.
`GPU_policy_effects.csv` gives descriptive same-GPU policy ratios relative to matched Atomic C1024 cohorts.
`gpu_job_allocation.csv` gives actual job identity and policy order.
`serving_runs.csv` gives the physical device for each serving result.
`GPU_interference_calibration.csv` gives all observed calibration rows.
Per-job telemetry records one-second utilization, memory, power, clocks, and temperature.
That telemetry is not HBM-bandwidth or kernel-efficiency evidence.

## Resource ownership

The resource helper checks the user's recognized burner sessions and ownership.
It refuses to remove an unrelated process.
The helper resides in the old PP checkout but does not execute PP research code.
The native-vLLM Ouro Loop PP program remains paused.
After each campaign stage, the wrapper restores recognized burners on GPUs4–7.
The next stage stops those burners before its workers initialize CUDA.

The last diagnostic stage includes a brief GPU4-only policy control.
It compares the same P≈8K/N8 inputs and trials with an earlier concurrent GPU4 block.
Other managed devices remain idle during that explicit interference check.
This exception prevents burner load from contaminating the quiet control.
The final exit handler restores all four recognized burners.

## Final quiet-device policy check

Two additional cohorts per policy use the same GPU4 and inputs as the concurrent startup-5 block.
These are descriptive checks at different times, not another independent capacity study.
All 64 matched request outputs agree exactly between quiet and concurrent runs.

| Policy | Quiet mean long TTFT, ms | Concurrent mean long TTFT, ms | Quiet mean cohort-p99, ms | Concurrent mean cohort-p99, ms |
| --- | --- | --- | --- | --- |
| Atomic C1024 | 858.4 | 882.9 | 328.0 | 341.1 |
| Grouped C1024 | 978.9 | 1038.6 | 180.2 | 194.8 |
| Split C512 | 1969.4 | 2187.5 | 116.7 | 142.7 |
| Mixed C512 | 1567.3 | 1677.5 | 98.8 | 113.5 |

The severe-tail ordering remains in the quiet check.
G2 still has a lower p99 and a higher TTFT than Atomic.
Mixed still has the lowest p99 among these four controls.
Absolute values change with temporal and shared-host conditions.
The table does not identify their individual causes or provide a universal correction factor.

Across the main fixed cohorts, G2/Atomic mean paired p99 ratios are 0.571/0.529/0.308/0.497 on GPUs4/5/6/7 respectively.
Those descriptive device subsets contain 2/4/1/3 cohorts.
The small subsets do not establish device-specific confidence intervals.
Use the main paired process-block analysis for inferential comparisons.

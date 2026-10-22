# Measurement validity and profiling

The study uses three separate lanes.
Clean observer controls measure client-visible cohort time without QuantumObserver.
Event controls record host and CUDA-event intervals.
Heavy Torch profiling records CPU/CUDA kernels and stream timelines.
Heavy-profile wall time is never headline latency.

The main cost grid uses the event lane.
Its target is safe scheduling impact, not pure kernel service.
The observer records event-clock anchors and drift bounds within one GPU.
It never subtracts timestamps from different GPUs.
The main maximum alignment bound is 0.166647 ms across 1,728 cohorts.
This is a measurement bound, not a deadline guarantee.

## Observer controls

There are 144 clean/event paired cohorts across eight devices.
Each shape has four independent process-start groups and three repeats per policy.
The policy order and lane order cross over.
The completed controls include twelve separate heavy traces.

| Shape | Atomic mean event-lane wall change | G2 change | Mixed change |
| --- | --- | --- | --- |
| P2033/N4 | +1.731%, CI [0.462%, 2.993%] | +1.970%, CI [0.942%, 2.922%] | +2.650%, CI [−2.482%, 8.447%] |
| P8177/N8 | +1.828%, CI [−0.733%, 3.389%] | +2.236%, CI [−1.736%, 5.733%] | +2.668%, CI [0.038%, 4.930%] |

Intervals resample the four process-start groups, not tokens.
These changes are not pure instrumentation cost estimates.
Only 121/144 paired cohorts are entirely token-exact.
At request level, 832/864 sequences are exact and 855/864 lengths match.
Changed batching can change BF16 outputs and the amount of natural work.
The evidence does not establish a state bug or explain every mismatch.
It prevents a claim of universal instrumentation output parity.

GPU0/GPU1 v2 retries ran while six other devices ran owned burners.
Other completed controls ran in the original eight-worker queue.
Within-start clean/event pairs remain the comparison unit.
Raw cross-start latency must not be treated as an isolated policy effect.

## Heavy traces

`heavy_profile_index.json` records each trace hash.
`kernel_service_summary.csv` records kernel calls and summed service time.
Summed service is not non-overlapped critical-path share.
`figures/kernel_timeline_gpu0.png` and `kernel_timeline_gpu1.png` show actual CUDA stream activity.
These are single-GPU timelines, not model-parallel execution.
Two representative full traces are preserved in compressed raw archives.
Other large profiler captures remain local with hashes.

The first two heavy-profile attempts retained a bound runner reference and caused a later allocation OOM.
The v2 attempts release those references and complete all six policies.
Original failure logs remain preserved.
No performance number comes from the failed partial attempts.

## Graph and staging

Native decode graphs, static buffers, and multi-stream execution remain enabled.
Grouped prefill uses the existing staged path.
Staged G4 separates that path change from group-size changes.
Actual graph counters and staged bytes appear in the quantum table.
This study does not claim measured HBM traffic from allocation size.
It also does not claim a complete kernel-level critical-path decomposition.

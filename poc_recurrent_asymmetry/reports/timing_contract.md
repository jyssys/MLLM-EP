# Timing contract

## Distinct timing quantities

| Quantity | Boundary | Permitted interpretation |
|---|---|---|
| Reference clean wall | Device synchronization before and after the phase | Actual eager-runtime hardware latency |
| Native clean wall | Submitted request through returned result | Actual optimized-runtime hardware latency |
| Native TTFT / TPOT | Scheduler timestamps through first/last returned token | Fixed-output hardware timing, not natural quality |
| CUDA component interval | Events on one execution stream | Includes exposed enqueue gaps between events |
| Kernel service | Actual kernel start/end intervals | Inclusive activity, not additive E2E fractions |
| Trace-local partition | Union of actual correlated GPU intervals | Instrumented activity envelope only |
| Oracle schedule | Measured inputs in a resource-limited simulator | Modeled latency, not measured E2E |

The observer validity limit is an absolute median wall overhead of 10%.
The study retains failing observations and labels their timing `UNVALIDATED`.
It does not normalize an invalid trace into a clean E2E breakdown.

## Reference sweep

Eight calibration question IDs define the frozen prompt shapes.
The context constructor preserves an intact final question within a repeated token prefix.
This is a hardware-shape workload, not a natural long-context benchmark.

The reference sweep uses 32 actual cached forwards with fixed token ID 13.
This setting removes output-length variation.
It does not measure natural-output TPS or user-visible reasoning E2E.
Initial kernel warmup uses a short prompt and three cached forwards.
Each trial clears reusable allocator blocks before the timed prefill.
The first 1K request group retains substantial startup variation.
No outlier removal or retrospective warmup exclusion occurs.

Five technical repetitions form a median for each frozen request group.
The analysis then averages those group medians.
Bootstrap confidence intervals resample groups 5,000 times with seed 20261017.
B1 has eight groups, B2 four, B4 two, and B8 one.
The B8 row therefore has no request-level confidence interval.
Layers, loops, cached steps, and repeated contexts are not independent questions.

The native controls use eight matched requests and five technical repetitions per context.
The paired TP comparisons resample request-level percentage differences.
Both controls keep the cross-backend `REFERENCE_PARTIAL` label.

## Observer limitations

The light lane measures one component family per run.
It repeats the full actual operation sequence five times.
It does not sample one forward and multiply it by 96.
Different family runs cannot form an exact exclusive E2E partition.

The light lane uses shape-specific warmup and separate allocations.
Its total wall time differs from the broad sweep's total wall time.
Use the matched total from that lane for its component ratios.
Do not substitute a faster total from another run.

| Nsight context | Median observer overhead | Status |
|---|---:|---|
| 1K | +56.879% | UNVALIDATED for clean-time shares |
| 16K | +13.363% | UNVALIDATED for clean-time shares |

The correlated trace contains 166,872 actual GPU events.
Its exclusive bins include idle/unresolved waits and multi-category overlap.
They sum to the instrumented activity envelope.
They do not sum to clean request wall time.
Idle time does not establish removable architecture cost.

Native profiler captures contain one real 16K prefill and three cached forwards.
They classify actual kernels without extrapolating a final CUDA Graph replay.
Rank-local profiler clocks do not establish inter-rank timestamp alignment.
The analysis therefore does not invent a synchronized four-rank critical-path trace.

## Communication

P2P host timing includes enqueue, allocation where applicable, and completion synchronization.
Destination-stream events report a separate interval.
Useful payload divided by that interval is `ApparentThroughput`.
It is not a pure-link bandwidth measurement.

NCCL measurements select one cumulative-critical rank for each benchmark case.
They retain all per-rank samples.
They do not sum maxima from different ranks.
NCCL tensor payload is not measured wire traffic.

Full clean multi-GPU critical-path attribution remains `PARTIAL`.
Actual HBM traffic remains `UNAVAILABLE`.
No measured architecture E2E improvement is claimed.

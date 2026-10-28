# Timing and uncertainty contract

## Client-visible scope

The client is an in-process serving harness. The experiment does not measure HTTP transport.
Closed-loop E2E starts before request admission and ends when the harness receives the final output.
Poisson mode instead starts at the scheduled arrival, including delayed harness delivery.
TTFT ends at the first emitted token.
TPOT is `(E2E - TTFT) / (output_tokens - 1)` for requests with at least two output tokens.
TPOT is undefined for one-token requests. The aggregate analysis excludes them from TPOT statistics.

The main workload uses real greedy generation. It permits EOS.
Output caps bound the workload but do not force output token IDs or suppress EOS.
The output-cap-128 regime is separate from the cap-32 regime.
Actual output lengths, not declared caps, determine output throughput.

Loading and the declared warmup precede timed request admission.
Any later graph preparation, shape compilation, scheduling, or synchronization remains inside measured request time.
Profiling runs are separate. Their latency does not enter clean serving comparisons.

## ITL and SLO

ITL uses harness-visible token emission times.
A coalesced emission contributes its observed waiting gap and explicit zero intervals between simultaneously delivered tokens.
The analysis also reports single-emission-only intervals separately.
It never removes a waiting gap merely because the next emission contains multiple tokens.

`configs/frozen_slo.json` fixes the SLO grid before repeated performance evaluation.
The relative ITL thresholds are 5/10/25 times the recorded native decode median.
The absolute ITL grid is 100/150/200/250ms.
The TTFT grid is 1/5/10 seconds. The E2E limit is 30 seconds.
Every ITL must pass for request-level joint attainment.
Population p95 or p99 ITL is a different statistic.

Length termination counts only as a valid completion for the declared bounded serving workload.
It does not establish reasoning completion or downstream answer correctness.
Goodput is the number of qualifying requests divided by the cohort wall time.
Closed-loop goodput does not establish maximum sustainable open-loop capacity.

## Phase percentages

`e2e_phase_breakdown.csv` partitions coordinator wall time into these host phases:

1. Coordinator queue.
2. Reservation and control.
3. P-assigned prefill, including source backpressure.
4. Final uncovered handoff.
5. D resume and first token.
6. Remaining decode.

For A, D resume includes the remaining prefill loops.
It is not only coda or sampling time.
The analysis also records the residual between harness E2E and coordinator E2E.
That residual includes harness delivery, request creation, and output polling.
The `_harness_percent` columns use the actual harness E2E denominator.
The older `_percent` columns use the coordinator denominator.

The analysis checks phase order. It does not clamp negative notification intervals to zero.
Invalid order must remain visible and cannot support a physical non-overlap interpretation.
These host phases are not pure GPU attention, MLP, or DMA times.
Their percentages must not receive those labels.

`e2e_phase_percent_summary.csv` aggregates the seven phases, including the harness residual.
Its primary percentage divides summed phase time by summed request E2E time.
It also reports the mean of individual request percentages separately.
Those two averages can differ when request durations differ.
Concurrent request intervals overlap.
Their aggregate shares do not partition cohort makespan or GPU utilization.
The accounting tests check that every valid request partition sums to its harness E2E.

`handoff_frozen_timeline_oracle.csv` removes only the observed final uncovered handoff interval from each request.
It keeps all other intervals fixed.
This counterfactual does not replay changed queueing or transfer contention.
It is not a global serving bound or a measured speedup.

## GPU and transfer accounting

The NIXL counter records completion-observation duration for each transfer.
Transfers can overlap. Their duration sum is not request wall time.
The final uncovered handoff is a separate host interval.

Short profiler summaries use interval unions for kernels and copies.
They retain inclusive operator durations separately.
The device window runs from its first recorded event to its last recorded event.
It excludes unrecorded leading or trailing idle time.
Copy/kernel overlap in that window is not automatically E2E latency savings.

The NVML monitor samples physical-device memory every 0.2 seconds.
It records a sampled peak, not a continuous exact peak.
Its utilization field uses the NVML averaging window. It is not exact kernel busy time.
PyTorch allocator peaks include warmup and remain a separate memory field.

CPU evidence logging also has a measured cost.
The I/O counter includes serialization, flush, and fsync.
Its duration can overlap GPU computation. It is not additive E2E time.
The lifetime maximum can include warmup. Cohort count and duration use counter deltas.
Some early trials predate this counter. Their I/O cost is unavailable, not zero.

## Statistical acceptance

Accepted gains require at least three independent starts and equivalent outputs.
The paired bootstrap resamples starts and matching request IDs.
It reports p95 E2E and TTFT speedup intervals.
Three starts still provide limited information about run-level variation.
The report must not imply statistical equivalence from a narrow small-sample interval alone.

Synthetic prompts can repeat token patterns. The master table records unique input hashes.
Output equality, output length equality, and depth-trace equality are different checks.
Any positive candidate requires a frozen-build confirmation if telemetry revisions differ between its paired trials.
Each attempt retains its complete source ZIP and runtime patch hash.

Speedup uses `baseline / method` at the same GPU budget.
A 10% latency reduction requires speedup of at least `1 / 0.9`, approximately 1.111×.
Single-request latency, cohort makespan, and throughput remain separate metrics.

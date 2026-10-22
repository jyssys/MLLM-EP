# Timing contract

## Three different clocks

1. Clean serving latency uses `time.perf_counter()` on the host.
2. Light submission descriptors record host entry and return, not GPU completion.
3. Heavy CUDA events record a device-stream elapsed interval after safe completion.

Do not sum these durations into an E2E breakdown.
Do not treat a host submission span as pure GPU compute.
Heavy event intervals can contain staging, stream waits, and host launch gaps.
Nested intervals and different streams can overlap.
Nsight kernel service totals are not non-overlapped E2E percentages.

## Primary open-loop definitions

An independent producer submits frozen arrivals without waiting for engine capacity.
The primary request clock starts at its scheduled arrival.
The trace also retains actual producer send, engine enqueue, and first GPU submission.
This client runs in-process. It does not measure HTTP or network latency.
The engine and model load before the declared warm-up and offered-arrival window.
Thus this is warmed serving, not process cold-start latency.
Any new request-bound graph preparation after an arrival remains inside that request's clock.

- TTFT is first observed token time minus scheduled arrival.
- ITL/TBT is the gap between two actual consecutive token observations.
- TPOT is last-to-first token time divided by output tokens minus one.
- E2E is final observed completion minus scheduled arrival.
- Request TBT compliance requires every observed ITL to meet its limit.
- Joint compliance also requires TTFT compliance and a complete natural response.

The observer asserts one newly observed token per delivery update.
It does not invent timestamps inside a multi-token delivery.
EOS, observed frontend stop, length cap, error, and timeout remain distinct.
A capped response does not pass the complete-response joint SLO.
A natural joint pass also requires at least two observed output tokens.
Every offered request remains in the denominator.

## Denominators

Offered-window goodput uses qualifying offered requests divided by the declared arrival window.
The declared drain permits those requests to finish after that window.
Completed-in-window goodput counts only qualifying completions before window end.
Drained-cohort throughput uses the observed first-arrival-to-finalization interval.
These are different denominators.
The continuity prompt-rate proxy divides prompt tokens by long-request TTFT.
It includes scheduling and decode interference, not only prefill kernel execution.
The configured Poisson rate and realized offered count/window are separate fields.
Short arrival windows can show substantial random differences between them.
Neither drained-burst throughput nor a single finite window proves sustainable capacity.

The finite-window stability screen is frozen in `slo_registration.json`.
It uses queued-backlog slope, unfinished fraction, errors, and timeouts.
A growing queue is an overload signal.
A flat queue with too many in-service requests can fail the screen without proving divergence.
The analysis reports those reasons separately.

## Quantiles and uncertainty

The fixed-grid primary tail statistic is the mean of cohort-level background ITL quantiles.
Its paired interval resamples process blocks and cohorts.
The CCDF is a descriptive pooled interval distribution.
Serving pooled p99 and per-request maximum TBT are separate fields.
The serving bootstrap resamples engine blocks and request families.
These are crossed cluster weights, not independent token samples.
Two excerpt lengths from the same source article share one family.
Repeated copies of that family across engine starts retain the same family weight.
All intervals use 5,000 draws with seed20261023.
Tokens and kernels are never independent inferential samples.

The registered capacity target is90% joint attainment.
Sensitivity targets are80% and95%.
A qualified rate requires sufficient requests, repeated nondivergent queues, and a confidence-aware bracket.
Small samples receive an exploratory label even when the point estimate exceeds90%.

## GPU fairness

Every ordinary policy block runs on one physical GPU with a fixed configuration.
Policy order changes across independent starts.
Comparisons pair the same block, trial, and workload.
GPU identity remains a blocking factor.
The solo/concurrent calibration records device and shared-host effects.
It does not identify a NUMA, power, or clock cause without another controlled experiment.

Each GPU has independent CUDA, Triton, and FA4 cache directories.
Each policy receives the same declared warm-up procedure.
Ordinary runtime graph preparation after an offered arrival remains inside client latency.
Heavy profiling executes in separate jobs.
Coarse one-second utilization telemetry is not a compute-utilization or bandwidth measurement.

## Prefill count definitions

An Atomic core-containing quantum includes its bundled prelude.
Grouped/Split/Mixed can execute a separate staged prelude.
The response surface reports core-containing and all-prefill submission counts separately.
Full-R4 work accounting counts token rows times executed recurrent loops.
It does not count prelude or coda as extra model depth.
The complete request TTFT includes those boundary operations.

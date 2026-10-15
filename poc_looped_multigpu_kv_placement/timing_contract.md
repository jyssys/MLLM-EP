# Timing contract

## Clean native measurements

The primary wall interval surrounds each `LLM.generate` call.
The model, native compilation, graph capture, and warmup precede that interval.
Inputs are frozen token IDs. No cross-request prefix cache hit is allowed.
Output serialization and detokenization remain inside the client wall boundary.
No profiling hook or memory RPC runs inside that boundary.

The native first-token interval starts at `scheduled_ts` and ends at `first_token_ts`.
Cached TPOT is `(last_token_ts - first_token_ts) / 63` for 64 returned tokens.
Native request latency starts at `queued_ts`.
These engine intervals exclude some client-side handling.
They are not isolated GPU kernel timings.

## Throughput

A controller measures concurrent independent engines with one shared release gate.
Prompt preparation and readiness acknowledgement precede that gate.
The interval ends after all engines return their results.
Output tokens divided by actual controller wall gives aggregate TPS.
For multiple waves, sum their measured walls before division.
Do not sum per-request TPS or use the largest isolated engine time as a measured controller wall.

Offered concurrency is not simultaneous resident concurrency.
Report native running/waiting evidence and cache-page admission limits separately.
The short-context maximum is censored at eight offered requests.

## Observer lanes

The outer-event lane records actual model and sampler calls on each worker.
It measures all forwards rather than multiplying one replay by a token count.
Clean/event order alternates within technical repeats.
Request-level bootstrap uses paired overhead after repeat averaging.
The observed mean is 0.536%, with 95% CI [0.429%, 0.641%].
Output tokens remain identical.

The native kernel lane profiles one request with four outputs.
Its observer overhead is 4.675% against three warmed same-lane clean calls.
The eager source lane is separate and disables native compile/graphs.
Its corrected observer adds 24.624% wall time.
Its fine quantitative attribution is therefore UNVALIDATED.
Loop/layer labels remain diagnostic evidence of execution order.

## Attribution limits

Per-rank CUDA event origins are not globally synchronized clocks.
Do not subtract timestamps across ranks to calculate waiting.
Kernel service is the sum of kernel durations on one rank.
GPU busy union merges overlapping intervals on that rank.
GPU execution span measures its first-to-last recorded kernel interval.
Client E2E includes additional engine/frontend work.
None of these quantities automatically equals a non-overlapped critical-path share.

The profile identifies native TP All-Reduce and output-logit All-Gather.
It does not establish pure transport bandwidth or separate readiness from protocol time.
KV read remains fused with attention.
Logical KV capacity is not measured HBM traffic.
No MoE EP All-to-All exists in these dense models.

## Statistical limits

Eight requests are the independent calibration units.
Three technical repeats precede 2000 request-bootstrap resamples.
Layers, tokens, loops, and repeats are not independent serving samples.
The eight-request throughput cohort has three repeats but no independent cohort-level population CI.
Request P95 values are descriptive for that cohort.
TP layout order is not interleaved across large engine instances.
Other GPUs remain active under their users' control.
These limits remain part of every interpretation.

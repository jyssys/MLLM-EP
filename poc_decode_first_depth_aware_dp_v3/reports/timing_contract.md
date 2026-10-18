# Timing and evidence contract

## Physical runtime

The selected runtime is the CDB authors' pinned implementation.
Each physical GPU hosts one complete BF16 Ouro-1.4B replica.
Physical GPUs 4, 5, 6, and 7 map to logical CUDA device 0 in their respective processes.
There is no TP collective or EP All-to-All in this DP4 experiment.

The local CDB engine uses its native recurrence, gate, KV, refill, and coda paths.
The new host interface supplies requests to its existing arrival-queue boundary.
It does not implement a new model engine.
Requests remain on their initial GPU. KV migration is forbidden.

## Metrics

| Metric | Definition |
|---|---|
| External arrival | Host monotonic arrival timestamp in the controller's common time base. |
| Assignment | Host timestamp when the controller selects the DP rank, before prefill. |
| Worker admission | Timestamp when the worker creates the native request state. |
| TTFT | First host-visible token timestamp minus external arrival. Includes queue delay. |
| E2E | Final native request-completion timestamp minus external arrival. |
| ITL | Difference between consecutive host-visible token timestamps. |
| Makespan | Last completion timestamp relative to case origin. |
| Output throughput | Generated token count divided by the arrival-to-last-completion span. |
| Relative goodput | Requests within the frozen calibration-relative E2E limit, divided by the same span. |

All processes use the same host monotonic clock.
These request timestamps are host-visible serving measurements. They are not GPU-ready timestamps.
We do not infer GPU collective wait time from them.

The serving supplement reports mean request TPOT, decode P50/P95, ITL P99, and allocated HBM.
The exact first-schedule timestamp was not exported.
Thus separate local prefill-queue delay remains `UNRESOLVED`. TTFT still includes that wait.

The first output token comes from full-R4 prefill.
Adaptive decode starts at the next generated token.
The depth denominator excludes the first token.
Executed loop work counts real recurrent batch members, not the configured maximum depth.

## Clean and trace lanes

The clean lane records requests, exits, token completion, and low-rate queue status.
It does not install per-stage CUDA events.
The trace lane adds events around prefill, recurrent, and coda calls.
It records host spans and compute-stream event spans separately.
These spans include their own dependencies. They are not a kernel service-time sum.

Both lanes use cached decode graphs. Prefill graphs remain disabled.
Unscored calibration work warms graph shapes before measurement.
Late captures remain visible in native counters.
We do not remove unfavorable raw measurements.
Cost fitting excludes its highest 0.5% stage samples. The raw samples remain intact.

The eight-request B1 observer check compares identical frozen prompts on physical GPU7.
It measures graph+trace versus graph+clean. It also checks eager-versus-graph output consistency.
This small check does not establish observer validity at every batch size.

## Simulation

The simulator preserves autoregressive order and native CDB slot grouping.
It includes prefill, coda, recurrence, capacity reservations, and request arrival.
It includes measured host-loop residuals.
It does not equate the sum of request depths with GPU service time.

Calibration uses only calibration-request traces.
Validation compares held-out live round-robin and queue cases.
We report makespan, P50/P95 E2E, P95 TTFT, and P95 ITL prediction errors.
The approximate 10% error target is an engineering heuristic.
Failure blocks claims of a validated realistic scheduling oracle.

## Statistical scope

Questions or requests are the statistical units.
Tokens, layers, and recurrent loops are not independent samples.
Quality uses 20,000 paired question-bootstrap draws with seed 20261018.
Oracle uncertainty uses 2,000 paired request-bootstrap draws.
Those oracle intervals condition on the selected placements. They do not repeat optimization within each draw.

The pilot has 128 distinct requests. It is not a final large-scale serving study.
The hardware discovery cap is 256 with natural EOS.
Most bounded outputs do not establish complete-answer quality.
The separate 64-question quality recheck uses cap 8192.

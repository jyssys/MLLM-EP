# Timing contract

## Separate measurements

Client latency uses the in-process frontend's monotonic clock.
Arrival-to-first-output is TTFT.
Successive delivered output timestamps define ITL.
Arrival-to-final-output defines request E2E.
This frontend does not claim HTTP, network, or external load-balancer latency.

Clean runs disable the per-submission service observer.
Diagnostic runs label observer activity explicitly.
The cost of the live selector, admission filter, and fairness rule remains in both lanes.
Warm-up results are not measured serving trials.
Graph capture that occurs after a measured arrival remains part of that request's latency.

The service observer starts its CUDA event at GPU preparation entry.
The native executor already issued request-dependency waits at this point.
The event ends after the native model execution on the same stream.
The interval includes required GPU staging and possible CPU launch gaps.
It is not pure kernel-busy time.
Other concurrent streams can still affect its measured service.

Each record preserves host dispatch time, event readiness, queued-event status, and enqueue-to-service delay.
These fields are not added to the service interval as independent E2E costs.
Events share a GPU clock within one worker.
Clock alignment does not compare timestamps from different GPUs.
Finished events supply durations only after native synchronization.

## Work completion

Prefill token-loop work counts actual span size times executed recurrent passes.
Only a span reaching R4 receives completed prompt-token credit.
An R1/R2 span has outstanding work and no completed-token credit.
Every complete-chunk objective includes prelude, remaining groups, and registered yield overhead.
The component sum is not a measured E2E saving.

Token-completion fairness observes native coda host delivery.
It does not credit `loops_done` or an issued recurrent batch.
The frontend records actual in-process output delivery separately.
The native-delivery timestamp is not an external-client timestamp.

## Sampling and uncertainty

Policies cross on the same GPU and model-process block.
Policy order rotates across GPUs.
Independent process/GPU blocks and source families define uncertainty units.
Tokens and quanta are not independent replications.
Pooled ITL quantiles remain distinct from mean cohort quantiles.

The primary budget lane reports actual per-quantum overrun rates by path.
A budget is not a guarantee about a complete-token gap.
Progress escapes and unknown shapes remain visible in their denominators.
No aggregate decode coverage can hide poor prefill coverage.

Capacity needs at least 1,000 offered requests per arm/rate and three independent starts.
It also needs a confidence-aware attainment bracket and stable queues.
Short capped cohorts cannot establish sustainable capacity.
Caps, errors, and unfinished requests remain in complete-response SLO denominators.

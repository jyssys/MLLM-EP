# Serving capacity qualification

Sustainable request-rate capacity is NOT_RUN and NOT_QUALIFIED in this completed study.
The executed pilot uses finite burst-plus-delayed-prefill cohorts.
The independent producer follows scheduled arrival timestamps without waiting for engine capacity.
This is not a Poisson rate sweep.

The completed quiet confirmation has three independent starts per shape/budget.
Its measured count per arm/budget is 24 requests at P2033/N4 and 48 at P8177/N8.
Those counts are far below the registered 1,000 requests per arm/rate.
A total count across different policies must not be used to satisfy that denominator.

The study can report finite-cohort TTFT, pooled ITL, E2E, drained output throughput, and complete-response attainment.
It cannot report a sustainable maximum request rate from those measurements.
Output tokens per second are not a substitute for SLO-qualified request capacity.

The frozen sensitivity grid uses TTFT thresholds 1/2/5/10 seconds.
The every-gap TBT thresholds are 100/150/200/250/500 ms.
Request-level every-gap attainment is distinct from pooled token P99.
Caps, errors, and unfinished responses remain in complete-response denominators.
Declared-drain goodput is distinct from completed-in-window goodput.

A full Poisson/burst capacity campaign is conditional on a verified incremental Joint benefit.
The quiet G-only/Joint comparison did not satisfy that progression gate.
The specification permits a focused negative diagnostic when G-only or native Mixed already matches or dominates Joint.
No missing capacity result is replaced by a modeled rate or a zero-gain measurement.

# Direct request-level impact

Run: `direct_stock_run4`; physical GPUs 1--4; 12 real-image serving waves.
Scheduler IDs in child workers were joined to `RequestOutput.request_id` by
their numeric prefix. Co-batched invocation excess was attributed to each
participating request and capped by that request's observed E2E, so the value
is an upper bound and does not double-count rank rows.

- requests: 84
- aggregate observed request E2E: 156,643.356 ms
- capped assigned excess: 1,710.985 ms
- aggregate removable upper bound: **1.092%**
- median request share: 0.294%; p90 1.774%; p99 13.158%; max 15.628%
- without largest logical event: 0.615%; without top five: 0.377%

This directly fails the 12% hard gate. Wave-critical projection is retained
only as sensitivity; it is not substituted for request E2E.

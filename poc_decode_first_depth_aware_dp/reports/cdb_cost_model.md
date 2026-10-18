# CDB cost model

Status: **NOT_RUN as a validated serving cost model**.

The study executes the existing CDB engine during Stage-0 quality experiments.
It does not infer an efficient high-concurrency cost surface from B1 quality generation.
The configuration enables refill, but one resident request cannot establish the benefit of mixed-depth refill.

The bounded physical probe verifies actual recurrent work omission.
Its dynamic run executes 44 decode core applications instead of 60 for the same 15 decode positions.
Its exit-KV observer synchronizes repeatedly to inspect copied data.
That diagnostic wall time is not clean serving latency.

The quality runner records request wall time, host-visible TTFT, and decode-span wall time.
It also records actual loop-token work and memory allocation.
These fields record quality-run observations, not pure GPU service intervals.
They do not isolate gate, sampling, KV copy, scheduler, or GPU-idle costs.

The following required Stage-2 measurements remain unvalidated:

- Per-loop cost versus actual active batch size and context.
- Mixed-depth versus homogeneous batch cost with matched request sets.
- Dynamic refill versus no-refill performance at approved quality.
- Queue waiting, GPU-idle intervals, and per-rank admission overhead.
- A trace simulator validated against live RR and queue-aware DP4.

No cost model multiplies a constant loop time by the sum of token depths.
No model converts loop-count reduction directly into latency reduction.
The simulator acceptance target remains approximately 10% error on live held-out serving metrics.
The study reports no simulator accuracy or realistic oracle gain without that validation.

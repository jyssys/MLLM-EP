# Conditional real-workload experiment

The real-workload stage starts only after a credible quality baseline.
Natural AIME and MATH outputs define the workload.
An input with a long prompt and 64 forced outputs is not a substitute.
The primary contrast is long generated reasoning with growing 192-slot KV state.

The planned physical layouts are TP1, TP2, TP4, TP1×DP4, and TP2×DP2.
The serving sweep offers concurrency 1, 2, 4, and 8.
Replica engines must execute concurrently on the same four GPUs.
The offered frozen cohort must remain the same across layouts.
Admission, queueing, eviction, caps, failures, and capacity limits remain visible.

Each request records TTFT, E2E, output tokens, EOS, correctness, and queue/service timing.
Token intervals use output bins 1–512, 513–2,048, 2,049–4,096, and 4,097–8,192.
Later bins are 8,193–16,384 and above 16,384 where available.
Actual rank allocation and filled KV accompany each layout.

A small exploratory audit is permitted if quality remains unresolved.
Such an audit cannot be called a completed real-workload campaign.
No unmeasured architecture speedup follows from theoretical KV capacity or ideal TP scaling.
The final report identifies every NOT_RUN configuration and its gate reason.

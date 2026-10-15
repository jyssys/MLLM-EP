# Timing contract

Only physical GPUs 4, 5, 6, and 7 execute kernels.
Scoped guards stop verified project burners, reject foreign work, and restore idle-device burners on every exit path.
No clock or power policy changes.

Quality and hardware use separate ledgers.
Quality has real EOS-enabled outputs and batch sizes up to eight. It supplies public accuracy, not B1 latency.
Hardware uses B1 frozen token IDs. Native R4 returns 64 forced output tokens for matched sequence shapes.
R1 component runs return four tokens. They measure one physical 48-layer stage and three actual cached forwards.
R1 is not a quality-preserving alternative to R4.

Clean E2E is client `LLM.generate` wall time after load, compilation, and warmup.
The prefill envelope measures wall time from scheduling to the first token.
Cached TPOT divides first-to-last-token duration by the number of cached intervals.
The teacher hook retains native logits and sampling work. It adds one frozen-token GPU copy in the hardware lane only.

Separate outer CUDA events measure each actual native forward.
The records are rank-local intervals, not globally aligned rank waiting.
An absolute observer difference above 10% invalidates fine timing attribution for that stratum.
No summed kernel service becomes an additive E2E percentage.

P2P measurements include send/receive completion, device synchronization, and the final coordination barrier.
Their host intervals use one clock per measured interval. Cross-GPU timestamps are never subtracted.
These are conservative physical components, not measured integrated-pipeline latency.

The DAG scheduler labels every forecast `CALIBRATED` or `ORACLE_NOT_ACHIEVABLE`.
It includes measured stage work, activation transfers, and stage resource/bubble constraints.
Native TP4 decode with a free cache transition is an optimistic bound, not an executable result.
Request-level bootstrap uses 5,000 draws. Chunks, layers, ranks, and seeds do not become independent requests.

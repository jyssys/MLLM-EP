# Latency and EP results

Status: `NOT_MEASURED_PRIMARY_LATENCY`.
Model and runtime identities appear in `00_runtime_audit.md`.
The current quality workers share system resources.
Their request wall times are not isolated primary latency results.

| Method | Group | B | Block | N isolated requests | Mean E2E | P50 | P90 | P95 | TTFT | Output tokens/s | Requests/s | Speedup | Paired 95% CI |
|---|---|---:|---:|---|---|---|---|---|---|---|---|---|---|
| P0 | A | 1 | 32 | NOT_MEASURED | NOT_MEASURED | NOT_MEASURED | NOT_MEASURED | NOT_MEASURED | NOT_MEASURED | NOT_MEASURED | NOT_MEASURED | NOT_MEASURED | NOT_MEASURED |
| P0-Hooked | A | 1 | 32 | NOT_MEASURED | NOT_MEASURED | NOT_MEASURED | NOT_MEASURED | NOT_MEASURED | NOT_MEASURED | NOT_MEASURED | NOT_MEASURED | NOT_MEASURED | NOT_MEASURED |
| P3 | A/B | 1 | 32 | NOT_MEASURED | NOT_MEASURED | NOT_MEASURED | NOT_MEASURED | NOT_MEASURED | NOT_MEASURED | NOT_MEASURED | NOT_MEASURED | NOT_MEASURED | NOT_MEASURED |
| P4 | A/B | 1 | 32 | NOT_MEASURED | NOT_MEASURED | NOT_MEASURED | NOT_MEASURED | NOT_MEASURED | NOT_MEASURED | NOT_MEASURED | NOT_MEASURED | NOT_MEASURED | NOT_MEASURED |
| P5 | A/B | 1 | 32 | NOT_MEASURED | NOT_MEASURED | NOT_MEASURED | NOT_MEASURED | NOT_MEASURED | NOT_MEASURED | NOT_MEASURED | NOT_MEASURED | NOT_MEASURED | NOT_MEASURED |

## E2E breakdown contract

| Component | Clean critical-path milliseconds | Clean E2E share |
|---|---|---|
| Attention | NOT_MEASURED | NOT_MEASURED |
| Router and top-k | NOT_MEASURED | NOT_MEASURED |
| Policy | NOT_MEASURED | NOT_MEASURED |
| Pack/scatter | NOT_MEASURED | NOT_MEASURED |
| Dispatch | NOT_MEASURED | NOT_MEASURED |
| Routed expert execution | NOT_MEASURED | NOT_MEASURED |
| Shared expert and dense computation | NOT_MEASURED | NOT_MEASURED |
| Combine | NOT_MEASURED | NOT_MEASURED |
| Rank waiting and synchronization | NOT_MEASURED | NOT_MEASURED |
| LM head and decode decision | NOT_MEASURED | NOT_MEASURED |
| Host and other overhead | NOT_MEASURED | NOT_MEASURED |

The observer records rank-local stage percentages separately in `ep_trace_audit.json`.
Those percentages do not form a validated clean global critical-path decomposition.
Do not force them to sum to 100% across overlapping ranks.

The clean benchmark must randomize variant order and perform A/B crossover.
It must use at least 30 repetitions per feasible timing condition.
It must record actual output lengths, NFE, peak HBM, and policy overhead.
It must distinguish native generation changes from matched-work hardware effects.

The batch grid is 1,2,4,8,16,32.
The block grid is 32,64.
Only the B1/block32 smoke path ran so far.
Other grid entries remain `NOT_RUN`, not OOM.

Two concurrent EP4 replicas must use the label `EP4×DP2`.
That throughput experiment remains `NOT_RUN`.

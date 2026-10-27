# Latency and EP results

Status: `ISOLATED_P0_MEASURED_CANDIDATE_REQUEST_SPEEDUP_NOT_MEASURED`.
Model and runtime identities appear in `00_runtime_audit.md`.
The completed quality runs shared system resources.
Their request wall times are not isolated primary latency results.

| Method | Group | B | Block | N isolated requests | Mean E2E | P50 | P90 | P95 | TTFT | Output tokens/s | Requests/s | Speedup | Paired 95% CI |
|---|---|---:|---:|---|---|---|---|---|---|---|---|---|---|
| P0 | A | 1 | 32 | 30 repeats, one prompt | 5924.739 ms | 5928.862 ms | 5953.122 ms | 5955.647 ms | NOT_MEASURED | 33.4192 | 0.168784 | 1.000× self | NOT_APPLICABLE |
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

Any candidate benchmark must randomize variant order and perform A/B crossover.
It must use at least 30 repetitions per feasible timing condition.
It must record actual output lengths, NFE, peak HBM, and policy overhead.
It must distinguish native generation changes from matched-work hardware effects.

The batch grid is 1,2,4,8,16,32.
The block grid is 32,64.
Only B1/block32 generation and operator conditions ran.
Other grid entries remain `NOT_RUN`, not OOM.

Two concurrent EP4 replicas must use the label `EP4×DP2`.
That throughput experiment remains `NOT_RUN`.

## Completed isolated natural-generation baseline

Evidence: `../results/refinement_aware_ep_expert_allocation/pilot/isolated_baseline_summary.csv`.
Individual observations appear in `latency_samples.csv` at the result-directory root.
One frozen validation prompt contains 1,477 input tokens.
Every repeat generates the same 198 output tokens with EOS and 154 model forwards.
Each group runs 33 natural requests and excludes the first three timing warm-ups.
This yields 30 measured repetitions per group, not 30 distinct questions.
Each group has one independent process start.

| Group | Physical GPUs | Repeats | Mean E2E ms | P50 ms | P90 ms | P95 ms | Std ms | Repeat-bootstrap 95% mean ms | Output tokens/s | Requests/s |
|---|---|---:|---:|---:|---:|---:|---:|---|---:|---:|
| A | 0–3 | 30 | 5924.739 | 5928.862 | 5953.122 | 5955.647 | 21.636 | [5917.188, 5932.305] | 33.4192 | 0.168784 |
| B | 4–7 | 30 | 6457.693 | 6176.276 | 6536.724 | 9203.360 | 965.635 | [6146.998, 6816.863] | 30.6611 | 0.154854 |

Evidence class: `MEASURED_WALL` isolated EP4 request.
The interval starts before prompt formatting/tokenization and ends after generated text decoding.
It includes the request-start rank barrier and native generation.
It excludes model load, warm-up, quiet-lane checks, and direct artifact writes.
It does not include HTTP transport or an external arrival queue.
Cross-rank artifact delays could still affect the request-start barrier.
No primary speedup comparison uses those potential delays.

Group B contains large generation-time outliers.
The three longest observed requests take approximately 9.495 s, 9.478 s, and 8.868 s.
Their generation regions account for most of that time.
Output tokens and NFE remain identical.
The experiment does not identify the cause of these outliers.
The table must not imply a fixed hardware-performance difference between A and B.
Repeat-bootstrap intervals are conditional on one prompt and one process start.
They do not establish population-level serving capacity.

Peak allocated HBM on rank 0 is 11.98754 GiB in both groups.
Per-rank raw results retain all four peak measurements.
Reserved memory and physical NVLink bytes remain separate unmeasured fields.
TTFT remains `NOT_MEASURED` because this harness does not expose a streaming first-commit timestamp.
The reported output throughput is serial natural-generation throughput, not open-loop serving capacity.

## Observer percentage breakdown

The 16-request observer uses a different, concurrent collection setup.
Its selected-rank wall denominator totals 112,999.52 ms.

| Observed region | Share of selected-rank observer wall |
|---|---:|
| Router linear region | 2.19544% |
| Dispatch region | 8.57484% |
| Routed expert region | 17.09546% |
| Combine region | 3.70053% |
| Residual other | 68.43374% |

These are not clean global critical-path percentages.
Residual other includes attention, shared experts, dense work, model head, decode, host gaps, and synchronization.
Those individual clean shares remain `NOT_MEASURED` in the earlier component table.
The median observer/baseline wall ratio is 1.16964.
Uniformly rescaling stage times would not remove observer overhead correctly.

## Policy performance boundary

P1/P2/P3 dense diagnostic timings are not sparse acceleration results.
P4/P5 have no measured request latency.
The separate physical operator ratios appear in `04_ep_cost_model.md`.
They cannot populate candidate E2E or SLO-goodput columns.
No measured 10% request improvement or 2% incremental joint EP gain is established.

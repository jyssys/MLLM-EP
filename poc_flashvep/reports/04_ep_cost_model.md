# EP cost model

Status: `PARTIAL_OBSERVER_ONLY`.
Runtime identity: existing BF16 dInfer/SGLang DeepEP normal bridge.
Model revision: `dad945cac317da394b390f82c7b40691d8a881ed`.
Evidence: `../results/refinement_aware_ep_expert_allocation/ep_trace_audit.json`.

The fresh sanity trace measures router, dispatch, routed expert, and combine intervals.
It records these intervals separately for each GPU rank.
It also records model forward spans and request wall time.
Heavy tracing can change those times.
The current trace is not the final clean latency benchmark.

## Sixteen-request observer breakdown

The table pools the selected longest-wall rank from each of 16 validation requests.
The denominator is 112,999.52 ms of observed rank wall time.
The two EP4 replicas ran concurrently.
These are observer-run shares, not clean global critical-path shares.

| Observed region | Share of pooled selected-rank wall time |
|---|---:|
| Router linear operation | 2.19544% |
| DeepEP dispatch region | 8.57484% |
| Routed expert region | 17.09546% |
| DeepEP combine region | 3.70053% |
| Residual other time | 68.43374% |

Residual other time includes attention, shared experts, dense work, LM head, decode, host gaps, and synchronization.
The trace does not separate all these costs yet.
The observer-to-baseline median wall ratio was 1.16964 across N=16 requests.
Do not remove that overhead with a uniform stage-time rescaling.
Do not interpret a fraction of routed edges as the same fraction of E2E savings.
Source: `pilot/observer_breakdown.csv` in the campaign result directory.

## Accounting boundary

Per-rank stage time does not equal the global critical path.
Communication and rank waiting can appear inside an observed stage interval.
The analysis must not add times from four ranks into one request latency.
It must not treat logical payload as physical fabric traffic.

| Cost component | Current evidence | Calibrated deployable estimator |
|---|---|---|
| Router | Per-rank CUDA event intervals | NOT_MEASURED |
| Dispatch | Real DeepEP dispatch and receive buffers | NOT_MEASURED |
| Routed experts | Existing BF16 expert runner intervals | NOT_MEASURED |
| Combine | DeepEP completion dependency and event intervals | NOT_MEASURED |
| Pack/scatter | Included in broader runtime regions | NOT_MEASURED_SEPARATELY |
| Rank waiting | Included in observed regions | NOT_MEASURED_SEPARATELY |
| Attention and LM head | Included in forward span | NOT_MEASURED_SEPARATELY |
| Policy scoring and compaction | No sparse policy executes yet | NOT_MEASURED |
| Physical NVLink bytes | No hardware counter captured | NOT_MEASURED |

## Required withheld validation

Fit expert time across actual received row counts and expert geometry.
Measure startup cost separately from payload-dependent dispatch cost.
Validate timing estimates on different calibration traces.
Include source-row all-gather and any new policy synchronization.
Charge irregular kernel shapes and policy preparation costs.
Do not assume time scales linearly with the number of removed edges.

The calibrated EP cost gate remains pending.
The current data cannot establish a conservative 5% E2E opportunity.

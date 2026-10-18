# Stage 2: native CDB cost model

The cost model uses real host spans and CUDA-event spans from calibration requests.
It fits separate prefill, recurrent, and coda costs.
The recurrent cost uses active batch size, context volume, and terminal-KV copy activity.
The fit also includes a common measured host-loop residual.
It does not use a constant time per request-depth unit.

The simulator preserves slot-homogeneous kernel batches, survivor priority, coda priority, and pinned KV ownership.
It includes full-depth prefill and finite resident/KV capacity.
It does not migrate running requests or independently redistribute their tokens.

Burst transition counts matched physical execution in 28/28 rank cases.
That check covers prefill, recurrent, and coda launch counts.
Transition-count parity does not prove timing accuracy.

## Held-out live timing validation

| Metric | Median RR/queue relative error |
| --- | --- |
| E2E_P50_ms | 1.093% |
| E2E_P95_ms | 1.861% |
| ITL_P95_ms | 1.310% |
| TTFT_P95_ms | 4.047% |
| makespan_ms | 2.004% |

The overall status is `VALIDATED` for the exploratory cost model.
The primary P95 E2E and ITL estimates meet the approximate median-error target.
The complete per-case errors remain in `cdb_cost_model.json`.
Some TTFT strata exceed 10% error:

| Case | P95 TTFT prediction error |
| --- | --- |
| clean_burst_C32_round_robin | 14.98% |
| clean_burst_C32_queue | 10.29% |
| clean_burst_C64_round_robin | 11.66% |

Those TTFT strata are `PARTIAL`. We do not promote their detailed TTFT oracle estimates.
The model-validation result is not a guarantee for unseen context lengths or workloads.

## Observer and cost surfaces

The eight-request B1 trace observer added 0.498% mean decode time.
Its paired request-bootstrap CI is [0.362, 0.588]%.
All eight token outputs and depth sequences matched.
The paired C64 burst had 64/64 exact outputs between trace and clean lanes.
Its median E2E difference was 0.700%.
That is one paired system run, not 64 independent observer repeats.

The 1K/4K/8K context sweep uses repeated synthetic prompts on physical GPU7.
It tests B1/B2/B4/B8/B16 within the explicit KV budget.
B32 and capacity-exceeding combinations remain documented skips.
These samples show physical batch/context costs, not task quality or serving gains.
The task model does not extrapolate its short-context validation to the full synthetic grid.

Fixed-R4, adaptive-refill, and adaptive-no-refill diagnostics use the same eight calibration prompts.
The adaptive arms retain the same q and KV policy.
Fixed R4 has a different quality operating point. It is only a contextual work-cost control.
See `outputs/ablation_*` and `artifacts/ablation_summary.json`.

# B repair and separate execution epoch

Status: FINALIZED_WITH_LIMITATIONS.
The original output failures remain unchanged. They are not accepted serving speedups.
This report separates later diagnostic changes and their qualification.
The original15-hour deadline and frozen SLOs still apply.
Checkpoint weights, BF16 tensor precision, and fixed recurrence remain unchanged.

## Completed expansion stages

| Model | Requests | Output cap | Control/B exact | Control repeat exact | Stage pass |
| --- | --- | --- | --- | --- | --- |
| ouro | 7 | 126 | 7/7 | 7/7 | True |
| ouro | 8 | 128 | 8/8 | 8/8 | True |
| ouro | 32 | 128 | 32/32 | 32/32 | True |
| ouro | 200 | 128 | 200/200 | 200/200 | True |
| huginn | 33 | 128 | 33/33 | 33/33 | True |
| huginn | 200 | 128 | 199/200 | 198/200 | False |

A small passing stage is not a full repair acceptance. Counts measure generated sequences, not answer accuracy.

## Model-level qualification

| Model | Epoch | Qualified | KV exact/N | Full-depth counter checks | Failure reason |
| --- | --- | --- | --- | --- | --- |
| ouro | canonical_core_linear_bf16_v1 | True | 8/8 | [True, True, True] |  |
| huginn | canonical_core_linear_bf16_v1 | False | NOT_MEASURED/NOT_MEASURED | NOT_MEASURED | Full200-request output/repeat qualification failed. See the expansion table. |

Qualification requires200 original W1 requests, an identical-control repeat, stop parity, and fixed-depth work counters.
It also requires reversed-fragment prompt-KV and final-hidden byte checks with the same repair.
The source hash must match all full qualification runs.
A failed model does not disable the other model.

## Reference and baseline-shift limitations

| Model | Comparison | Exact generated sequences |
| --- | --- | --- |
| ouro | Original control vs canonical control | 142/200 |
| ouro | Canonical Native vs saved official HF | 7/8 |
| huginn | Original control vs canonical control | NOT_MEASURED |
| huginn | Canonical Native vs saved official HF | NOT_MEASURED |

Canonical geometry can change BF16 rounding and generated outputs relative to the original execution.
Internal B/control parity does not establish output parity with the pre-repair baseline or official HF.
The eight-prompt official-reference check is diagnostic only. It is not downstream accuracy.
No original-model-lossless performance claim follows from the repair qualification.

## Separate-epoch decision scope

| Model | GPUs | Verdict | Internal qualification | Serving runs |
| --- | --- | --- | --- | --- |
| ouro | 2 | NO_GO_CORRECTNESS | True | 6 |
| ouro | 4 | NOT_MEASURED_TIME_BUDGET | True | 0 |
| ouro | 8 | NOT_MEASURED_TIME_BUDGET | True | 0 |
| huginn | 2 | NO_GO_CORRECTNESS | False | 0 |
| huginn | 4 | NOT_MEASURED_TIME_BUDGET | False | 0 |
| huginn | 8 | NOT_MEASURED_TIME_BUDGET | False | 0 |

A conditional verdict is not proof of a performance gain or original-model losslessness.
No new4/8GPU result is inferred from a two-GPU repair.

## Actual separate-epoch serving

| Model | Arm | Offered req/s | Seed | Completed/planned | Joint SLO % | Window goodput req/s | Queue scope |
| --- | --- | --- | --- | --- | --- | --- | --- |
| ouro | B | 1.0 | 2026106001 | 400/400 | 80.25 | 0.7928394073060304 | NO_PERSISTENT_GROWTH_OBSERVED |
| ouro | B | 1.0 | 2026107001 | 400/400 | 68.25 | 0.7068961686563188 | NO_PERSISTENT_GROWTH_OBSERVED |
| ouro | DP_Mixed | 1.0 | 2026106001 | 400/400 | 43.75 | 0.43223332174004775 | NO_PERSISTENT_GROWTH_OBSERVED |
| ouro | DP_Mixed | 1.0 | 2026107001 | 400/400 | 53.25 | 0.5515343733472378 | NO_PERSISTENT_GROWTH_OBSERVED |
| ouro | PD_G1 | 1.0 | 2026106001 | 400/400 | 87.25 | 0.861996738784438 | NO_PERSISTENT_GROWTH_OBSERVED |
| ouro | PD_G1 | 1.0 | 2026107001 | 400/400 | 74.5 | 0.7716302500351025 | NO_PERSISTENT_GROWTH_OBSERVED |

These rows never enter the original-epoch capacity table.
Latency and transfer CSVs identify the execution epoch for each run.
All three serving arms use the same canonical execution changes and physical GPU pair at each point.
The second independent seed uses a different GPU pair and a rotated arm order.
Rates and budgets are declared before the new epoch sees serving outcomes.
No SLO threshold changes. No pre-repair and post-repair samples are pooled.

Canonicalization can add GPU work. Its overhead belongs to every new-epoch comparator.
A gain over canonical controls is not automatically a gain over the unmodified optimized engine.
Pre/post E2E ratios are not equal-work speedups when generated outputs differ.

## Serving output gate

| Model | Offered req/s | Seed | Control/B exact | All outputs exact |
| --- | --- | --- | --- | --- |
| ouro | 1.000 | 2026106001 | 400/400 | True |
| ouro | 1.000 | 2026107001 | 399/400 | False |

A new mismatch closes that model's new performance gate. It does not erase its recorded failure.

## Observed paired goodput, not an accepted lossless speedup

| Model | Offered req/s | Baseline | Independent starts | Goodput ratio | 95% CI | Exact outputs |
| --- | --- | --- | --- | --- | --- | --- |
| ouro | 1.000 | DP_Mixed | 1 | 1.834× | NOT_MEASURED_SINGLE_START | False |
| ouro | 1.000 | PD_G1 | 1 | 0.920× | NOT_MEASURED_SINGLE_START | True |

The bootstrap resamples whole paired starts. It does not treat tokens as independent requests.
Finite rate grids do not establish an unrestricted sustainable maximum.
Between-start uncertainty needs multiple completed starts. A single point has no such confidence interval.
No original-model-lossless GO is established. Official-HF fidelity remains unresolved.
Generic immutable-layer streaming is not independently measured. Loop-exclusive novelty remains unresolved.
Read `b_repair_diagnosis.md`, `b_repair_ablation_summary.csv`, and the per-run source snapshots.

# Dynamic Cut Headroom: What the Evidence Establishes

Overall status: CONDITIONAL. The study did not qualify an online-policy expansion.
No validated dynamic serving oracle or stable-goodput result exists here.

## Source, models, and resource contract

The study uses the numerical release `3b7a722898dab5f6dd797b474c1ff461659d8fa1`.
It inherits RCC release `b53a8fed36a1706ac74754b53753f20edb8b913e`.
The initial implementation checkpoint is `28b1cbf75b0b26a758e98e26c60af9860e7c498e`.
Code and raw-evidence release: `71453683a69d600a7cabc667386df4dd1b622f7a`.
Every run preserves its exact overlay archive and pinned model revision.
The final summary records the final implementation revision.

Ouro executes BF16 R4. Huginn executes BF16 R32.
The new migration path preserves the full recurrent work.
The primary maps disable Native Mixed and Prefill/Decode GEMM fusion.
Clean timing uses one isolated 1P+1D pair.
The configurations cross GPU0/1, GPU4/5, and GPU6/7.
Four-GPU and eight-GPU serving are NOT_RUN.

## The completed analysis is not a Dynamic Oracle

Stage 1 executes complete fixed-cut cohorts with their actual contention.
The analysis never selects independently minimal request latencies within a cohort.
It can select one complete measured cohort per homogeneous load profile.
That operation retains the observed within-cohort queueing effects.
It does not model transitions between profiles or live per-request cut decisions.

The evidence label is `OFFLINE_FIXED_REGIME_COMPOSITION`.
It is neither a simulated dynamic oracle nor a measured adaptive serving result.
It is not an upper bound on all possible Dynamic Cut policies.

The loaded-D profiles initialize two background requests with Full Cut R.
They release target requests only after both background requests generate four tokens.
Both background requests must remain active at release.
The background cut stays constant across target-cut comparisons.
These profiles are state-triggered mechanism probes, not Poisson capacity workloads.

## Frozen selection rule

The rule uses start 0 only.
It minimizes mean all-request E2E with equal weights for three load profiles.
One selector chooses a single fixed target cut.
The other selector chooses one target cut for each known experimental profile.
Neither selector uses held-out outputs to choose its cuts.

| Model | Frozen static target cut | P-pressure target cut | Balanced target cut | Loaded-D target cut |
|---|---:|---:|---:|---:|
| Ouro R4 | 1 | 2 | 3 | 4 |
| Huginn R32 | 32 | 28 | 32 | 16 |

| Calibration composition | Static mean E2E ms | Selected mean E2E ms |
|---|---:|---:|
| Ouro R4 | 3,779.902 | 3,630.312 |
| Huginn R32 | 10,994.586 | 10,900.241 |

These values describe calibration. They are not held-out gains.

The rule remained frozen for independent starts 1, 2, and 3.
The held-out starts change the request seed and server start.
They do not provide new prompt domains. The diagnostic prompt IDs remain fixed.
The source records that this analysis was not a preregistered main SLO evaluation.

## Held-out fixed-regime composition

Positive values mean lower composed mean E2E than the calibration-selected static target cut.
The confidence interval resamples independent starts, not tokens.

| Model | Held-out starts | Mean E2E reduction | Paired-start bootstrap 95% CI | Evidence class |
|---|---:|---:|---|---|
| Ouro R4 | 3 | 4.98% | [0.90%, 8.24%] | OFFLINE_FIXED_REGIME_COMPOSITION |
| Huginn R32 | 3 | -3.62% | [-4.49%, -2.81%] | OFFLINE_FIXED_REGIME_COMPOSITION |

Ouro shows a modest benefit in this restricted composition.
The individual reductions were 8.51%, -0.86%, and 7.29% after rounding.
Huginn's calibration-selected load-dependent cuts did not generalize to these independent starts.
The small calibration difference did not provide a reliable cut-selection advantage.

The static comparator is calibration-selected. It is not a proven global optimum.
The complete cut maps preserve stronger alternative fixed-cut comparisons.
The intervals are conditional on the frozen diagnostic prompts.
Three starts cannot establish benchmark-population uncertainty or stable serving capacity.

## Primary-metric gate

The frozen Stage 2 target requires credible goodput headroom before online-policy expansion.
The saved contract sets a 10% threshold above an eligible tuned static control.
The primary metric is stable TTFT-plus-request-average-TPOT goodput.
The composition above measures mean E2E instead.
Its confidence interval must not be compared with a goodput threshold as if the metrics were identical.

No main W1 SLO calibration or stable open-loop rate sweep ran.
No measured 90%-attainment maximum request rate exists.
The primary expansion gate therefore did not pass.
This is not proof that every possible Dynamic Cut policy fails.
It is a decision not to expand the present prototype without stronger evidence.

## Why a queue simulator is not reported

The host Prefill interval includes worker queueing, CPU scheduling, CUDA submission, and GPU work.
It is not an isolated GPU service time.
The profile traces also include instrumentation overhead.
Their kernel-duration sums are not E2E critical-path durations.
Using either quantity directly as clean service time would double-count or omit contention.

A valid simulator must model both GPU resources and the CPU submission path.
It must also model NIXL submission limits, transfer completion, and KV admission credits.
It must preserve request dependencies and Decode interference.
It needs out-of-sample validation against unseen fixed-cut traces.
That model and validation are NOT_RUN.
No future-arrival oracle is presented as an online-achievable result.

## Granularity and novelty gate

A generic logical-layer action space contains the matching recurrent boundaries under identical constraints.
At a matching boundary, the same weights, tensors, and transfer protocol produce the same mechanism.
This case is `EQUIVALENT_MECHANISM`.
It does not establish a new partition axis.

The study did not measure a practical control-cost advantage over internal logical-layer migration.
It did not implement Token-Dynamic or Layer-Dynamic.
Thus it cannot establish Loop-specific superiority over those stronger alternatives.
The present evidence does not support a GO for a new Loop-specific serving method.

## Actual data and reproduction

The full fixed-cut metrics are in `adaptive_recurrence_cut/static_cut_metrics_by_start.csv`.
The frozen selector is in `adaptive_recurrence_cut/configs/frozen_regime_selections.json`.
The held-out composition and uncertainty have separate CSV files.
The static-map report documents TTFT, TPOT, ITL, E2E, memory, and transfers.
The online-policy report identifies unimplemented admission and fallback behavior.
See `adaptive_recurrence_cut/reproduce.md` for actual commands.

Original-HF fidelity and downstream accuracy remain separate from internal output/KV parity.
Neither lossless original-model equivalence nor an online SLO guarantee is claimed.

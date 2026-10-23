# Optimizer cost and budget audit

Evidence classes differ in this document.
CPU timers are LIVE_MEASURED host observations.
Ready-stream intervals are separate CUDA-event observations.
Neither quantity is automatically a non-overlapped E2E critical-path share.

## Selector implementation

The selector enumerates the existing discrete legal candidates.
It builds features on the full common candidate universe before restricting C-only or G-only.
No binary solver, beam search, learned policy, or new CUDA kernel runs online.
The shared tie rule preserves exact equality when the selected C surface agrees.
The causal telemetry uses delivered outputs from completed native engine steps.
It does not use future arrivals, future execution times, or future output tokens.

The host timer starts before candidate enumeration and ends after selection and trace construction.
Admission checks have a separate timer around readiness and feasibility queries.
Selected-action scoring occurs outside that admission timer in the native scheduling path.
The tables preserve both timers separately.
They do not subtract their sum from serving wall time as an achievable gain.

## Fresh validation CPU measurements

The table reports the mean of 16 cohort statistics per policy.
There are four independent process starts per width and two cohorts per process.
P95 here means the mean of each cohort's selector-call P95, not a pooled-call percentile.

| Policy | Selector P95 ms | Selector total / drained wall % | Admission total / drained wall % |
| --- | ---: | ---: | ---: |
| O6 Joint |0.399 |0.0557 |0.1303 |
| O6 G-only C2048 |0.369 |0.0469 |0.1217 |
| O6 G-only C1024 |0.391 |0.0655 |0.1390 |
| O6 C-only |0.460 |0.0842 |0.1355 |
| O2 Joint |0.375 |0.0628 |0.1468 |
| O2 G-only C2048 |0.374 |0.0469 |0.1230 |

The selector is not the dominant measured wall-time cost in these validation cohorts.
This does not explain every latency difference or identify a removable GPU bottleneck.
FINAL timers and their source rows appear in `optimizer_overhead.csv`.

## CUDA-event cost audit

Four CALIBRATION jobs cross widths4/8 and budgets40/120ms.
The observer records start events after native dependency waits and end events after native execution submission.
All tensor inspection waits for native synchronization.
These intervals include preparation and host-submission gaps. They are not pure kernel service sums.
The largest recorded host/device alignment bound is 0.1412ms.

There are 770 matched selected-decision/service units.
Of these,498 have empirical cost-shape support. The remaining 272 use enclosing/inflated shapes.
The observer is not a blind independent test of all possible actions.
It measures selected actions on CALIBRATION inputs only.
It does not establish a calibrated future-token SLO probability.

| Budget / policy / recurrent path | Units | MAE ms | Absolute-error P95 ms | Measured duration exceeds B |
| --- | ---: | ---: | ---: | ---: |
|120 / Joint |55 |6.733 |8.093 |0/55 |
|120 / G-only |51 |7.098 |8.511 |0/51 |
|120 / C-only |38 |7.105 |12.779 |0/38 |
|40 / Joint |64 |7.314 |8.016 |3/64 |
|40 / G-only |60 |6.158 |7.454 |2/60 |
|40 / C-only |201 |2.346 |9.356 |201/201 |

The 40ms C-only path cannot fit any full grouped recurrent quantum under the frozen cost model.
It uses the explicitly recorded progress fallback.
Its201 recurrent budget overruns are not concealed as compliant work.
A Joint advantage over that infeasible control would not establish incremental C value.
The primary FINAL budget remains120ms for all matched adaptive surfaces.

Prelude is a separate G=0 action with no completed-prefill token credit.
Its low service time must not replace the cost of its planned following recurrent group.
The raw table records actual G and planned G separately.

## Observer overhead limit

Observer ON/OFF token IDs match on 87/96 pairs.
Prompt plans match on 95/96 pairs, but decode batch geometry is not fully replayed.
The recorded wall ratios can therefore include changed causal plans and output lengths.
They are not pure observer-overhead estimates.
Clean serving never uses these event observers.

Data: `optimizer_overhead.csv`, `event_budget_residuals.csv`, `observer_execution_diagnostics.csv`, and `audit_summary.json`.

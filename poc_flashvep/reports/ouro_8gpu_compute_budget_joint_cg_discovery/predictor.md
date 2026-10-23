# Path-conditioned service model

The frozen v1 model predicts a native ready-stream service interval.
It does not predict a future output-token completion time.
The old E1 and D1 models are not used.

The fresh dataset contains 298,842 native quantum records from 1,056 observer cohorts.
TRAIN contains 86,855 records. VALIDATION contains 71,140 records.
CALIBRATION contains 70,746 records. NEW_HELDOUT contains 70,101 records.
These totals include decode and mixed paths, not only prediction targets.

The deployed lookup uses single-prompt, non-mixed samples.
Its 334 cells condition on path, actual rows, G, depth frontier, and context bucket.
It uses TRAIN medians and CALIBRATION positive-residual P95 margins.
The model artifact froze before NEW_HELDOUT evaluation.
FINAL_ONLINE families do not train the lookup or its margins.

| Blind path | Target quanta | Model-process starts | Source families | Median-model MAE, ms | Upper-estimate exceedance, % [block 95% CI] |
| --- | ---: | ---: | ---: | ---: | ---: |
| Atomic | 674 | 16 | 3 | 9.613 | 5.786 [2.343, 8.646] |
| Staged Core | 7,301 | 16 | 3 | 2.769 | 3.862 [2.237, 6.578] |
| Staged Prelude | 3,370 | 16 | 3 | 0.797 | 4.481 [3.097, 8.960] |

The uncertainty unit is a model-process block, not a quantum.
Only three source families contribute single-prompt blind targets.
Large quantum counts do not remove this limitation.
Native Mixed remains a measured baseline, not a deployed mixed-row predictor.
Decode timings remain diagnostic and do not become a D1 deadline predictor.

The model does not establish a universally valid 5% risk bound.
Online policies therefore retain the EXPERIMENTAL_BUDGET_RISK label.
Unknown shapes use an explicit inflated empirical fallback.
The policy records unknown shapes, prediction errors, and actual overruns separately.

The measured CPU lookup query P50 was 0.663 microseconds.
Its P99 was 16.991 microseconds across 3,000 repeated validation-state queries.
This measures a model query, not the entire selector or admission filter.
Live selector and admission overhead remain in measured request latency.

The model does not condition on every recorded feature.
Graph transitions, issued-but-pending work, thermal state, and CPU contention can still affect errors.
The retained residual tables expose these limits without an aggregate decode-coverage shortcut.

Sources: `cost_model_artifacts/`, `predictor_heldout.csv`, `prediction_residuals.csv.gz`, and `predictor_false_feasible.csv`.

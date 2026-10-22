# Runtime and completed-token predictors

E models predict scheduling-impact duration for a native quantum.
D models predict time until the next completed output token under the recorded stock policy.
A recurrent-step completion never substitutes for output-token completion.

`predictor_protocol.json` froze the menu, selection rule, margins, and thresholds before the first fit.
Source families, arrival cohorts, and process starts remain partitioned.
TRAIN fits models, VALIDATION selects them, and CALIBRATION supplies safety residuals.
HELDOUT receives the blind test.
FINAL_ONLINE remains unused.
Artifacts record selected models before test metrics are computed.

The menu is a constant, binned median lookup, and ridge regression.
Selection minimizes process-group-weighted validation MAE.
The simpler model wins within 5% of the minimum validation MAE.
Both selected models are lookups: E1 and D1.

| Target | Held-out observations / starts | MAE | Positive error p95 / p99 | Calibration margin | Bound coverage | Mean query cost |
| --- | --- | --- | --- | --- | --- | --- |
| Quantum E1 | 134,113 / 24 | 1.935 ms | 1.075 / 31.789 ms | 24.241 ms | 98.861% | 3.701 µs |
| Next completed token D1 | 367,636 / 24 | 17.311 ms | 86.970 / 235.384 ms | 230.204 ms | 98.897% | 0.561 µs |

The quantum MAE 95% interval is [1.009, 3.313] ms.
The full-token MAE interval is [12.830, 21.190] ms.
Coverage intervals are [97.484%, 99.729%] and [98.438%, 99.334%].
Bootstrap intervals resample 24 process-start groups with 2,000 draws and seed 20261023.
Quanta are not independent trials.
These intervals condition on the frozen source families, which recur across C choices.
They do not estimate uncertainty over the whole prompt population.
Query costs are analysis-process CPU measurements, not selector-in-loop overhead.

Expected predictions underpredict 40.766% and 47.187% of labels, respectively.
That sign statistic differs from safe-bound violations and false-feasible action selection.
No live false-feasible selection rate is measured.

E features include native C/G, rows, context, path, graph state, staging bytes, and available host/queue indicators.
D features include request phase, remaining issued depth, current queues, graph state, and past token clocks.
The selected D1 key uses only a subset of that feature menu.
Feature functions exclude future outputs, completion labels, and future arrivals.
Earlier observer records retain missing-feature flags.

`predictor_artifact/` contains all models and frozen selections.
`predictor_blind_results.json` includes all models and shape/path/schema strata.
Compressed data tables preserve source-family IDs and run IDs.
`predictor_dataset_sources.json` preserves original-file hashes.
The canonical fitting command is `fit_models`, not the earlier exploratory entry points.

These models describe stock-policy behavior, not changed-policy rollout.
D1 labels include future stock-policy interruptions and pending GPU work.
Adding a proposed quantum to D1 can double-count existing exposure.
A candidate-specific dependency schedule remains necessary.
Aggregate coverage alone does not pass the online gate.

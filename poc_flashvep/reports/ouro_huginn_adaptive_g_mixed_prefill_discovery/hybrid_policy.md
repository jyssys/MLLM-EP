# Hybrid policy status

The implemented selector is the fixed event hybrid control.
If compatible prompt and decode work is ready, it selects native Mixed G1.
Otherwise it permits the registered fixed pure prompt G.
The executor clips the common group at every selected row's remaining depth.
It never applies a grouped prompt descriptor to unrelated decode rows.

`selector_candidates.csv` contains available fixed-selector telemetry.
The pre-native selector time excludes the native policy call.
The native policy interval remains a separate measurement.
Neither CPU sum becomes an exclusive E2E saving.

OURS_ADAPT_G_MIXED and ADAPT_G_NO_MIX remain NOT_RUN.
The experiment does not relabel fixed event switching as a new adaptive G method.
No future arrivals, future tokens, or future GPU service times enter a live decision.
No broad scalar-objective or predictor search follows the failed incremental screen.

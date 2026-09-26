# Simulator held-out validation

Verdict: **SIM_UNVALIDATED**

Validation is request-level, not a random layer-row split.

| Metric | Result | Operating target |
|---|---:|---:|
| Held-out requests | 4 | 8 minimum |
| Held-out layer observations | 192 | descriptive |
| Median APE | 15.705% | ≤10% |
| P90 APE | 46.817% | ≤20% |
| Signed bias | +0.267 ms | report only |
| Material-gain sign accuracy | NOT_AVAILABLE | ≥90% |
| Configuration Spearman | NOT_AVAILABLE | ≥0.8 |

The model misses both latency error targets, and the one-baseline cohort cannot
establish configuration ranking or gain sign. The result is not repaired with
an arbitrary correction scalar.

Permitted use:

- structural EP4/EP8 histograms, traffic, fanout, and OOD detection;
- identifying additional real shapes to measure.

Prohibited use:

- declaring a small method speedup or NO-GO from predicted milliseconds;
- presenting EP8 as physical validation;
- inferring a new resolution/top-k trajectory from the baseline trace.

Next calibration must collect at least 16 image-disjoint calibration and 8
held-out requests and include multiple real configuration/material-gain pairs.
Evidence: `artifacts/analysis/heldout_validation.json`.

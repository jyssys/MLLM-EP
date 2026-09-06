# Tail definition

Primary normality is defined within matched `(run, phase, layer, M)` groups.
Normal: `<=p95`; moderate: `p95--p99`; severe: `p99--p99.9`; extreme:
`>p99.9` and normalized latency at least 10x the group median; giant: at least
100x or 100 ms. Secondary thresholds are >5, >10, >20, >50, >100 ms.

The prior three-run stock population (89,664 logical invocations) has matched-
p50 positive excess mass 28.14% of MoE time and >20 ms excess mass 12.10%.
The new direct contextual run has 6,912 logical invocations (4,896 prefill,
2,016 decode); its >20 ms rate is 0.0723% and matched-p50 excess mass 2.95%.
See `analysis_direct_run4/direct_analysis.json` for machine-readable values.

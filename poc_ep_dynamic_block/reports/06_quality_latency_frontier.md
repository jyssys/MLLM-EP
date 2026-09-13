# Fixed-B quality-latency frontier

QUALITY_LATENCY_FRONTIER.csv includes all serving-native thresholds rather than
only threshold 0.9. This prevents a mixed schedule from being credited for
beating an untuned fixed baseline.

## Reproducible key points

- GSM8K B32/mini32/t0.9: 9.453 s, 13/32, three restarts.
- GSM8K B64/mini16/t0.9: 11.725 s, 14/32, three restarts.
- HumanEval B32/mini16/t0.85: 13.881 s, 18/32, three restarts.
- HumanEval B8/mini16/t0.85: 20.167 s, 19/32, three restarts.

The official B32 remains a strong Pareto point, but not the only one: B64 buys
one GSM8K success at higher latency, while B8 buys one HumanEval success at
substantially higher latency. The bounded 32-sample score is reported as a
coarse diagnostic, not a population accuracy estimate.

The key successor gate is stronger: a dynamic trajectory must improve this
whole fixed-B/mini/threshold frontier, not merely fixed B32 at threshold 0.9.

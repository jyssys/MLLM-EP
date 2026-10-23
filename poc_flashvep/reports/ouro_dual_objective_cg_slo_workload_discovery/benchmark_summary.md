# Benchmark summary

Verdict: NO_GO_INCREMENTAL for the tested dual-objective Joint formulation.
All research jobs use independent single-GPU Ouro-1.4B Base workers, BF16, Full R4, on physical GPUs4–7.
These results are not TP/PP scaling results.

| Serial workload | Joint / matched G2048 SLO goodput | Joint / G2048 TTFT P95 | Requests per arm | Independent starts |
| --- | ---: | ---: | ---: | ---: |
| MIXED |0.9921× |1.5335× |36 |3 |
|8K stress |0.9939× |1.1155× |36 |3 |

The primary SLO is TTFT≤5s and every delivered ITL≤200ms with a complete stop.
Goodput uses drained cohort wall time. It is not qualified sustainable capacity.
Latency ratios use means of cohort quantiles.
The full paired intervals, seven-arm serial table, and nine-arm characterization appear in [phase_c_performance.md](phase_c_performance.md).

Joint and matched G-only score47/64 GSM8K and32/50 extracted-answer MATH.
Completed-and-matching MATH answers total30/50 in both arms.
Their correctness labels match on114/114 questions, while token IDs match on112/114.
The observed0pp difference does not prove population quality equivalence.
All nine quality configurations, cap counts, parsing failures, and paired intervals appear in [quality.md](quality.md).

The main program completes888 cohorts and7,200 requests.
A separate observed budget supplement completes30 cohorts and180 requests.
The quality pilot completes1,026 generations, not1,026 independent questions.
All99 completed GPU attempts pass the hash/device audit. Four failed initial smoke attempts remain preserved.

The cause analysis and final research boundary appear in [analysis.md](analysis.md).
Reproduction commands and exact source locks appear in [reproduce.md](reproduce.md).
Machine-readable results are `main_table.csv`, `paired_comparisons.csv`, and `quality_matched_controls.csv`.
No full1,819-question expansion or qualified1,000-request-per-arm/rate capacity sweep runs after promotion fails.

# Benchmark summary

## Evidence lanes

| Lane | Meaning | Valid claim |
| --- | --- | --- |
| Historical | Frozen commit `c458f6a2a764f667a0da6784bcb2592b36626837` | Historical fixed-quantum trade-off |
| New continuity | Counterbalanced bounded128 natural generation | Same observed output count and static depth, not full-answer quality |
| New open-loop | Scheduled Poisson/burst arrivals and actual natural output | Measured offered-request outcomes, with EOS/cap and length confounding |
| Heavy profile | CUDA events and separate Nsight targets | Diagnostic stream/kernel timing, not headline clean latency |
| Numerical | Isolated hidden/logit/KV comparisons | Component correctness, not benchmark accuracy |

The dataset, model, and SLO registrations are separate from candidate outcomes.
The fixed-grid section below is complete.
The open-loop queue and final diagnostic controls are complete.
The completeness audit checks all 71 distinct jobs required by the main manifests.
The final serving table contains 176 measured policy runs.

## Fixed P≈8K/N8 shortlist

All values are `LIVE_MEASURED`.
Each row uses ten cohorts across six independent process blocks.
TTFT and tail entries are means of cohort statistics.
Throughput is the mean drained-cohort output rate, including background priming.
It is not steady open-loop capacity.
The actual long prompt has 8177 tokens.
Every request emits 128 capped tokens in this lane.

| Policy | Mean long TTFT, ms | Mean cohort-p95 background ITL, ms | Mean cohort-p99 background ITL, ms | Mean drained output tok/s | Staged hidden, MiB |
| --- | ---: | ---: | ---: | ---: | ---: |
| Atomic C256 G4 | 3084.564 | 372.563 | 409.359 | 126.895 | 0 |
| Atomic C1024 G4 | 971.261 | 38.845 | 388.869 | 170.262 | 0 |
| Grouped C512 G2 | 1970.514 | 174.719 | 188.792 | 142.892 | 16 |
| Grouped C1024 G2 | 1027.939 | 130.377 | 189.721 | 166.110 | 32 |
| Split C512 G1 | 2161.498 | 108.342 | 134.537 | 139.949 | 16 |
| Mixed C128 G1 | 5799.701 | 94.669 | 99.922 | 98.645 | 4 |
| Mixed C512 G1 | 1715.895 | 106.583 | 110.277 | 153.536 | 16 |
| Staged C1024 G4 | 1077.439 | 40.515 | 273.991 | 166.017 | 32 |

The physical KV reserve is 54 GiB for every ordinary new configuration.
These staged-memory values do not include all graph/workspace or HBM overhead.
The full P/N/C/G surface and per-mean intervals are in `fixed_grid_summary.csv`.
Matched comparisons are in `paired_comparisons.csv`.

## Paired effects of G2 C1024

Positive reduction means lower latency.
Intervals use 5,000 paired process-block/cohort bootstrap draws.

| Reference | Metric | Reduction, % | 95% paired CI |
| --- | --- | ---: | --- |
| Atomic C1024 | Mean long TTFT | −5.836 | [−15.182, 9.141] |
| Atomic C1024 | Mean cohort-p99 ITL | 51.212 | [45.036, 59.925] |
| Atomic C1024 | Drained-cohort time | −2.159 | [−5.375, 3.174] |
| Staged G4 C1024 | Mean cohort-p99 ITL | 30.756 | [21.428, 43.868] |
| Mixed C512 | Mean cohort-p99 ITL | −72.041 | [−89.402, −42.941] |

G2's p99 benefit over Atomic does not imply an E2E or throughput gain.
Stock Mixed remains a stronger decode-tail control in this comparison.
The same metric cannot silently change from mean cohort quantiles to pooled quantiles.

## Stall counts explain the quantile trade-off

These are pooled descriptive background intervals, not independent statistical samples.

| Policy | Observed ITLs | >100ms | >200ms | >250ms |
| --- | ---: | ---: | ---: | ---: |
| Atomic C1024 | 8890 | 253 (2.846%) | 196 (2.205%) | 131 (1.474%) |
| Grouped C1024 | 8866 | 480 (5.414%) | 23 (0.259%) | 0 |
| Mixed C512 | 8827 | 760 (8.610%) | 0 | 0 |
| Staged G4 C1024 | 8883 | 366 (4.120%) | 280 (3.152%) | 68 (0.766%) |

Atomic's >100 ms mass stays below 5%.
G2's mass exceeds 5% but concentrates below 200 ms.
That pattern explains a worse P95 alongside a better P99.
Mixed C512 retains one outlier cohort with a 175.06 ms p99.
The other nine cohort-p99 values range from 96.81 ms to 113.83 ms in this lane.
Thus its pooled p99 is not the 110.277 ms mean shown above.

## Open-loop outputs

`serving_runs.csv` retains one row per measured policy/run.
`slo_attainment_by_rate.csv` retains all frozen request-SLO grid points and confidence intervals.
`serving_paired_differences.csv` gives paired attainment differences in percentage points.
`slo_capacity_brackets.csv` separates exploratory rate screens from qualified capacity.
`request_work_accounting.csv` checks actual full-R4 submitted work.

Configured Poisson rate and realized offered count/window are separate fields.
Natural output lengths can differ between policies.
All capped, failed, and timed-out requests remain in the offered denominator.
No qualified production-capacity or full benchmark-accuracy claim follows from a short run.

## Primary natural-serving comparison

The following Poisson table uses TTFT ≤5 s and every TBT ≤271.893227 ms.
Joint success also requires a complete natural response.
All offered instances count. Natural output lengths can differ.

| Configured req/s | Offered per policy | Atomic C1024 joint pass | G2 C1024 joint pass | Mixed C512 joint pass | G2 minus Atomic, pp [95% paired CI] |
| --- | --- | --- | --- | --- | --- |
| 0.125 | 52 | 42/52 (80.77%) | 42/52 (80.77%) | 46/52 (88.46%) | 0.000 [−14.286, 14.815] |
| 0.25 | 94 | 65/94 (69.15%) | 74/94 (78.72%) | 85/94 (90.43%) | 9.574 [1.481, 21.876] |
| 0.5 | 191 | 91/191 (47.64%) | 145/191 (75.92%) | 159/191 (83.25%) | 28.272 [11.236, 50.289] |
| 1.0 | 379 | 51/379 (13.46%) | 110/379 (29.02%) | 101/379 (26.65%) | 15.567 [4.891, 26.722] |

All 1.0 req/s Poisson blocks show queued-backlog growth.
The lower-rate sample counts and stability screens do not qualify sustainable capacity either.
Mixed's 90.43% point at 0.25 req/s is not a qualified capacity claim.
The full primary tables include TTFT-only, TBT-only, completion, cap, timeout, and latency statistics in `serving_tables.md`.

Across both arrival patterns, there are 6,880 offered instances and 1,195,403 generated output tokens across eight arms.
There are 6,171 complete natural responses, 707 capped responses, and two timeouts.
All-offered SLO estimates retain those failures and caps.
These measured instances share a frozen pool of 109 held-out source families.
They are not independent evidence for 6,880 different prompts.

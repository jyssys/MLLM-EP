# Final benchmark summary

Verdict: **NO-GO for demonstrated Loop-PP superiority over optimized native TP**.
All eight primary arms completed the same 400 questions.
Each accuracy cell below uses GSM8K N=300 and MATH N=100.
The full 1,819-question suites are USER_REDUCED_NOT_RUN.

## Eight-arm quality and actual task-serving table

| Arm | GPUs | GSM8K correct/N | MATH correct/N | Task cohort sum s | Speedup vs mode Single | Mean E2E speedup vs mode Single | Numerical scope |
|---|---|---|---|---|---|---|---|
| S4 | 1 | 232/300 | 61/100 | 6074.192 | 1.000x | 1.000x | Custom reference |
| T4 | 4 | 233/300 | 61/100 | 6219.233 | 0.977x | 0.984x | Exploratory TP |
| L4 | 4 | 232/300 | 61/100 | 3707.747 | 1.638x | 2.298x | Same-backend PP parity |
| O4 | 4 | 232/300 | 61/100 | 3311.532 | 1.834x | 2.534x | Same-backend PP parity |
| SD | 1 | 232/300 | 59/100 | 4373.494 | 1.000x | 1.000x | CDB-policy custom reference with native drift |
| TD | 4 | 231/300 | 61/100 | 4267.465 | 1.025x | 0.999x | Exploratory TP |
| LD | 4 | 232/300 | 59/100 | 3326.628 | 1.315x | 1.763x | Same-backend Dynamic parity |
| OD | 4 | 232/300 | 59/100 | 3777.943 | 1.158x | 1.366x | Same-backend Dynamic parity |

These are actual natural generations, not frozen answers.
The task clocks include finite-output checks, detokenization, and controller work.
They exclude model loading and request-bound graph preparation.
Static PP FIFO and Single/TP wave scheduling differ.
Output lengths can differ across engines.
These ratios are descriptive task-service measurements, not clean architectural speedups.

S4 is the Static denominator. SD is the Dynamic denominator.
The overall task-cohort ratios against S4 are SD=1.389x, TD=1.423x, LD=1.826x, and OD=1.608x.
[reviewed_eight_arm_results.csv](reviewed_eight_arm_results.csv) gives unrounded ratios.

Request-cohort paired 95% intervals use 5000 draws, stratified by dataset.
O4 versus L4 is 1.120x [1.114, 1.125].
OD versus LD is 0.881x [0.853, 0.904].
These intervals describe cohort composition in one observed task pass.
They do not measure repeat-to-repeat timing stability.
See [task_cohort_paired_ratios.csv](task_cohort_paired_ratios.csv).

[quality_results.md](quality_results.md) supplies accuracy percentages, both accuracy deltas, paired intervals, parse failures, and completed-correct scores.
[serving_summary.md](serving_summary.md) supplies TTFT, TPOT, P95 E2E, generated tokens, and per-GPU throughput.

## Identical-work clean replay

This example uses C4, 32 frozen output tokens per request, and the same frozen prompt/output IDs across all eight arms.
Dynamic also uses the same observed SD exit schedule.
One warm-up precedes five measured repetitions.

| Arm | Median cohort s | Mean TTFT ms | Mean TPOT ms | Speedup vs mode Single | Overall vs S4 |
|---|---|---|---|---|---|
| S4 | 2.602 | 305.413 | 74.054 | 1.000x | 1.000x |
| T4 | 2.794 | 238.723 | 81.479 | 0.931x | 0.931x |
| L4 | 0.921 | 184.967 | 23.681 | 2.824x | 2.824x |
| O4 | 1.159 | 175.177 | 31.508 | 2.245x | 2.245x |
| SD | 2.171 | 279.213 | 61.007 | 1.000x | 1.198x |
| TD | 2.306 | 241.166 | 65.777 | 0.941x | 1.128x |
| LD | 0.944 | 169.586 | 23.901 | 2.298x | 2.755x |
| OD | 1.465 | 145.704 | 42.197 | 1.482x | 1.776x |

The exact replay CSV is authoritative for unrounded metrics.
TP timing remains numerically exploratory.
Replay is not free-running accuracy or an online future-aware method.
The short-cap replay has a different KV allocation from O4096 task generation.
A separate all-eight serving-layout trace preserves O4096 allocation and verifies bounded prefix parity.

All 72 selected replay comparisons completed five measured repetitions.
The complete shape matrix, ranges, request-paired intervals, and preparation-inclusive controls are in [replay_summary.csv](replay_summary.csv).
Five repeat samples and four requests give limited uncertainty coverage.
C1 has no request-level interval.

## Fully instrumented preparation control

The additional 286 questions are the same preselected extension in every arm.
These 72 cohorts have preparation-inclusive clocks.
Both columns exclude initial model loading.

| Arm | Service sum s | Resident-model preparation-inclusive sum s |
|---|---|---|
| S4 | 3477.654 | 3674.989 |
| T4 | 3882.051 | 4205.054 |
| L4 | 2153.960 | 2331.706 |
| O4 | 1922.691 | 2006.134 |
| SD | 2504.401 | 2701.458 |
| TD | 2511.680 | 2832.424 |
| LD | 1807.227 | 2037.107 |
| OD | 2095.709 | 2182.319 |

The full 400-question Static preparation coverage is partial.
Missing earlier timestamps are not reconstructed.
The prototype recaptures request-bound graphs and does not demonstrate optimized graph pooling.

## Production-oriented native Static controls

| Engine | GPUs | GSM8K | MATH | Task cohort sum s | Within-native speedup | Output tok/s |
|---|---|---|---|---|---|---|
| Native TP1 | 1 | 232/300 | 61/100 | 948.674 | 1.000x | 171.812 |
| Native TP4 | 4 | 232/300 | 60/100 | 780.024 | 1.216x | 229.060 |

Native TP4 remains numerically exploratory.
The native engine uses FA3, packed batching, paged KV, fused projections, and a different graph lifecycle.
Custom PP has no optimized-native-attention counterpart in this study.
Different output totals prevent a pure equal-work interpretation of this table.
Nevertheless, the tested custom Loop PP does not show a serving advantage over this native control.

## Evidence status

- Actual quality: 400 questions per primary arm, with normal greedy feedback and EOS.
- Clean hardware: 72 comparisons, five measured repetitions each.
- Physical traces: 18 completed captures and full SQLite exports.
- Observer semantics: all 18 bounded comparisons preserve output IDs.
- Heavy timing: 24.9% to 405.0% overhead. UNVALIDATED for headlines.
- Dynamic PP: all 400 output and exit traces match the custom Single reference.
- Native CDB equivalence: bounded match only. Longer natural traces drift.
- Historical TP strict numerical gate: FAIL, unchanged.
- Optional DP4 capacity, optimized PP, C32/C64 expansion, and full public suites: NOT_RUN.

See [analysis.md](analysis.md), [correctness.md](correctness.md), and [reproduce.md](reproduce.md).

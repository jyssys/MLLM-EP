# Quiet serving summary

The quiet confirmation completed all 12 registered starts.
It contains 168 finite natural cohorts and 1,008 request executions across seven arms.
Each shape/budget/arm has three independent starts.
The per-arm offered counts are 24 at P2033/N4 and 48 at P8177/N8.
The natural generation cap is 512. EOS remains enabled.
This is not a Poisson request-rate capacity result.
Policy `G1` is the adaptive G-only control, not a fixed G=1 execution.

| Shape | B, ms | J1/G1 output throughput | J1/M1 output throughput | G1/J1 actual plan agreement |
| --- | ---: | ---: | ---: | ---: |
| P2033/N4 | 40 | 0.997× | 1.046× | 24/24 |
| P2033/N4 | 120 | 1.003× | 1.024× | 24/24 |
| P8177/N8 | 40 | 0.996× | 1.019× | 48/48 |
| P8177/N8 | 120 | 1.001× | 1.034× | 48/48 |

The G1/J1 output tokens also matched exactly in all 144 request pairs.
Both arms generated 16,623 tokens in total.
Joint therefore has no new measured execution choice beyond the tuned G-only control in this confirmation.
The G1/J1 rate differences remain within about 0.4%.

At P8177/N8/B40, J1 and G1 rates were 81.54 and 81.85 tokens/s.
Their paired rate difference was -0.31 tokens/s, with a block 95% interval of [-0.77, -0.02].
At P8177/N8/B120, the rates were 82.25 and 82.17 tokens/s.
Their difference was +0.08 tokens/s, with interval [-0.04, +0.20].
Only three process blocks support each interval.
These small-block intervals are exploratory, not broad population guarantees.

Native Mixed offers a different Pareto trade-off.
At P8177/N8/B120, M1 pooled ITL P99 was 59.32 ms, versus J1's 172.19 ms.
M1 TTFT P95 was 1,372.74 ms, versus J1's 1,062.16 ms.
M1 is not constrained by the experimental B controller.
Its B label is a comparison stratum, not a budget-compliance claim.

The main table contains pooled request TTFT/E2E quantiles and pooled token ITL quantiles.
It also records actual tokens, completed responses, caps, HBM, and drained output rates.
The paired interval table resamples whole process blocks with their two source cohorts.
It does not resample individual tokens as independent trials.

Stock/fair-Mixed and C-only comparisons have natural-output differences.
For J1 versus M1, 86/144 sequences and 93/144 output lengths matched.
Those contrasts therefore do not have frozen identical output work.
The identical-work G1/J1 contrast supplies the clearest negative incremental evidence.

The quality pilot completed 798 natural generations.
J1 scored 47/64 GSM8K and 32/50 extracted MATH answers.
Its completed-correct MATH result was 30/50, with six capped answers.
J1/G1 correctness labels matched on all 114 questions, with 113/114 token sequences identical.
C1A/C1S had GSM8K regressions of 3.125/6.25 percentage points versus stock Atomic.
Do not interpret serving-token agreement or this small quality pilot as population-level equivalence.
Final incremental verdict: NO_GO_INCREMENTAL.

## Separate full-answer quality pilot

These scores use B40 and width four on the preserved benchmark prompts.
They are not per-serving-cell accuracy estimates for B120 or width eight.
F0A is the stock Atomic C1024 quality reference, not the tuned fair F1A serving arm.

| Arm | GSM8K correct/N | MATH extracted correct/N | MATH completed-correct/N | MATH caps |
| --- | ---: | ---: | ---: | ---: |
| F0A | 47/64 | 28/50 | 26/50 | 8 |
| M0 | 47/64 | 28/50 | 26/50 | 8 |
| M1 | 47/64 | 28/50 | 26/50 | 8 |
| C1A | 45/64 | 32/50 | 29/50 | 7 |
| C1S | 43/64 | 33/50 | 30/50 | 7 |
| G1 | 47/64 | 32/50 | 30/50 | 6 |
| J1 | 47/64 | 32/50 | 30/50 | 6 |

The paired question intervals and parsing failures remain in the quality tables.
No full-benchmark equivalence claim follows from these pilot scores.

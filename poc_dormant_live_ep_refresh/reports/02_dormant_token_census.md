# Dormant-Live token census

`g(i)` is taken from the unmodified baseline trajectory.  At refinement step
`t`, a still-masked token is ACTIVE-H when `g(i)-t <= H` and DORMANT-H
otherwise.  This is a future-aware diagnostic, not a deployable label.

## All-phase weighted census

| task | H | ACTIVE % | DORMANT % | DEAD % |
|---|---:|---:|---:|---:|
| GSM8K | 0 | 7.40 | 35.84 | 56.76 |
| GSM8K | 1 | 13.59 | 29.65 | 56.76 |
| GSM8K | 2 | 18.89 | 24.35 | 56.76 |
| GSM8K | 4 | 27.52 | 15.72 | 56.76 |
| GSM8K | 8 | 37.81 | 5.43 | 56.76 |
| HumanEval | 0 | 10.33 | 34.75 | 54.92 |
| HumanEval | 1 | 16.63 | 28.46 | 54.92 |
| HumanEval | 2 | 21.51 | 23.58 | 54.92 |
| HumanEval | 4 | 29.11 | 15.97 | 54.92 |
| HumanEval | 8 | 38.42 | 6.66 | 54.92 |

The phenomenon exists: at H=1, about 29% of logical token-row opportunities
remain MASK yet are at least two refinement steps from baseline acceptance.
It is not confined to one task.

## H=1 phase structure

| task | phase | ACTIVE % | DORMANT % | DEAD % |
|---|---|---:|---:|---:|
| GSM8K | early | 27.15 | 63.23 | 9.62 |
| GSM8K | middle | 13.88 | 32.81 | 53.31 |
| GSM8K | late | 9.95 | 14.38 | 75.67 |
| HumanEval | early | 41.70 | 36.62 | 21.68 |
| HumanEval | middle | 16.88 | 30.43 | 52.69 |
| HumanEval | late | 8.64 | 13.31 | 78.05 |

Logical dormancy is largest early for GSM8K, but the cost-weighted opportunity
is largest in the middle phase on both tasks.  A prefix-distance/window rule is
therefore not an economic model by itself.

`DORMANT_TOKEN_CENSUS.csv` weights logical refinement records.  The EP-cost
tables weight timing-aligned layer-wave rows and therefore have a different
denominator; the two percentages must not be substituted for one another.

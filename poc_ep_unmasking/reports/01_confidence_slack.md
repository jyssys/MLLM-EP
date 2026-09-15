# 01 — Confidence slack

The atlas includes only measured steps with at least one MASK and at least
one finalized position. GSM8K has 518 complete decision rows. HumanEval has
247 decision rows before the observer-heavy run failed; this is a partial
trajectory. An alternative is an *unselected* eligible position whose
confidence is within `delta` below the least-confident selected position.

| Delta | GSM8K steps with ≥1 alternative | HumanEval partial | Both tasks |
| ---: | ---: | ---: | ---: |
| 0.001 | 0.77% | 0.40% | 0.65% |
| 0.002 | 1.16% | 0.81% | 1.05% |
| 0.005 | 1.93% | 2.43% | 2.09% |
| 0.010 | 4.25% | 4.86% | 4.44% |
| 0.020 | 10.23% | 11.74% | 10.72% |
| 0.050 diagnostic | 25.68% | 26.32% | 25.88% |
| 0.100 diagnostic | 46.14% | 42.11% | 44.84% |

At delta 0.02, two or more alternatives occur in only 2.32% of GSM8K and
2.43% of partial HumanEval decisions; four or more never occur. The median
alternative count is zero for every tested delta, and the median stock
transfer count is one in early, middle, and late refinement.

Slack is not strongest in late refinement. At delta 0.02 the fractions with
an alternative are GSM8K early/middle/late `11.1/12.5/7.2%` and partial
HumanEval `9.7/19.1/4.6%`. Delta 0.05/0.1 are sensitivity controls, not
established semantic-safe regions; no alternative-policy quality was measured.

The strict semantic-slack gate is weak: the useful near-tie set is sparse,
particularly for the 0.001–0.01 confidence differences in the motivating
example. See `CONFIDENCE_SLACK.csv` for all decision-level counts.

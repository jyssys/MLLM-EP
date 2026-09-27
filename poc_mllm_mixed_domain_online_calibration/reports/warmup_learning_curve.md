# Domain-balanced warm-up learning curve

Primary metric below is held-out aggregate 1–128 decode expert-histogram cosine under frozen calibration. Ratio selection used discovery only. Zero warm-up is an explicitly uncalibrated uniform prior, not future truth.

| nominal ratio | P_text frozen | P_full frozen | P_separated frozen | P_vision frozen | domain label frozen | P_separated continual |
|---:|---:|---:|---:|---:|---:|---:|
| 0% | 0.4306 | 0.4306 | 0.4306 | 0.4306 | 0.4306 | 0.9510 |
| 1% | 0.9484 | 0.9486 | 0.9487 | 0.9489 | 0.9305 | 0.9589 |
| 5% | 0.9552 | 0.9558 | 0.9555 | 0.9545 | 0.9481 | 0.9600 |
| 10% | 0.9574 | 0.9563 | 0.9560 | 0.9546 | 0.9526 | 0.9598 |
| 20% | 0.9601 | 0.9605 | 0.9599 | 0.9589 | 0.9578 | 0.9610 |

Frozen practical knee: **10%**. Rule: smallest 5/10/20% ratio within 0.005 histogram cosine of discovery best. Domain-balanced realized counts are reported in `domain_cohort.md`; the 1% nominal point realizes 2.5% per domain. Warm-up requests are ordinary baseline service requests (no duplicate inference); their opportunity cost is the first K requests/domain without a calibrated predictor, plus the separately measured CPU update/lookup cost.

# 05 — Semantic risk calibration boundary

No deployable `P(aggressive action changes final correctness)` calibration
model was trained. The future-aware action oracle and EP-specific controller
did not pass their gates, so assigning per-step risk from a few final labels
would be unjustified. [RISK_CALIBRATION.csv](../RISK_CALIBRATION.csv) has a
header and no fabricated rows.

What *is* measured is paired full-trajectory risk on promoted cohorts:

| Cohort | Phase vs baseline paired wins/losses | Score delta | Bootstrap 95% delta (pp) | Exact final-answer strings |
|---|---:|---:|---:|---:|
| GSM8K n=512 | 4/5 | -0.195 pp | [-1.367,+0.977] | not a quality certificate |
| GSM8K full n=1,319 | 9/7 | +0.152 pp | [-0.455,+0.758] | 643/1,319 = 48.7% |
| HumanEval full n=164 | 1/0 | +0.610 pp | [0,+1.829] | 88/164 = 53.7% |

The full GSM8K McNemar paired-difference p-value is 0.804; it does not
establish quality *improvement*. A conservative bootstrap lower bound of
-0.455 pp supports the 0.5-pp non-inferiority budget on this bounded full
test, not the 0.3-pp budget. HumanEval is coarse (one item=0.610 pp) and has
very low baseline pass rate. About half the final strings changed even where
benchmark score barely moved. String difference is not automatically a task
failure, but it is a future-trajectory robustness warning.

The structural [decision trace](../results/observer_base_gsm32/DECISION_TRACE.csv)
records confidence, masked positions and accepted count per ready wave.
Trace-and-clean final outputs matched 32/32 for both baseline and phase,
but tracing raised BCT substantially. Without action-level future labels and
larger quality-valid generalization cohorts, early/middle/late risk
calibration, epsilon scheduling, and statistical semantic safety cannot be
claimed.

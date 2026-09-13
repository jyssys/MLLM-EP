# Track B3 — AsyncDiff-like policy sweep

Eighteen policies were evaluated on each bounded task after a single model
load.  Each policy sequentially replaces selected pipeline-boundary rows with
their lag-1 `(hidden,residual)` state.  Thus the complete dLLM trajectory and
benchmark output are measured, but speed is not: a real PP schedule is not
running.

Axes covered:

- PP2 boundary 16 and PP4 boundaries 8/16/24;
- warm-up W=0/4/8;
- exact refresh K=2/4/8 or no refresh;
- early/middle/late-only operation;
- individual boundary selection.

Important results:

| Policy | GSM score / NFE delta / seq exact | HumanEval score / NFE delta / seq exact | Predicted speed upper, GSM / Human |
|---|---|---|---:|
| PP4 W0, no refresh | 3/5, +52, 0% | 1/6, +39, 12.5% | 1.411x / 1.613x |
| PP4 W4, K2 | 5/5, +0, 12.5% | 6/6, +14, 37.5% | 1.212x / 1.060x |
| PP4 W4, K8 | 6/5, +17, 6.25% | 6/6, +30, 31.25% | **1.291x / 1.184x** |
| PP4 W4, K4, late | 7/5, +5, 34.38% | 6/6, -2, 53.13% | 1.051x / 1.063x |
| PP2 W4, no refresh | 5/5, +14, 18.75% | 7/6, +51, 34.38% | 1.154x / 0.923x |

The apparent score improvements are not evidence of higher quality: bounded-32
scores are coarse (baseline 5/32 and 6/32).  Sequence exactness and NFE show
large trajectory changes.  The best policy retaining at least baseline score
on both tasks is PP4 W4/K8, but its worst-task analytical speed upper is only
1.184x, below the 1.2x characterization threshold and far below the 1.3x HOLD
gate.

Raw quality is in `analysis/trackB_{task}_policy_quality.csv`; the conservative
NFE-adjusted analytical joins are in
`analysis/trackB_speed_quality_pareto.csv`.

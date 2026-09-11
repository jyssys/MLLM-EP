# Semantic compression oracle

## Scope and evidence

Stage 1 keeps every text token at K=8 and permits every integer visual K in
`{1,...,8}`.  The semantic-perfect oracle perturbs one 4x4 visual group at a
time across all 48 language MoE layers and scores the answer-token KL/NLL.
This oracle uses the ground-truth answer and therefore bounds quality rather
than providing a deployable selector.  Full-model masking runs still execute
all eight experts and are quality evidence only.

Calibration uses eight GQA and eight ChartQA requests; the held-out screen is
16 GQA plus 16 ChartQA requests with 16-token greedy generation.  An exact
multiple-choice dynamic program is used because isolated error is not
monotone in K (`K=1 <= K=7` in only 41.4% of GQA group cases).

## Main results

| Held-out policy | Realized vision assignment reduction | Short greedy exact | Benchmark score | Delta |
|---|---:|---:|---:|---:|
| Stock | 0.00% | 32/32 | 24/32 | — |
| Semantic full K, target 20% | 18.61% | 32/32 | 24/32 | 0 pp |
| Semantic full K, target 30% | **27.15%** | **32/32** | **24/32** | **0 pp** |
| Semantic coarse K, target 20% | 18.13% | 31/32 | 23/32 | -3.125 pp |
| Semantic coarse K, target 30% | 26.19% | 31/32 | 23/32 | -3.125 pp |
| Fixed visual K=6 | 22.52% | **32/32** | **24/32** | **0 pp** |
| Router mass 0.8 | 17.86% | 31/32 | 25/32 | +3.125 pp, not evidence of improvement |
| Contribution oracle, target 30% | 27.02% | 31/32 | 25/32 | +3.125 pp, not evidence of improvement |
| Semantic full K, target 40% | 35.81% | 31/32 | 24/32 | 0 pp |
| Semantic full K, target 50% | 45.04% | 30/32 | 24/32 | 0 pp |
| Fixed visual K=4 | 45.03% | 30/32 | 25/32 | +3.125 pp, not evidence of improvement |

The full integer grid is useful: at matched isolated assignment budgets it
has lower ChartQA additive-KL than the coarse grid at every tested target, and
on held-out full-model generation the full policies retain all outputs while
both coarse policies change one.  This is a real quality frontier effect, not
yet a new EP systems result.

The simple-fix attack is decisive.  Fixed K=6 recovers
`22.52 / 27.15 = 82.9%` of the safe 30%-target assignment reduction with the
same held-out benchmark score and exact short outputs.  Thus a complicated
semantic scheduler does not pass the specification's 80% non-triviality gate
on this cohort.

The 40/50% stress test did not lower the aggregate 24/32 score, but changed
one/two generated outputs respectively.  Fixed K=4 reaches the same 45%
assignment reduction and changes the same number of outputs.  With only 32
requests, unchanged aggregate accuracy cannot establish that compensating
correct/incorrect flips are quality-safe.  We therefore report 27.15% as the
strict output-preserving ceiling and about 36% as the observed benchmark-score
ceiling, rather than promoting the 45% maximum.

## Selector versus fundamental limit

The expensive actual-contribution oracle, which explicitly evaluates all
expert branches before choosing the retained prefix, is not materially better
than router-only selection.  At the 30% target it changes one of 32 short
outputs and has no credible benchmark advantage over stock.  Captured-layer
local error agrees: at 10% branch removal contribution selection has median
combined relative L2 17.97% versus a similar router frontier, despite much
higher oracle cost.  The data do not expose a missing high-value contribution
selector.

Keeping the original retained router weights is consistently safer than
renormalizing them.  On 54 captured sample/layer cases at 10% router-risk
removal, the combined relative-L2 median is 16.54% without renormalization and
29.03% with it.  The runtime prototype therefore must not assume normalized
top-K weights after omission.

## Interpretation

G0/G1 pass at the tested 27% held-out ceiling: meaningful visual work can be
removed without an aggregate quality loss on this small benchmark cohort.
However, Stage 1 by itself is not novel and fails the trivial-fix attack.  Its
main value in this PoC is to establish a quality-safe compute budget against
which the EP-specific rearrangement can be tested.

The 32-request cohort is a screening set, not a benchmark-quality confidence
interval.  No production-quality claim is made, especially for OCR/math
tails, without a much larger task evaluation.

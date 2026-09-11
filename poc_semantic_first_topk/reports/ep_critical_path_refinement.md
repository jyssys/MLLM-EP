# EP critical-path refinement

## Counterfactual

Stage 2 preserves the exact number of retained expert assignments.  It swaps
an omitted prefix-suffix branch back onto a non-critical rank while omitting a
slightly higher-risk suffix branch on the current critical rank.  The router
risk increase is bounded to 1.05x.  Expert placement, weights, token count,
and text K=8 remain unchanged.

This is deliberately stronger than the predecessor's tail-first comparison:
the starting point is a semantic schedule already selected for task quality.

## Captured-route logical oracle

Across nine real images, six layers (54 cases), full integer-K router risk
produces the following medians.

| Vision branch budget | Stage-1 max-rank reduction | Stage-2 max-rank reduction | Ratio |
|---:|---:|---:|---:|
| 5% | 4.11% | 7.86% | 1.91x |
| 10% | 8.12% | 14.71% | 1.81x |
| 20% | 16.19% | 25.25% | 1.56x |
| 30% | 25.10% | 33.28% | 1.33x |

The count-level G2 signal is reproducible and strongest at small budgets.  It
weakens as the semantic schedule itself removes enough rows to flatten rank
loads.

## Full-model semantic controls

On the 8-request-per-task calibration cohorts, refinement preserves the same
assignment count.  At 10%, its max-rank reduction is 1.65x the semantic-only
schedule on GQA and 1.85x on ChartQA.  At 20/30% the ratios fall to roughly
1.4x/1.27x.  These masked runs are not performance measurements and the
refinement's CPU oracle cost is excluded; they only test the quality/load
counterfactual.

The decisive held-out 32-request control separates the two useful budgets.
At 20%, the refined schedule preserves 32/32 short outputs and the 24/32
baseline score, with 25.05% median max-rank reduction.  At 30%, it reaches
33.43% max-rank reduction but preserves only 31/32 outputs (score 25/32; the
single positive flip is not treated as improvement).  Semantic-only 30%
preserves 32/32.  Thus the refinement's 13% latency point is not on the same
strict quality frontier; its strictly output-preserving point projects only
9.92% total TTFT reduction.

## What the count oracle does not prove

Max-rank assignment count is not proportional to DeepEP latency.  Real replay
shows only 1.25–1.34x amplification of MoE latency benefit at 30% and
approximately 1.30x at 20%, even when count reduction is around 1.5–1.6x.
Dispatch/combine fixed costs, expert shape, and non-critical ranks dilute the
count win.

The relevant paper gate is not whether Stage 2 can rebalance rows.  It is
whether the same-quality, same-compute rearrangement adds material request
latency beyond semantic adaptive K.  The measured incremental projected TTFT
is only about 1.5–3.7 percentage points depending on the selector; the
deployable calibration-aggregate exact-budget schedule adds 2.59 points at
the 30% target.
That is an N2 outcome: a real physical-load effect but weak distinct systems
headroom.

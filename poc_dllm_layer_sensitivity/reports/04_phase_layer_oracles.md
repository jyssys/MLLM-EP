# Phase × layer necessity oracles

## Evidence definitions

- **Current-step exact:** first affected wave has zero live-token top-1 flip and
  zero accepted-set change.
- **Benchmark-vector safe:** every one of the 32 samples keeps its baseline
  pass/fail value. This is deliberately weaker than trajectory safety because
  the bounded baseline has only 5/32 GSM8K and 6/32 HumanEval correct.
- **Final-trajectory exact:** all 32 token sequences, NFE, and per-sample task
  outcomes equal baseline.
- **Post-Epoch/live-weighted:** analytical sensitivity that multiplies each
  measured phase/layer cost by its decision-live ratio. It is not measured
  Epoch performance, but prevents claiming already-dead physical rows as an
  independent successor opportunity.

## O1: independent single-layer upper bound

Summing every independently benchmark-vector-safe cell gives misleadingly
large values (for example 38.22% routed-MoE on GSM8K). It is interaction-unsafe:
the same cells fail once combined. Under the final-trajectory criterion, no
cross-task full-layer or routed-bypass cell with nonzero cost survives. The
only GSM8K stale single-layer total is 0.68% E2E.

This is why the report does not headline O1.

## O2/O3: actual multi-layer interventions

The best cross-task policy that keeps the small benchmark pass/fail vector is:

`stale routed MoE, middle phase, greedy layers {2,4,6,7,15,20,27,28}`

| metric | GSM8K | HumanEval | two-task mean |
|---|---:|---:|---:|
| optimistic current-runtime E2E ceiling | 6.89% | 10.65% | **8.77%** |
| live-weighted residual ceiling | 3.06% | 4.89% | **3.98%** |
| final exact requests | 13/32 | 9/32 | — |

The strongest contiguous quarter has a similar result: 6.38/10.23% current
ceiling and 2.84/4.70% live-weighted ceiling, but only 4/32 and 7/32 final
trajectories exact.

For fresh routed bypass rather than staleness, the best cross-task tested
multi-layer policy is only 2.67% mean optimistic E2E (four late layers), and it
leaves 16/32 GSM8K and 19/32 HumanEval trajectories exact. No strict cross-task
multi-layer policy survives at any nonzero cost.

## O4: final-trajectory oracle

The strongest final-trajectory-exact cross-task oracle is **0.00%** among all
tested nonzero-cost group and greedy policies. A HumanEval-only early stale
policy reaches 0.42%, and GSM8K single stale cells sum to only 0.68%; neither is
material or general.

The 8.77% bounded-score result is therefore not a quality-safe oracle. Even if
its benchmark preservation were confirmed on a much larger sample, its
post-liveness residual is below 5%, and cache lookup, refresh, verification,
layout, and fallback costs can only reduce it.

## Gate

- Full-layer refresh: killed by causal sensitivity and interaction.
- Fresh routed-MoE selective refresh: below 5% credible cross-task headroom.
- Shared-only reduction: below 3.5% whole-request component ceiling.
- Stale routed MoE: raw weak/HOLD-sized ceiling, but below 5% independent
  post-Epoch residual, trajectory-unsafe on this sample, and directly adjacent
  to DICE.

No candidate reaches the 8% method-prototype gate with credible quality and
independent headroom. No live optimization prototype was implemented.

Detailed results: `SINGLE_LAYER_ORACLE.csv`, `GROUP_ORACLE.csv`,
`MULTI_LAYER_GREEDY_ORACLE.csv`, `CROSS_TASK_ORACLE.csv`, and
`FINAL_TRAJECTORY_ORACLE.csv`.

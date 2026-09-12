# Causal decision-sensitivity results

All interventions compute the exact baseline unit first and then replace its
contribution, so these runs test semantics rather than claim a speedup. Each
single-phase policy reaches an identical pre-state before the first affected
wave. Output correctness is compared with the unmodified baseline policy from
the same loaded model.

## Routed-MoE-only bypass

Median across all 32 layers (layer 0 is dense and serves as a zero-effect
control):

| task / phase | live top-1 flip | accepted-set change | final exact requests | median NFE |
|---|---:|---:|---:|---:|
| GSM8K early | 13.87% | 2.05% | 25.0/32 | 64.5 |
| GSM8K middle | 19.37% | 2.84% | 10.0/32 | 62.0 |
| GSM8K late | 14.60% | 2.38% | 18.5/32 | 64.0 |
| HumanEval early | 13.78% | 2.74% | 24.0/32 | 90.0 |
| HumanEval middle | 23.40% | 5.85% | 11.5/32 | 86.0 |
| HumanEval late | 14.29% | 3.90% | 19.0/32 | 86.0 |

Representative sampled-logit KL is largest in the middle phase: 0.0567 GSM8K
and 0.0929 HumanEval, versus 0.0093/0.0338 early and 0.0229/0.0551 late. There
is no increasing population of decision-insensitive routed layers in late
refinement.

## Full-layer bypass

Full-layer identity is more destructive, as expected. GSM8K median live top-1
flip is 19.3/26.3/20.3% in early/middle/late, while median final exact requests
fall to 25/7/14. On ten representative HumanEval layers the corresponding
final exact medians are 23/9/16. No individual full-layer policy preserves all
32 final trajectories.

This is an intentionally strong diagnostic. A real skipped layer would still
need at least the state required for future attention/KV use, so the normalized
full-layer cost is an optimistic ceiling, not an implementable saving.

## Shared versus routed contribution

`routed_only` retains routed experts and removes the shared-expert contribution.
On representative GSM8K layers, its causal sensitivity score is only modestly
below routed removal:

| phase | routed removed | shared removed |
|---|---:|---:|
| early | 0.0876 | 0.0707 |
| middle | 0.1968 | 0.1771 |
| late | 0.1121 | 0.1046 |

Routed removal is more sensitive in 16/27 matched cells, but both paths are
decision-relevant. Moreover, the entire shared-expert component is only
2.98--3.50% of clean E2E, below the method gate even under perfect removal.

## Multi-layer compensation attack

Contiguous eight-layer quarter bypasses reject the hypothesis that individually
sensitive layers become jointly redundant through compensation:

- GSM8K full-layer quarter policies have 0--3/32 final exact in early,
  0/32 in middle, and 0--2/32 in late.
- HumanEval full-layer quarters have median 1/32 early, 0/32 middle, and
  7.5/32 late.
- Routed-only quarter bypasses are less destructive but still fall to 0--10/32
  in GSM8K middle/late and 0--15/32 in HumanEval.

Noncontiguous greedy sets produce the same conclusion. On GSM8K, the two
least-sensitive routed layers leave only 10/32 final exact in middle and 20/32
in late; four layers leave 8/32 and 16/32. HumanEval produces 10/32 and 20/32
already at two layers.

## The stale-output trap

Previous-iteration routed output is a sensitivity probe, not a novelty claim.
In middle phase, its first affected wave often has zero top-1 or accepted-set
change, yet the median final exact count is 14/32 for GSM8K single layers and
9--13.5/32 for greedy multi-layer sets. Therefore:

> current-step decision unchanged does not imply final diffusion trajectory
> unchanged.

This is the most important validation warning from the sprint. A controller
trained on only instantaneous acceptance agreement would certify unsafe work.

Primary tables: `DECISION_SENSITIVITY_COST.csv`, `GROUP_ORACLE.csv`,
`MULTI_LAYER_GREEDY_ORACLE.csv`. Required figures are
`figures/08_layer_phase_decision_sensitivity_map.png`,
`figures/09_decision_sensitivity_vs_latency_scatter.png`,
`figures/10_moe_only_sensitivity_map.png`, and
`figures/11_shared_vs_routed_contribution_map.png`.

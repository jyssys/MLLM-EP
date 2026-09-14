# Causal routed-MoE interventions

The diagnostic path keeps attention, dense operations, shared expert, and
router fresh.  For selected DORMANT rows it computes the original routed work
and then replaces the routed contribution with the cached prior output.  This
isolates trajectory causality but **does not save runtime**.

| task/policy | would-defer routed rows | sequence exact | NFE | acceptance-iteration exact | task score vs baseline | feasible E2E oracle |
|---|---:|---:|---:|---:|---:|---:|
| GSM H1/K2 | 10.25% | 12/32 | 61 vs 66 | 55.30% | 6 vs 5 | 3.18% |
| GSM H1/K4 | 16.28% | 10/32 | 65 vs 66 | 50.74% | 6 vs 5 | 4.42% |
| GSM H1/K8 | 18.44% | 5/32 | 61 vs 66 | 41.87% | 4 vs 5 | 4.98% |
| GSM H1/route-K8 | 2.32% | 17/32 | 62 vs 66 | 60.32% | 5 vs 5 | 0.63% |
| Human H1/K2 | 10.94% | 16/32 | 84 vs 86 | 75.08% | 6 vs 6 | 2.96% |
| Human H1/K4 | 15.31% | 12/32 | 103 vs 86 | 70.52% | 7 vs 6 | 4.37% |
| Human H1/K8 | 18.92% | 10/32 | 84 vs 86 | 68.85% | 6 vs 6 | 5.15% |
| Human H1/route-K8 | 1.84% | 20/32 | 90 vs 86 | 77.77% | 6 vs 6 | 0.54% |

Bounded-32 scores are coarse—one sample is 3.125 percentage points—and the
baseline itself scores only 5/32 and 6/32 because generation is truncated to
32 tokens.  Score equality therefore is not a quality certificate.  Every
intervention changes NFE, the acceptance schedule, and at least 12/32 final
sequences.  The unconstrained H1 stale variants preserve only 2/32 and 8/32
sequences.  Current-step plausibility does not survive the full trajectory.

Fresh-router route/destination gating is safer by sequence exactness, but it
does not make the intervention quality-equivalent and collapses the oracle to
well below 1%.  A router-score threshold could trade between these endpoints;
it cannot exceed the already sub-8% perfect future-removal ceiling.

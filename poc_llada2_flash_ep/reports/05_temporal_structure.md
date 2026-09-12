# Temporal structure

## Fresh LLaDA2.0-Flash measurement

The primary temporal trace contains 13,578 logical sparse-layer invocations,
399,776 token pairs at lag 1, and four requests with 87/106/102/151 NFE. Rank
rows are joined into one logical invocation; they are never summed as request
latency.

| Lag | ordered expert agreement | exact top-k set | top-k overlap | exact destination set | destination overlap | rank-load cosine | critical-rank persistence |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 41.04% | 29.11% | 74.29% | 67.48% | 93.18% | 0.9942 | 87.18% |
| 2 | 33.76% | 21.96% | 67.41% | 60.43% | 91.48% | 0.9908 | 83.84% |
| 4 | 25.94% | 15.11% | 58.28% | 51.88% | 89.30% | 0.9843 | 79.11% |
| 8 | 16.87% | 7.76% | 46.07% | 41.50% | 86.51% | 0.9713 | 72.21% |

Lag-1 top-k overlap rises from 65.38% early to 75.60% middle and 81.11% late,
but exact set agreement is only 15.22%/31.05%/39.81%. LLaDA2 therefore does
not reproduce the earlier SDAR figure by assumption: branch identities are
volatile even while coarse destination and load geometry is exceptionally
stable.

## Causal/economic implication

The stable coarse state is predictive, but it does not identify much
removable work:

- exact route-plan reuse applies to only 29.11% of lag-1 tokens;
- router itself is only 10.13% of the inner observed MoE span;
- even perfect router removal maps to 5.38% of block time, and exact-set-gated
  reuse maps to only 1.57%;
- 74.29% branch overlap cannot remove a DeepEP call because typical tokens
  still touch about 3.07 destination ranks.

Temporal predictability is thus a useful characterization, not a 12% method
oracle.

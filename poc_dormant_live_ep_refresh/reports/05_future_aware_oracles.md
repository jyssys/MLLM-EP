# Future-aware routed-MoE oracles

All numbers below are request-E2E projections over the best-static clean
baseline.  They exclude DEAD rows, keep attention/shared/router work, and are
therefore additional post-Epoch opportunity.  `optimistic` removes the
allocated dispatch/expert/combine duration; `feasible` retains calibrated
DeepEP fixed floors.

## Perfect future-dormant removal

| H | GSM optimistic | GSM feasible | Human optimistic | Human feasible |
|---:|---:|---:|---:|---:|
| 0 | 9.96% | 8.55% | 10.50% | 8.88% |
| 1 | 8.09% | 6.95% | 8.61% | 7.27% |
| 2 | 6.52% | 5.61% | 7.11% | 6.01% |
| 4 | 4.08% | 3.52% | 4.76% | 4.02% |
| 8 | 1.33% | 1.15% | 1.94% | 1.64% |

H=0 is not operationally useful: every still-MASK token not accepted on this
step is called dormant using future knowledge.  H=1 is the first meaningful
frontier guard, and even its impossible perfect-removal feasible mean is only
**7.11%**.

## H=1 refresh policies

| policy | GSM feasible | Human feasible | mean |
|---|---:|---:|---:|
| periodic K=2 | 3.18% | 2.96% | 3.07% |
| periodic K=4 | 4.42% | 4.37% | 4.39% |
| periodic K=8 | 4.98% | 5.15% | 5.06% |
| route/destination-trigger K=8 | 0.63% | 0.54% | 0.59% |

The best periodic oracle is only characterization-level before cache lookup,
scatter, and controller overhead.  The expert-aware exact route/destination
trigger—the mechanism most likely to preserve normal routed semantics—retains
less than 0.6% mean E2E.  No scheduler or physical skipping implementation is
justified by the gate.

# 1A — Predictive overlapped transient replication

## Decision

**NO-GO.** The predictor premise survives, but even an impossible zero-cost,
unlimited-replication oracle projects only **2.63% request E2E**. The best
costed one-replica oracle is 0.75%; the simple causal policy is 0.46%.

## Measured copy economics

The live microbenchmark copied the exact runtime tensor footprint of one
`(layer, expert)`: 12,582,912 BF16 bytes. All 12 directed GPU pairs among
physical GPUs 4–7 were tested.

| Metric across directed pairs | Result |
|---|---:|
| Copy-only p50, median | 0.05547 ms |
| Copy p90, median | 0.05763 ms |
| Effective bandwidth, mean | 227.56 GB/s |
| Concurrent useful compute, p50 | 0.148 ms |
| Visible copy cost, median | 0.04684 ms |
| Visible copy cost range | 0.03690–0.09842 ms |

The oracle charges the median measured visible cost once per replica lease.
Future tokens are split optimally between the home and replica ranks to minimize
their maximum load. Non-overlapping leases prevent double-counting future work.

## Perfect-future versus causal policy

The latency mapping below assumes, favorably, that expert CUDA time scales
perfectly with max-rank assignments. The empirical fitted mapping is zero.

| Lease H | Oracle installs | Oracle E2E | Causal E2E | Causal recovery | Wasted causal copies |
|---:|---:|---:|---:|---:|---:|
| 1 | 486 | 0.081% | -0.042% | negative | 61.30% |
| 2 | 706 | 0.326% | 0.138% | 42.18% | 36.49% |
| 4 | 638 | 0.601% | 0.368% | 61.16% | 31.10% |
| 8 | 523 | **0.746%** | **0.459%** | **61.56%** | 35.46% |

At H=8, the perfect-future oracle saves 89.22 ms of optimistic expert work and
64.72 ms net after copies over 8,672.22 ms of clean request time. The causal
policy saves 39.84 ms net. It passes the requested ≥50% predictor-recovery gate
but fails the economic gate by more than an order of magnitude.

The split/destination controls point in the same direction. At H=8, the best
50/50 split projects 0.372% E2E, versus 0.746% for per-iteration rank
equalization. The causal current-state destination is the currently
least-loaded rank in 78.2% of installed H=8 leases (84.5% at H=1); allowing a
more expensive destination search therefore does not manufacture the NO-GO.

## Absolute kill bound

The one-replica formulation could be criticized as too restrictive. An even
stronger counterfactual therefore sets copy cost to zero, permits unlimited
replicas, perfectly equalizes every invocation's four ranks, and assumes the
entire expert stage falls linearly with max-rank assignments. It removes only
227.79 ms, or **2.6266%** of clean request time. No exact implementation of this
mechanism can exceed that bound under the measured workload.

## Memory and implementation

One replica is 12 MiB. Sixteen simultaneous replicas would be 1.5625% of the
global 64-expert × 16-layer storage, but memory is not the limiting resource;
critical-path mass is. The memory curve is an interpolated optimistic oracle,
not a measured cache implementation.

No production replica manager, token redirector, or communication kernel was
implemented after the early kill gate. The result does not deny that dLLM
expert hotness is predictable. It establishes that transient exact GPU
replication is economically immaterial on this four-H100 EP4 regime.

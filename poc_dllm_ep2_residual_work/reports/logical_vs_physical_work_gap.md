# Logical useful work versus physical executed work

## Definitions

`Physical executed work` is the stock dInfer M×top-k assignment count presented
to DeepEP and TritonExperts.  `Epoch-equivalent fresh work` is the live plus
newly-decoded generation set, with a full M=5 refresh and a cold full pass.  It
is a deliberately favorable offline model of Epoch's algorithm, not an Epoch
implementation in dInfer.

`Decision-relevant` does not mean that an output happens to be numerically
similar.  A fresh live value remains decision-relevant unless a future-known
oracle proves its contribution can be reused within a stated tolerance.

## Gross logical/physical gap

Across all captures:

| Quantity | Assignments | Fraction of stock |
|---|---:|---:|
| Stock physical work | 884,736 | 100.00% |
| Epoch-equivalent fresh work (M=5) | 422,912 | 47.80% |
| Stable-decoded/removable work | 461,824 | 52.20% |

This is a large structural work gap, but it is not a new result after Epoch.

## Why 52% assignments do not imply 52% latency

The measured DeepEP HT curve on EP2 is startup dominated over M=2--128:
dispatch is about 0.286--0.290 ms, expert 0.316--0.419 ms, and combine
0.080--0.057 ms.  Shrinking M therefore leaves most per-layer fixed cost.
Mapping each iteration's fresh M through that curve gives:

| Request | Assignment reduction | MoE-stage reduction | Projected request E2E upper bound |
|---|---:|---:|---:|
| expository | 51.85% | 7.15% | 2.68% |
| math | 52.48% | 7.26% | 4.16% |
| systems | 51.95% | 6.26% | 0.74% |
| **Median** | **51.95%** | **7.15%** | **2.68%** |

The request mapping uses the clean 620.63 ms median as denominator.  It is an
offline upper bound, not a measured Epoch request speedup.  It includes no
packing/cache overhead, so real gain can only be lower under this EP2 curve.

## Residual perfect-oracle bounds after removing prior-art work

| Oracle | Work assumed free | Removable stock assignments | MoE-local gain | Request E2E upper bound |
|---|---|---:|---:|---:|
| Router elimination | every router linear in the request | 0% | n/a | **2.693%** |
| Stable route/layout delta | every stable dispatch-layout rebuild | 0% | n/a | **0.0249%** |
| Exact branch reuse | bit-identical fresh persistent branches | 0% | 0% | **0%** |
| <=1% expert-output-change branch reuse | every future-known safe fresh branch | 0.073% | 0.0096% | **0.0014%** |
| <=5% expert-output-change branch reuse | every future-known safe fresh branch | 0.998% | 0.158% | **0.0233%** |

The router row is the strongest possible bound: it removes the complete router
linear even though live hidden states and top-k values change.  A practical
predictor can recover only a subset and must pay validation/metadata overhead.

The stable-layout bound uses the profiled 3.936 us
`get_dispatch_layout` kernel and the observed route stability.  It does not
claim dispatch payload is removable; hidden payload is still fresh.

## Layer- and iteration-selective refresh

Restricting <=5%-change reuse to early layers captures almost all available
branches but still projects only 0.02316% E2E.  Middle and late bands contribute
0.000080% and 0.000035%.  A future-known selection of only transition/layer
cells whose aggregate one-step error is <=1% projects 0.01123% E2E.

Therefore the early-layer pocket is not an economic opportunity on EP2.  It is
not rescued by avoiding risky middle/late layers.

## Figures

- `analysis/plots/logical_vs_physical_ratio.png`
- `analysis/plots/candidate_removable_work.png`
- `analysis/plots/candidate_projected_e2e.png`

All candidate gains are offline oracles calibrated from real EP2 stage timing;
none is an implemented method or measured request speedup.

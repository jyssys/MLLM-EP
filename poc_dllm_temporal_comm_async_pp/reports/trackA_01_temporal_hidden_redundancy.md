# Track A1 — temporal dispatch-hidden redundancy

Actual inputs to routed DeepEP dispatch were captured for layers
1/8/16/24/31 and matched by request, output block, logical token position,
layer, timestep, and destination rank.  The tracing run is observer-heavy and
is used only for geometry/error measurements.

## Lag-1 result

| Task / phase | Same remote destination | Same exact expert | Same-rank, different expert | Hidden cosine p50 | Hidden rel-L2 p50 |
|---|---:|---:|---:|---:|---:|
| GSM8K all | 84.65% | 78.80% | 5.86% | 0.9627 | 0.2775 |
| GSM8K middle | 84.73% | 78.62% | 6.10% | 0.9413 | 0.3431 |
| GSM8K late | 93.20% | 88.15% | 5.06% | 0.9778 | 0.2135 |
| HumanEval all | 81.03% | 72.80% | 8.23% | 0.9493 | 0.3221 |
| HumanEval middle | 82.54% | 74.06% | 8.47% | 0.9317 | 0.3709 |
| HumanEval late | 91.70% | 83.01% | 8.69% | 0.9832 | 0.1841 |

The premise is real but nuanced: late states are both more cacheable and more
similar, while early states are poor.  The delta is not sparse: only about
11.6--12.4% of elements fall below 1e-2, its INT8-symbol entropy is about
7.05 bits, and the top 10% of elements carry only about 44.8% of energy.
Sparse delta and low-rank branches were therefore not promoted.

Lag-2 and lag-4 degrade materially.  For example, all-phase same-rank hits fall
from 84.65% to 74.68% to 57.51% on GSM8K and from 81.03% to 68.86% to 51.32%
on HumanEval.  The implementation-relevant opportunity is lag-1 only.

Raw/derived evidence: `trackA/temporal_trace/*` and
`analysis/trackA_temporal_summary.csv`.

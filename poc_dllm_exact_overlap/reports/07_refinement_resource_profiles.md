# Refinement resource profiles

Stage 2 was triggered because no Stage 1 candidate had >=8% credible direct
request headroom. The phase analysis reuses the exhaustive layer×wave trace from
the immediately preceding validated true-EP4 discovery run, and adds fresh
cross-state concurrency replay in this PoC.

## Critical-rank stage medians

| dataset | phase | expert | dispatch | combine | comm fraction of E+D+C | whole MoE |
|---|---|---:|---:|---:|---:|---:|
| GSM8K | early | 1.028 ms | 0.420 ms | 0.250 ms | 39.47% | 1.961 ms |
| GSM8K | middle | 1.036 ms | 0.392 ms | 0.214 ms | 36.90% | 1.904 ms |
| GSM8K | late | 0.787 ms | 0.312 ms | 0.146 ms | 36.81% | 1.545 ms |
| HumanEval | early | 1.081 ms | 0.415 ms | 0.270 ms | 38.79% | 1.998 ms |
| HumanEval | middle | 0.902 ms | 0.396 ms | 0.187 ms | 39.25% | 1.742 ms |
| HumanEval | late | 0.622 ms | 0.318 ms | 0.137 ms | 42.29% | 1.510 ms |

Logical liveness falls sharply, and absolute expert/communication time tends to
fall as the ready pool drains. But the expected strong oscillation does not
appear: communication fraction spans only 36.8--39.5% on GSM8K and 38.8--42.3%
on HumanEval. HumanEval late is modestly more communication-heavy; GSM8K late is
not.

The physical wave size is also not a pure function of normalized phase. In the
fresh cross-state captures, local source rows were GSM8K 248/40/8 and HumanEval
144/200/16. HumanEval wave 40 is larger than wave 0 despite occurring later.
Ready-pool drain, block position, and request completion dominate the concrete
shape. This weakens a controller based only on “early/middle/late.”

TensorCore and NVLink utilization figures use explicit time-share proxies because
hardware counters are privilege-blocked. They are not labeled as measured
bandwidth/utilization.

# FrontierEP routed-MoE projection

SIMULATED-EP4/8-EP2-CALIBRATED; not E2E latency.

| Target | B1 ms/request | F1 ms/request | actual rollout gain | NFE-normalized gain |
|---|---|---|---|---|
| EP4 | 1145.246 | 1220.236 | -6.548% | 0.979% |
| EP8 | 1190.366 | 1253.036 | -5.265% | 2.172% |

The normalization exposes the per-forward sparse-work effect independently from changed NFE. It does not rescue the failed semantic contract.

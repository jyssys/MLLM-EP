# Q1 — DeepEP communication share

Measured on TRUE TP2-DP2-EP4. For each layer invocation, the component values are taken from the rank with maximum measured MoE-stage latency, so the reported shares remain a consistent critical-rank decomposition.

| Phase | Dispatch mean / share | Expert mean / share | Combine mean / share | Communication share |
|---|---:|---:|---:|---:|
| Visual prefill | 2.2522 ms / 74.138% | 0.5855 ms / 19.274% | 0.2001 ms / 6.588% | 80.726% |
| B8 decode | 0.4282 ms / 40.251% | 0.4707 ms / 44.246% | 0.1649 ms / 15.503% | 55.754% |

Verdict: **COMMUNICATION_FIRST_CLASS_BOTTLENECK**.

The component trace and clean serving-wall runs are separate. Deferred event synchronization occurs after a sampled 48-layer boundary, not in each layer's measured interval.

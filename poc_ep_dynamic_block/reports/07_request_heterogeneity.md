# Request heterogeneity

The isolated mini=1 tournament executes every request independently under five
fixed B choices and seven mixed schedules. It gives future knowledge every
advantage, but is not a deployable throughput configuration.

| task | preferred fixed-B histogram | safe dynamic wins | median safe gain | mean safe gain |
|---|---|---:|---:|---:|
| GSM8K | B8:1, B32:4, B64:1, B128:2 | 7/8 | 1.014% | 3.171% |
| HumanEval | B16:1, B32:6, B64:1 | 6/8 | 0.369% | 0.518% |

Two GSM8K requests show approximately 9.15% and 10.26% isolated gains, but the
median is below the 5% gate and the configuration sacrifices batching. The
result establishes genuine request heterogeneity while falsifying the stronger
claim that heterogeneity translates into material aggregate scheduling value.

All values come from actual trajectories. No independent block latencies are
summed.

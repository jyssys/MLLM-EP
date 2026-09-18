# Temporal-Edge EP systems oracle

This is an exact route-removal counterfactual over the existing trace and an EP2-calibrated EP4/EP8 simulator. It is not a runtime measurement. E1 source-side removes dispatch, expert work and combine for every still-live STAY edge; owner-side removes expert work but retains communication.

| target | live-MASK pair reduction | whole physical route reduction | source stage gain | owner stage gain |
|---|---:|---:|---:|---:|
| EP4 | 68.821% | 0.515% | 0.140% | 0.107% |
| EP8 | 68.821% | 0.515% | 0.040% | 0.008% |

The apparent 68.8% live-MASK opportunity is diluted to about half a percent of all physical expert routes by prompt/prior/current-decoded full rows. Even impossible identity reuse is far below the suggested 5% whole-stage signal. Cache peak for E1 raw outputs is 0.95 MiB per invocation. E2/E3/E4 are evaluated in the output-stability report; no production cache or kernel was built.


## Actual-output policies on targeted 16-request trace

| policy | EP4 stage gain | EP8 stage gain | EP8 whole-route reduction |
|---|---:|---:|---:|
| E1_identity | 0.0439% | 0.0177% | 0.1545% |
| E2_l2_1 | 0.0000% | 0.0000% | 0.0000% |
| E2_l2_2 | 0.0000% | 0.0000% | 0.0001% |
| E2_l2_5 | 0.0013% | 0.0001% | 0.0052% |
| E2_l2_10 | 0.0042% | 0.0007% | 0.0185% |
| E3_practical | 0.0000% | 0.0000% | 0.0001% |
| E4_conservative | 0.0000% | 0.0000% | 0.0001% |

E2 uses future branch outputs and is an oracle. E3/E4 use current-observable input and router drift selected on requests 0--7 and evaluated once on 8--15. All gains are simulated routed-MoE stage gains, not E2E latency.


## Actual-output policies on targeted 16-request trace

E1/E2/E3/E4 in this table are applied only to the five layers with measured branch vectors; the GSM8K-128 all-19-layer E1 structural upper bound remains in the first table.

| policy | EP4 stage gain | EP8 stage gain | EP8 whole-route reduction |
|---|---:|---:|---:|
| E1_identity | 0.0439% | 0.0177% | 0.1545% |
| E2_l2_1 | 0.0000% | 0.0000% | 0.0000% |
| E2_l2_2 | 0.0000% | 0.0000% | 0.0001% |
| E2_l2_5 | 0.0013% | 0.0001% | 0.0052% |
| E2_l2_10 | 0.0042% | 0.0007% | 0.0185% |
| E3_practical | 0.0000% | 0.0000% | 0.0001% |
| E4_conservative | 0.0000% | 0.0000% | 0.0001% |

E2 uses future branch outputs and is an oracle. E3/E4 use current-observable input and router drift selected on requests 0--7 and evaluated once on 8--15. All gains are simulated routed-MoE stage gains, not E2E latency.

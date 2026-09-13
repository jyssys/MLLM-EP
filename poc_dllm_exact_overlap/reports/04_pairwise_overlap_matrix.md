# Exact pairwise overlap matrix

Each cell uses real LLaDA2 hidden/routes, resident BF16 expert weights, five
warmups, 30 measured repetitions, same-rank CUDA events, and the request-critical
maximum across four ranks. Diagnostic generation output remains bitwise
identical at the answer-sequence boundary.

## Stable findings

| candidate | GSM8K waves 0 / 30 / 60 saving | HumanEval waves 0 / 40 / 80 saving | interpretation |
|---|---|---|---|
| dispatch + attention | -0.050 / -0.075 / -0.078 ms | -0.075 / -0.061 / -0.130 ms | stable contention; kill |
| remote dispatch + local expert | -0.021 / -0.024 / -0.055 ms | -0.034 / -0.020 / -0.092 ms | exact but slower; kill |
| dispatch i+1 + expert i | 0.043 / 0.026 / 0.023 ms | 0.025 / 0.025 / 0.040 ms | small complete-wave overlap |
| expert i + combine i-1 | 0.066 / 0.033 / 0.034 ms | 0.045 / 0.048 / 0.070 ms | repeatable complete-wave overlap |
| 3-stage complete waves | **0.140 / 0.106 / 0.126 ms** | **0.115 / 0.126 / 0.186 ms** | strongest local result |

The three-stage pipeline has median overlap efficiency 0.48--0.61 across all six
states and is exact. It is nevertheless not a direct single-request result:
same-request denoising wave `t+1` cannot begin before `t` produces the decision,
and the best-static `mini32` runtime has one physical wave at a time.

These wave IDs are not relabeled as clean early/middle/late categories. In the
measured request, ready-pool draining and block progress interact with logical
liveness: the source-rank row counts were GSM8K 248/40/8 and HumanEval
144/200/16. The full phase analysis therefore uses the exhaustive prior trace;
these points are concrete large/medium/small execution shapes.

## Same-wave candidates

Shared expert and routed EP are independent, but the entire shared path is only
2.98--3.50% of clean wall. Starting shared work beside dispatch regresses in all
states; combine+shared is positive in some states but cannot exceed that ceiling.
The local-vs-remote split is negative in every state. These outcomes reject the
two largest plausible within-request exact opportunities without building a
production scheduler.

The observer-heavy generation wall (roughly 6.8--12.7 s in pair runs) includes
hundreds of replayed operations and is not compared to the clean baseline.
Machine-readable medians and p10/p90 ranges are in
`PAIRWISE_OVERLAP_MATRIX.csv`.

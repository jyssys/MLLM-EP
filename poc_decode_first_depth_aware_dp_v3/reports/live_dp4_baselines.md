# Live DP4 baselines

These are physical online runs on four independent CDB replicas.
Requests receive a rank before prefill. No KV migration occurs.
The gate, weights, precision, cache policy, and generation cap remain unchanged across policies.
The controller sends requests to the authors' native arrival/scheduler loop.

Round-robin uses request order. Queue placement uses actual outstanding requests.
Length placement uses calibration output-length priors and known prompt length.
The cost placement uses calibration service-time priors, queue state, and nominal equal rank rates.
It is an inexpensive causal cost baseline, not a proven optimal CDB-aware scheduler.
Its label `strong_cost` must not imply dominance over every generic service-cost policy.

## Burst sweep

| C total | Policy | P95 E2E, ms | P95 ITL, ms | Requests/s | Idle/eligible witnesses |
| --- | --- | --- | --- | --- | --- |
| 1 | round_robin | 4200.7 | 18.89 | 0.238 | 0 |
| 1 | queue | 4189.9 | 18.92 | 0.239 | 0 |
| 1 | length | 4193.4 | 18.92 | 0.238 | 0 |
| 1 | strong_cost | 4196.1 | 18.94 | 0.238 | 0 |
| 2 | round_robin | 4134.2 | 18.92 | 0.476 | 0 |
| 2 | queue | 4120.1 | 18.88 | 0.478 | 0 |
| 2 | length | 4123.5 | 18.91 | 0.477 | 0 |
| 2 | strong_cost | 4126.3 | 18.93 | 0.477 | 0 |
| 4 | round_robin | 4124.3 | 18.88 | 0.954 | 57 |
| 4 | queue | 4124.1 | 18.89 | 0.953 | 56 |
| 4 | length | 4127.3 | 18.89 | 0.952 | 56 |
| 4 | strong_cost | 4121.0 | 18.86 | 0.954 | 56 |
| 8 | round_robin | 7252.1 | 20.26 | 1.080 | 43 |
| 8 | queue | 7183.8 | 19.91 | 1.095 | 39 |
| 8 | length | 7410.7 | 20.13 | 1.071 | 62 |
| 8 | strong_cost | 7337.4 | 20.14 | 1.089 | 58 |
| 16 | round_robin | 12191.7 | 49.96 | 1.169 | 61 |
| 16 | queue | 12131.3 | 50.33 | 1.187 | 52 |
| 16 | length | 12948.3 | 52.06 | 1.224 | 153 |
| 16 | strong_cost | 13137.1 | 51.10 | 1.198 | 106 |
| 32 | round_robin | 21743.4 | 179.11 | 1.291 | 211 |
| 32 | queue | 21895.6 | 179.63 | 1.280 | 219 |
| 32 | length | 20775.8 | 240.73 | 1.439 | 116 |
| 32 | strong_cost | 20809.4 | 238.50 | 1.510 | 84 |
| 64 | round_robin | 34722.1 | 429.32 | 1.681 | 403 |
| 64 | queue | 34735.6 | 431.69 | 1.681 | 409 |
| 64 | length | 33784.1 | 454.09 | 1.678 | 255 |
| 64 | strong_cost | 33008.7 | 486.39 | 1.777 | 262 |

The concurrency is the total burst size across four GPUs, not the per-GPU size.
Closed-loop and Poisson discovery use the C16 setting and 32 requests per case.
The Poisson rate is 1 request/s. Actual active concurrency is not fixed at 16.
The full non-burst concurrency/arrival-rate matrix is `NOT_RUN`.
All additional results and the frozen relative-goodput limit appear in the CSV/JSON artifacts.

The idle/eligible witness requires a completed rank and another rank's sampled runnable queue.
It is not a GPU timestamp measurement of collective waiting.
Multiple witnesses in one run are not independent statistical samples.
The witnesses show residual cross-rank work after one rank becomes idle.

## Output consistency

| C | Policy vs RR | Exact token requests | Exact depth-sequence requests |
| --- | --- | --- | --- |
| 16 | queue | 16/16 | 16/16 |
| 16 | length | 7/16 | 3/16 |
| 16 | strong_cost | 9/16 | 2/16 |
| 32 | queue | 32/32 | 32/32 |
| 32 | length | 13/32 | 1/32 |
| 32 | strong_cost | 17/32 | 3/32 |
| 64 | queue | 64/64 | 64/64 |
| 64 | length | 38/64 | 4/64 |
| 64 | strong_cost | 33/64 | 3/64 |

Identical placement controls retain their output traces.
Changed batch composition can produce different outputs and exit sequences.
We do not assert that every such difference comes only from BF16 rounding.
No full benchmark establishes quality parity between these placement arms.

The separate batched correctness probe passed 744 exact terminal-KV copy checks.
Its actual decoder-layer count also matched executed recurrent work.
The first probe failed because its counting assertion omitted the four prefill passes.
The corrected v2 probe uses the same model and cache code. The failed v1 log remains intact.
This diagnostic is not clean latency or a benchmark-quality pass.

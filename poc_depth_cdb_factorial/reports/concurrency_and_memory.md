# Concurrency, queues, and memory

Each C is a finite simultaneous burst, not C requests per GPU.
Every replica has the same 16-request resident limit and KV pool.
A0 queues requests that exceed its local capacity.
The reserve policy prevents those queued requests from allocating full resident KV immediately.

| C | Arm | Mean core batch | Refill events | Depth-work CV | Rank-finish CV | Peak allocated/GPU, GiB |
|---|---|---|---|---|---|---|
| 4 | A0 | 3.474 | 0 | 0.0000 | 0.0000 | 26.763 |
| 4 | A1 | 1.000 | 0 | 0.1458 | 0.1306 | 26.755 |
| 4 | A2 | 1.000 | 0 | 0.1458 | 0.1307 | 26.755 |
| 4 | A3 | 1.000 | 0 | 0.1458 | 0.1396 | 26.755 |
| 4 | A4 | 1.000 | 0 | 0.1458 | 0.1205 | 26.755 |
| 8 | A0 | 6.530 | 0 | 0.0000 | 0.0000 | 26.787 |
| 8 | A1 | 1.795 | 0 | 0.0686 | 0.0322 | 26.772 |
| 8 | A2 | 1.774 | 0 | 0.0428 | 0.0166 | 26.760 |
| 8 | A3 | 1.106 | 1122 | 0.0776 | 0.0847 | 26.772 |
| 8 | A4 | 1.079 | 1285 | 0.0324 | 0.0386 | 26.760 |
| 16 | A0 | 12.257 | 0 | 0.0000 | 0.0000 | 26.836 |
| 16 | A1 | 3.493 | 0 | 0.0331 | 0.0466 | 26.804 |
| 16 | A2 | 3.479 | 0 | 0.0191 | 0.0364 | 26.775 |
| 16 | A3 | 1.331 | 3487 | 0.0335 | 0.0784 | 26.804 |
| 16 | A4 | 1.293 | 3602 | 0.0175 | 0.0547 | 26.775 |
| 32 | A0 | 12.494 | 0 | 0.0000 | 0.0000 | 26.836 |
| 32 | A1 | 6.343 | 0 | 0.0645 | 0.0328 | 26.870 |
| 32 | A2 | 6.323 | 0 | 0.0546 | 0.0543 | 26.816 |
| 32 | A3 | 1.564 | 7422 | 0.0511 | 0.1598 | 26.870 |
| 32 | A4 | 1.524 | 7210 | 0.0511 | 0.0914 | 26.816 |
| 64 | A0 | 12.327 | 0 | 0.0000 | 0.0000 | 26.836 |
| 64 | A1 | 12.465 | 0 | 0.0463 | 0.0422 | 26.947 |
| 64 | A2 | 12.316 | 0 | 0.0213 | 0.0354 | 26.845 |
| 64 | A3 | 1.915 | 15199 | 0.0602 | 0.1276 | 26.947 |
| 64 | A4 | 1.873 | 15356 | 0.0183 | 0.0840 | 26.845 |

All cache tensors together occupy 24.0234375 GiB per replica.
The nominal request-row pool is 24 GiB. Padding accounts for the remaining 24 MiB.

Peak allocated memory excludes some non-PyTorch CUDA context allocations.
Peak reserved memory appears separately in the raw measurements.
The configured launch limit is not the achieved recurrent batch size.
A deficit from 16 slots is not a direct measurement of GPU hardware idle.

## Operating-point diagnostics

| C | Arm | Output tokens | Scheduled loop work | P95 queue, ms | Trace critical-rank core, ms | Token-exact vs A0 | Rank-summed no-local-request tail, s | Whole-GPU idle with backlog |
|---|---|---|---|---|---|---|---|---|
| 4 | A0 | 1024 | 2929 | 6.67 | 4993.11 | 100.00% | 0.000 | UNRESOLVED |
| 4 | A1 | 947 | 2727 | 3.61 | 4131.06 | 50.00% | 2.393 | UNRESOLVED |
| 4 | A2 | 947 | 2727 | 3.27 | 3844.62 | 50.00% | 2.392 | UNRESOLVED |
| 4 | A3 | 947 | 2727 | 3.07 | 3856.75 | 50.00% | 2.396 | UNRESOLVED |
| 4 | A4 | 947 | 2727 | 2.86 | 3739.73 | 50.00% | 2.394 | UNRESOLVED |
| 8 | A0 | 2048 | 5968 | 9.48 | 4999.79 | 100.00% | 0.000 | UNRESOLVED |
| 8 | A1 | 1971 | 5733 | 5.63 | 4382.50 | 50.00% | 0.533 | UNRESOLVED |
| 8 | A2 | 1971 | 5819 | 5.06 | 3933.79 | 37.50% | 0.458 | UNRESOLVED |
| 8 | A3 | 1971 | 5744 | 5.89 | 6464.93 | 62.50% | 3.376 | UNRESOLVED |
| 8 | A4 | 2048 | 6043 | 5.85 | 6829.59 | 50.00% | 1.413 | UNRESOLVED |
| 16 | A0 | 4096 | 12269 | 16.11 | 5917.15 | 100.00% | 0.000 | UNRESOLVED |
| 16 | A1 | 4096 | 12183 | 9.24 | 4799.78 | 50.00% | 1.530 | UNRESOLVED |
| 16 | A2 | 4096 | 12204 | 9.25 | 4508.01 | 50.00% | 0.793 | UNRESOLVED |
| 16 | A3 | 4096 | 12153 | 8.95 | 12040.22 | 50.00% | 6.399 | UNRESOLVED |
| 16 | A4 | 4096 | 12150 | 7.78 | 11713.60 | 50.00% | 3.646 | UNRESOLVED |
| 32 | A0 | 8106 | 24450 | 6691.08 | 11005.83 | 100.00% | 0.000 | UNRESOLVED |
| 32 | A1 | 7797 | 23388 | 16.39 | 5187.32 | 43.75% | 1.047 | UNRESOLVED |
| 32 | A2 | 7797 | 23352 | 12.32 | 4984.58 | 40.62% | 1.476 | UNRESOLVED |
| 32 | A3 | 8008 | 24007 | 16.48 | 22138.84 | 53.12% | 21.828 | UNRESOLVED |
| 32 | A4 | 7797 | 23370 | 12.57 | 19935.78 | 46.88% | 7.893 | UNRESOLVED |
| 64 | A0 | 16043 | 48114 | 20287.41 | 22080.67 | 100.00% | 0.000 | UNRESOLVED |
| 64 | A1 | 16043 | 48191 | 29.21 | 5854.51 | 73.44% | 1.609 | UNRESOLVED |
| 64 | A2 | 16043 | 48130 | 23.94 | 6094.50 | 75.00% | 1.226 | UNRESOLVED |
| 64 | A3 | 15949 | 47606 | 29.66 | 34146.20 | 48.44% | 23.645 | UNRESOLVED |
| 64 | A4 | 16055 | 47977 | 23.71 | 32849.93 | 46.88% | 14.392 | UNRESOLVED |

Clean statistics use medians across five paired repeats.
Token parity is the median fraction of exact request outputs versus the paired A0 run.
The core stream interval uses one separate trace run.
The rank-summed tail adds independent rank resource seconds, not serial wall-clock delay.
The observer report identifies trace runs whose output drift prevents a clean overhead estimate.

## Waiting and actual GPU work

`rank_time_analysis.csv` records per-rank core stream milliseconds and request/loop totals.
The CUDA-event timeline records prefill, recurrent, prelude, and coda intervals.
Post-completion tails identify ranks with no remaining local request while another rank continues.
Sampled reservation blocks and admission state distinguish memory pressure from resident-cap pressure where possible.

An unoccupied compute stream is not necessarily an idle whole GPU.
Exact whole-GPU idle-with-backlog duration remains `UNRESOLVED`.
Every required performance cell contains a live measured time.
No simulated time replaces an unavailable execution.

## Optional operating points

C128/C256 are `NOT_RUN` optional sensitivities.
The study prioritizes five repeats of every required configuration and 128 quality questions per domain.
The fixed local memory limit would queue additional requests.
No measured C128/C256 latency or memory-stability claim follows from that structural expectation.
Rolling and Poisson arrivals are outside this completed burst factorial.
They are not inferred from burst makespan ratios.

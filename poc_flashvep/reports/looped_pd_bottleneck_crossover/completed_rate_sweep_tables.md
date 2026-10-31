# Completed G2 Rate Sweeps

Evidence: MEASURED. G2 is 3,072 input tokens and 128 controlled output tokens.
All rows use C256 and the default pool. Each row has 64 requests.
The factor 0.8 has three independent starts per model.

| Model | Factor | Start | Target requests/s | Joint passed/N | TTFT p95 ms | TPOT p95 ms | E2E p95 ms | Finite goodput requests/s |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| huginn | 0.4 | 0 | 0.0630 | 39/64 | 22125.83 | 103.68 | 35156.02 | 0.0439 |
| huginn | 0.6 | 0 | 0.0945 | 17/64 | 56124.79 | 103.56 | 69170.10 | 0.0282 |
| huginn | 0.8 | 0 | 0.1260 | 9/64 | 107761.29 | 102.69 | 120769.21 | 0.0177 |
| huginn | 0.8 | 1 | 0.1260 | 3/64 | 98347.50 | 104.07 | 111515.95 | 0.0058 |
| huginn | 0.8 | 2 | 0.1260 | 6/64 | 170162.27 | 104.42 | 183328.01 | 0.0094 |
| huginn | 1.0 | 0 | 0.1574 | 3/64 | 216768.71 | 108.42 | 229545.83 | 0.0053 |
| huginn | 1.2 | 0 | 0.1889 | 2/64 | 280616.83 | 104.65 | 293236.40 | 0.0034 |
| ouro | 0.4 | 0 | 0.3585 | 26/64 | 7510.89 | 39.00 | 12334.01 | 0.1645 |
| ouro | 0.6 | 0 | 0.5378 | 2/64 | 17409.90 | 37.93 | 20945.21 | 0.0185 |
| ouro | 0.8 | 0 | 0.7171 | 0/64 | 44493.07 | 39.05 | 47831.16 | 0.0000 |
| ouro | 0.8 | 1 | 0.7171 | 3/64 | 15608.94 | 43.03 | 19690.70 | 0.0353 |
| ouro | 0.8 | 2 | 0.7171 | 4/64 | 13652.83 | 42.91 | 17420.59 | 0.0451 |
| ouro | 1.0 | 0 | 0.8963 | 0/64 | 34553.89 | 45.11 | 38069.29 | 0.0000 |
| ouro | 1.2 | 0 | 1.0756 | 0/64 | 33495.59 | 41.48 | 37961.84 | 0.0000 |

No measured point in these C256 sweeps reaches 90% joint attainment.
This does not locate a zero capacity. Lower rates and tuned-engine sweeps remain unmeasured.

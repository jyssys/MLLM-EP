# Independent-start Uncertainty

Evidence: MEASURED runs with TRACE_ANALYSIS intervals. Each setting has three independent starts and 64 requests per start.
The intervals use Student-t with two degrees of freedom. Bounds intersect the valid parameter range.
A p95 entry summarizes the mean of three per-start p95 values. It is not a pooled p95.

| Model | Cell | Chunk / pool / slots | Target requests/s | Goodput mean [95% CI] | Attainment mean [95% CI] | TPOT p95 mean [95% CI], ms |
| --- | --- | --- | --- | --- | --- | --- |
| huginn | G2 | 2048 / 26624 / 8 | 0.2260 | 0.0376 [0.0000, 0.0782] | 0.1771 [0.0021, 0.3521] | 132.0453 [117.3893, 146.7013] |
| huginn | G2 | 256 / 18432 / 8 | 0.1260 | 0.0110 [0.0000, 0.0262] | 0.0938 [0.0000, 0.2102] | 103.7249 [101.4466, 106.0032] |
| ouro | G2 | 256 / 8192 / 8 | 0.7171 | 0.0268 [0.0000, 0.0857] | 0.0365 [0.0000, 0.1173] | 41.6641 [36.0305, 47.2978] |
| ouro | G3 | 2048 / 22528 / 32 | 2.5206 | 0.0367 [0.0000, 0.1507] | 0.0312 [0.0000, 0.1339] | 101.8609 [94.9571, 108.7647] |

These are setting-level intervals, not paired method-improvement intervals.

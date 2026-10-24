# Native baseline audit

The checkpoint, source, and physical topology are pinned in `source_lock.json`.
`source_audit.md` lists the actual cache and attention call sites.
The model executes 24 shared layers four times.
The cache contains 96 logical layer/depth namespaces.

The baseline is native RLT Async/Mixed with FA4 and CUDA Graphs.
It uses a common 18,432-block physical pool and 16-token pages.
The allocator keeps all depth planes in HBM.
Prefix caching is disabled.
Each measured request therefore performs its full prefill again.

| Prompt tokens | Independent processes | Mean TTFT, ms | Mean delivered ITL, ms | Logical KV, GiB | Peak allocated, GiB |
| --- | --- | --- | --- | --- | --- |
| 1024 | 2 | 201.82 | 18.759 | 0.750 | 57.113 |
| 2048 | 2 | 380.09 | 20.359 | 1.500 | 57.113 |
| 4096 | 2 | 754.13 | 22.961 | 3.000 | 57.113 |
| 8192 | 2 | 1521.89 | 28.802 | 6.000 | 57.113 |
| 16384 | 2 | 3023.05 | 39.989 | 12.000 | 57.113 |
| 32768 | 2 | 6152.68 | 62.748 | 24.000 | 57.113 |
| 65408 | 2 | 14298.74 | 108.342 | 47.906 | 57.113 |


Evidence: `LIVE_MEASURED_SERVING`, bounded hardware lane.
The output cap is 64 tokens, and EOS is ignored in this table.
TTFT includes request prefill in a resident, warmed engine.
It excludes model loading and deployment startup.
ITL uses actual frontend-delivered output tokens, not recurrent callbacks.
The two starts use disjoint corpus families and are not paired quality samples.
`baseline_summary.csv` contains all tested B1/B2/B4/B8 cells.

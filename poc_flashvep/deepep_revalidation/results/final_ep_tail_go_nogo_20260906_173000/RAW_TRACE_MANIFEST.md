# Raw trace manifest

Raw live artifacts are intentionally retained on the shared workspace but not
committed into Git because they contain hundreds of megabytes of JSONL/NPZ
data.  The committed summaries identify the exact source paths and run
parameters.

| run | raw invocation rows | contextual rows | status |
|---|---:|---:|---|
| `direct_stock_run1` | 0 | 0 | hook import error (`NameError`); excluded |
| `direct_stock_run2` | 49,536 | 0 | HT live sanity; context patch not yet enabled |
| `direct_stock_run3` | 37,248 | 4,608 | HT, 2,304 logical contextual |
| `direct_stock_run4` | 86,400 | 13,824 | HT, 6,912 logical contextual; primary direct analysis |
| `direct_low_latency_attempt` | 0 | 0 | unmodified V1 NVSHMEM queue-depth assertion |

The raw files include `invocations.jsonl`, `stages.jsonl`, per-invocation
route NPZ files, `waves.dp*.json`, `topology.dp*.json`, `run.log`, and
`driver_status.json`.  The analyzer consumes only contextual rows and
same-device TP-deduplicated spans.

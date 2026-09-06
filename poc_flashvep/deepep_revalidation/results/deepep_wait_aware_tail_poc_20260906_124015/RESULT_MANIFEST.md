# Wait-aware tail result manifest

The timestamped directory contains raw `invocations.jsonl`, `stages.jsonl`,
routes, topology proofs, and `run.log` for every run.  Raw traces are retained
locally for replay but are intentionally not committed because they occupy
approximately 4.5 GB.  Committed bounded summaries are:

* `policy_run_summary.csv/json`, `event_readiness.csv`, `stage_decomposition.csv`
* `EVENT_READINESS_ANALYSIS.md`, `POLICY_COMPARISON.md`, `gate_summary.json`
* `runtime_topology.json`, `runtime_metadata.json`, `oracle_ids.txt`

Policies and repetitions:

* STOCK: 3 runs
* ALWAYS_SYNC: 3 runs
* unconditional communication-stream drain: 3 runs
* ORACLE selective stream drain: 3 runs (two ID-selection variants)
* ONLINE_SIMPLE (`previous dispatch >=1 ms`): 3 runs

All runs used physical GPUs 1–4 only, model snapshot
`9c4b90e1e4ba969fd3b5378b57d966d725f1b86c`, and TP2/DP2/EP4 DeepEP HT.

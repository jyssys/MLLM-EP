# Staging and graph accounting

Atomic G4 executes a prompt chunk through the native fused prefill path.
Split and Grouped retain intermediate prompt hidden rows in native staging storage.
Staged G4 retains the staged path but executes four loops per group.
Mixed also packs compatible decode and prompt rows.
It is not merely another G value.

All compared main paths receive the same 2,048-token budget and 54-GiB KV pool.
Native decode graphs remain enabled, with maximum graph batch equal to configured N.
The maximum graph-entry budget is 32 for every path.
The observer records actual graph captures/replays and actual selected rows.
Configured C must not substitute for actual mixed prompt rows.

`quantum_observations.csv.gz` preserves staged allocation bytes, graph entries, and capture/replay deltas.
Staged allocation size is not actual copy traffic.
CUDA stream intervals include waits and launch gaps.
They do not isolate graph staging, GEMM, and KV service into additive E2E components.
That finer decomposition remains PARTIAL.

At P8177/N8/C1024, staged G4 already improves mean cohort-p99 relative to Atomic.
The additional G2-versus-staged-G4 change has an interval spanning zero.
These controls prevent attributing the entire raw G2 tail change to recurrent grouping.

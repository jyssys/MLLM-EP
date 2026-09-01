# Live trace inventory

The full per-rank JSONL streams are retained in the local archive below. They
are intentionally not duplicated in this compact, commit-friendly result
directory because the four runs total roughly 1.5 GiB.

| run | topology | budget | scope | trace directory |
|---|---:|---:|---|---|
| `20260901_1530_A` | TP2/DP2/EP4 | 16384 | primary, warmup=1, measured=2 | `../dp_ep_arrival_skew_two_topologies_20260901_1530_A/` |
| `20260901_1600_B` | TP1/DP4/EP4 | 16384 | primary, warmup=1, measured=2 | `../dp_ep_arrival_skew_two_topologies_20260901_1600_B/` |
| `20260901_1645_A_8192` | TP2/DP2/EP4 | 8192 | stress, warmup=1, measured=1 | `../dp_ep_arrival_skew_two_topologies_20260901_1645_A_8192/` |
| `20260901_1720_B_8192` | TP1/DP4/EP4 | 8192 | stress, warmup=1, measured=1 | `../dp_ep_arrival_skew_two_topologies_20260901_1720_B_8192/` |
| `20260901_4096_A` | TP2/DP2/EP4 | 4096 | stress, warmup=1, measured=1 | `../dp_ep_arrival_skew_two_topologies_20260901_4096_A/` |
| `20260901_4096_B` | TP1/DP4/EP4 | 4096 | stress, warmup=1, measured=1 | `../dp_ep_arrival_skew_two_topologies_20260901_4096_B/` |
| `20260901_2048_A` | TP2/DP2/EP4 | 2048 | stress, warmup=1, measured=1 | `../dp_ep_arrival_skew_two_topologies_20260901_2048_A/` |
| `20260901_2048_B` | TP1/DP4/EP4 | 2048 | stress, warmup=1, measured=1 | `../dp_ep_arrival_skew_two_topologies_20260901_2048_B/` |

Each source run contains `scheduler_trace/`, `raw_live/rank*.jsonl`,
`raw_live/arrival_*.jsonl`, topology proofs, backend proofs, `schedule.json`,
and driver records. The compact `analysis/` CSV in this directory is the
reproducible aggregation of all four runs; autotuner probe rows with fewer
than 64 assignments are excluded by the preregistered analyzer filter.

# External CUDA Graph observations

## Scope

These tests observe the existing optimized native baseline.
They do not implement a candidate architecture.
Both models use Base checkpoints, BF16, fixed R4, TP2, and asynchronous scheduling.
The physical devices are GPU4 and GPU5.

Each context uses one frozen request shape and three observed repeats.
Each request has 128 forced output tokens.
This controlled workload does not measure task accuracy.
The protocol also measures three clean requests before and after each observed group.

## Timing boundary

The worker records an external event pair around every actual CUDA Graph replay.
The worker also records outer model and sampling intervals.
Events remain on each rank's native stream.
The worker reads them after the request.
No per-layer host synchronization enters the measured request.

The analysis selects one cumulative-critical rank for each request.
It selects the rank with the largest sum of complete graph intervals.
It retains that rank for the complete local-window calculation.
It does not add component maxima from different ranks.
It does not align clocks across GPUs.

The local window starts at the first cached graph start and ends at the last cached graph end.
Prefill and the first and final sampling intervals are outside this denominator.
The graph fraction is not an E2E fraction.
The complete graph envelope can contain compute, communication, and rank waiting.
Its internal decomposition remains unresolved.

## Results

| Model | Context tokens | Median graph interval, ms | Graph fraction of local window | Observer versus clean-before | Observer versus clean-after |
|---|---:|---:|---:|---:|---:|
| 1.4B | 1024 | 6.79139 | 97.7986% | -1.2249% | +0.4626% |
| 1.4B | 4096 | 7.21747 | 97.9514% | -0.1341% | -1.8789% |
| 1.4B | 8192 | 7.33866 | 97.9237% | +0.1551% | +0.0962% |
| 2.6B | 1024 | 13.54253 | 98.8440% | +0.2882% | +0.2090% |
| 2.6B | 4096 | 14.39610 | 98.9081% | +0.1124% | +0.0842% |
| 2.6B | 8192 | 15.56672 | 98.9896% | +0.0802% | +0.0036% |

Percent differences compare median parent E2E latency within the stated before/observed/after group.
They are observer checks, not candidate speedups.
The small sample does not establish broad timing equivalence.

The selected-rank inter-graph gaps total approximately 19.3–20.2 ms across the 127 cached graph replays.
Those gaps are unattributed.
They are not proven GPU idle time or removable recurrence-specific overhead.
The dominant measured local envelope lies inside graph execution for these shapes.
This observation does not reject opportunities inside those graphs.

## Evidence

- Contract: [external_graph_timing_contract.md](external_graph_timing_contract.md).
- Base 1.4B: `outputs/graph_intervals_base14_P2_v1/interval_analysis/summary.json`.
- Base 2.6B: `outputs/graph_intervals_base26_P2_v1/interval_analysis/summary.json`.
- Raw per-request and per-rank events remain in those output directories.
- Reproduction scripts: `native_graph_intervals.py` and `analyze_graph_intervals.py`.

Earlier heavy profiler timings remain separate.
They do not supply headline E2E fractions.
This report makes no HBM-traffic, graph-internal bottleneck, or new-architecture performance claim.

# Measured C/G cost surface

Evidence: DIAGNOSTIC CUDA-event lane, not headline clean serving.
The main matrix contains 96 process starts and 1,728 policy cohorts.
Each start contains six policies and three request cohorts.
The GPU can change between starts, but never between paired policy cells.

Shapes are P512/N1, P2033/N4, and P8177/N8.
C is 64/128/256/512/768/1024/1536/2048.
Policies are Atomic, Split G1, Grouped G2, Grouped G3, staged G4, and Mixed.
One long request arrives at 250 ms among short background requests for N4/N8.
Configured N does not mean every request has P prompt tokens or remains active throughout.
Natural output has a 32-token cap in this diagnostic lane.

P8177/N8/C1024, twelve cohorts per policy:

| Policy | Median prefill quantum | Mean long-request TTFT | Pooled observed ITL p99 |
| --- | --- | --- | --- |
| Atomic | 103.424 ms | 932.532 ms | 451.370 ms |
| Staged G4 | 102.791 ms | 1,032.223 ms | 284.260 ms |
| Grouped G2 | 51.970 ms | 1,163.531 ms | 251.784 ms |
| Split G1 | 26.872 ms | 1,164.629 ms | 159.498 ms |
| Mixed | 28.061 ms | 973.398 ms | 133.596 ms |

These are not selected best policies or clean SLO results.
`cost_surface_summary.csv` includes every measured C and policy.
G3 combines G3 first groups and G1 remainders.
A pooled G3 median is not the cost of one fixed three-loop call.
The quantum table preserves actual G and starting depth.

Smaller C increases recurrent submissions.
At P8177/C64, Atomic median prefill quantum is 76.173 ms and long-request TTFT is 10,425.825 ms.
At C2048, these values are 133.642 ms and 754.231 ms.
This is a latency-efficiency trade-off, not a universal preference for small chunks.

Reducing G shortens individual recurrent quanta but introduces more scheduling opportunities.
Some opportunities interrupt an unfinished output token repeatedly.
The data contain 43,216 gaps with at least two prefill submissions among 176,787 observed token gaps.
This diagnostic distribution is not a production opportunity-frequency estimate.

Staged G4 changes the executor path without changing group depth.
Its tail change shows why Atomic-vs-G2 alone cannot identify a loop-group effect.
Mixed changes row packing and actual prompt-token selection.
Its configured C can exceed its actual selected prompt rows.

The scheduling-impact target includes host submission, queued waits, and safe completion.
The device-stream interval can include dependency waits and host launch gaps.
Neither target is pure kernel service.
Do not sum overlapping intervals across streams into an E2E breakdown.

Observers evolved through schemas 1/2/3 during early collection.
Their counts are 21,494 / 160,795 / 348,537.
Missing features remain explicit.
Each process-start policy block retains one observer version.
Read the contention and observer-control reports before interpreting latency.

Artifacts include `results_raw.csv`, `quantum_observations.csv.gz`, and `full_token_exposure.csv.gz`.
`p2_comparison_pairs.csv` records four-start paired intervals for each C and shape.
Figures include `cost_heatmaps.png`, `diagnostic_pareto.png`, and the offline budget envelope.

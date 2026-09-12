# Candidate tournament and request-level oracles

## Mapping rule

For each candidate, the optimistic request bound is:

`observer-light MLP share × detailed inner-component share × removable fraction`.

This assumes every saved component lies on the request critical path and
therefore favors candidates. Detailed traces are never treated as clean wall
time. The strongest static microbatch per topology is the baseline.

## Tournament

| Candidate | Perfect E2E | credible/feasible E2E | Verdict |
|---|---:|---:|---|
| A. Denoising expert-work packing | 1.80% | 1.80% | KILL `<5%` |
| B. Refinement microbatch coalescing | 47.36% measured vs default | 47.36% | existing `mini_batch_size`; trivial engineering |
| C. Block-scoped replication | 14.33% | ≤2.99% before copy cost | KILL feasible `<5%` |
| D. Perfect denoising load shaping | 5.34% | ≤5.34% | KILL `<8%` |
| E. Phase-adaptive topology | not identifiable | 0% feasible | dual layout/transition infeasible in HBM/runtime |
| F. Fresh-work compaction | 25.00% full routed; 10.58% expert-only | excluded | direct Epoch collision |
| G. Route/dispatch-plan reuse | 5.38% | 1.57% | KILL feasible `<5%` |
| H. Attention–EP overlap | 19.29% zero-contention | 0% established | generic overlap collision; no live causal PoC |
| I/J. Locality/support grouping | no independent oracle | 0% | bottleneck not activated; crowded prior art |
| K. Phase precision | not measured | 0% | no quality-valid path |

## Important derivations

### C. Replication

Lag-1 branch overlap is 74.29%, so proportional removal of every corresponding
dispatch/combine cost would imply 14.33%. But an EP call remains because mean
destination fanout is 3.07. The primitive sweep attributes only 20.88% of
M=256 communication to payload-sensitive time above the observed fixed floor.
Even with free weight copies, the resulting upper bound is 2.99%; copy time,
HBM, metadata, and changed branches can only reduce it.

### D. Load shaping

Perfectly replacing each observed critical rank with mean rank work removes
29.48% of expert time. Expert work is 34.13% of inner MoE and MLP is 53.06% of
block time, producing a 5.34% optimistic E2E ceiling. This already assumes a
free semantics-preserving mechanism.

### F. Fresh work

Only 41.56% of physical rows are decision-live on average. Proportional removal
of the whole routed pipeline maps to 25.00%; routed-expert compute alone maps
to 10.58%. This validates Epoch's motivation, not a successor contribution.

## Faster-backend durability

The replication upper bound falls from 2.99% at current communication cost to
2.24%/1.50%/0.75% at cost multipliers 0.75/0.5/0.25. Perfect rank balance stays
at 5.34% but is already below the candidate gate and has strong prior-art
collision. No surviving candidate is both large and durable.

## Decision

No novel candidate reaches the 8% HOLD gate, so no method prototype is
authorized. The strongest novel credible oracle is 5.34%.

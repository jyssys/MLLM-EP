# EP replay dataset

`EP_SHAPE_REPLAY.jsonl` contains 40 deterministic cases.

- 25 real cases sample five quantile/end-point shapes from each of five size
  bins. They retain the exact per-source fresh top-k expert IDs, expert
  histogram, rank load, remote assignment count, and model provenance.
- 15 controlled cases hold M in {32,128,512} and top-k=8 while varying uniform
  fanout-4, mild skew/fanout-4, strong skew/fanout-2, fanout-1 local, and
  fanout-1 remote routes.

The controlled cases are synthetic communication controls, not LLaDA2 routing
evidence. `EP_SHAPE_SUMMARY.csv` exposes their exact fanout, remote fraction,
expert support, and rank-load CV. For example, fanout-1 local/remote controls
hold identical expert rows and rank totals but move remote fraction from 0 to
1; uniform and skew controls hold M and remote fraction while changing
critical-rank concentration.

All real cases and all controlled cases were executed on physical GPUs 4--7.
The controlled results are in `CONTROLLED_KERNEL_BENCH.csv`; their raw logs are
under the ignored result root. This addition checks that the existing-path
envelope is not an accidental property of one natural route geometry.

Reproduction order:

1. `build_shape_atlas.py` reconstructs the measured compacted atlas and 25 real
   cases.
2. `add_replay_controls.py` appends the 15 deterministic controls and writes
   `EP_SHAPE_SUMMARY.csv`.
3. `benchmark_existing_paths.py` executes the communication semantics with
   identical random BF16 inputs and uniform router weights for each policy.

Expert GEMM is intentionally not replaced by an identity in the request
oracle. Communication replay is joined with the measured compact-expert
envelope from the same LLaDA2 substrate.

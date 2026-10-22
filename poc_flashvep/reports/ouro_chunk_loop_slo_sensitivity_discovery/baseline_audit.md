# Baseline audit

The historical experiment is immutable commit `c458f6a2a764f667a0da6784bcb2592b36626837`.
The audit matched eleven required report/data files against that commit.
It also checked all 375 lossless raw archives against recorded hashes and local originals.
See `baseline_audit.json` for individual evidence and hashes.

The historical verdict remains `CONDITIONAL_FIXED_QUANTUM_ONLY`.
This follow-up does not replace it with GO or NO-GO.
Historical means of cohort quantiles remain distinct from pooled token quantiles.
The historical all-capped 128-token generation is not complete-answer accuracy evidence.

The CPU reanalysis produced 11,172 absolute-SLO cells from actual request timestamps.
It did not infer request compliance from aggregate p99.
The selected historical lanes exclude warm-up and the aborted Atomic-first cold lane.
The warm confirmation still carries its original fixed-order limitation.

The new fixed-grid lane uses the same historical prompt families for continuity.
It does not call them a new held-out dataset.
New public conversation/article calibration and held-out families are frozen separately.
Their sources and selection rules appear in `traffic_workload_manifest.json`.

The new measurements reserve 54GiB of physical KV per engine.
The old serving configuration reserved 12GiB.
Thus old-to-new performance changes are not attributed solely to policy.
Within a new ordinary policy block, backend, token budget, and KV reserve stay fixed.
No-chunk is a separately labeled budget/admission control.

Every new policy block executes its policies on the same physical GPU.
The dispatcher changes policy order across starts and workers.
`GPU_policy_crossover.csv` records the actual assignments.
`GPU_interference_calibration.csv` records the solo and concurrent controls.

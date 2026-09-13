# EP-aware dynamic block diffusion deep PoC

This workspace follows
`poc_flashvep/reports/ep_aware_dynamic_block_diffusion_deep_poc_spec.md`
from the shared project checkout.  It studies variable diffusion block size on
the LLaDA2.0-Flash 100B dense-TP4/routed-EP4 path using physical GPUs 4--7.

Evidence boundaries:

- `results/clean/`: request-level clean timing and outputs;
- `results/trace/`: observer-heavy physical EP traces;
- `analysis/`: derived tables and figures;
- `reports/`: evidence-first stage reports.

The experiment runner records every launch in `ATTEMPT_LOG.csv` and all
four-GPU wall time in `GPU_TIME_LOG.csv`.  The most important machine-readable
products are `STATIC_BLOCK_SWEEP.csv`, `BLOCK_EP_METRICS.csv`,
`BLOCK_QUALITY_METRICS.csv`, `DYNAMIC_SCHEDULE_ORACLE.csv`, and
`POLICY_COMPARISON.csv`.  `DYNAMIC_VS_GLOBAL_STATIC.csv` applies the strict
comparison required by the contract: an actual mixed trajectory must beat the
strongest measured fixed-B/mini/threshold frontier, not merely a fixed policy
bundled into the same diagnostic run.

Runtime changes are preserved as `dinfer_dynamic_block.patch`; the external
dInfer worktree itself is not vendored.  See `REPRODUCE.md` for the exact model,
environment, patch application, GPU mapping, and analysis commands.

The original spec SHA256 and source path are recorded in `SPEC_PROVENANCE.md`.

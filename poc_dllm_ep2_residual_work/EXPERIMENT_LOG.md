# Experiment log

All task-owned GPU launches must use exactly `CUDA_VISIBLE_DEVICES=6,7`.
Observer-heavy captures are never used as clean latency evidence.

## 2026-09-11

- Revalidated the patched dInfer substrate at true TP1/DP2/EP2 on physical
  GPUs 6 and 7.  The two ranks own global experts 0--31 and 32--63.
- Captured an execution trace containing DeepEP HT layout, notify, dispatch,
  cached-notify-combine, combine, Triton expert, and MoE reduction kernels.
- Collected three observer-heavy temporal trajectories: 6, 19, and 29 model
  forwards.  The 19-forward trace contains exact post-hoc individual branch
  outputs for all 16 MoE layers; the other two cover layers 0, 7, and 15.
- Collected three independent clean engine restarts: 620.63, 615.33, and
  673.00 ms.  The 620.63 ms median is the request denominator.
- Calibrated assignment-removal oracles with the previously measured DeepEP
  HT M=2--256 stage curve.  No production cache, variable-work kernel, or
  scheduler was implemented.
- After GPU measurements, started repository-requested utilization work on
  physical GPUs 6 and 7 only.  It is not counted as experiment time.

## Decision checkpoint

The dominant logical/physical gap (52.20% of assignments) is exactly the
stable-decoded/live-lane opportunity claimed by Epoch.  Independent residual
oracles are below 5% request E2E: perfect router removal 2.69%, stable-layout
reuse 0.025%, and 5%-change fresh-branch reuse 0.023%.  The direction therefore
hits the user-defined `EP2 NO-GO` gate.

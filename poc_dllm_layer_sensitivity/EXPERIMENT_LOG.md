# Experiment Log

## 2026-09-13 — setup

- Created isolated project worktree from validated discovery commit `43184db`.
- Created isolated dInfer worktree from instrumented true-EP4 commit `9132ce9`.
- Confirmed previous best-static substrate is submitted batch 32 / mini 32.
- Source audit located exact routed/shared merge and block residual boundaries;
  causal interventions can be inserted without changing router decisions or
  dense attention semantics.

## 2026-09-13 — live EP4 campaign

- Revalidated the best-static EP4 GSM8K baseline and exact baseline output/NFE
  invariants for both bounded tasks.
- Captured observer-separated layer stability on GSM8K and HumanEval.
- Executed all-layer routed-MoE bypass and stale-routed sweeps, representative
  full-layer/shared-routed decomposition, eight-layer contiguous groups, and
  noncontiguous greedy policy attacks.
- Total task-owned live GPU time: 5,313 seconds across 11 four-GPU runs,
  equivalent to 1.4758 hours of four-GPU wall time and 5.9033 GPU-hours.
- No optimization prototype was launched because no cross-task,
  final-trajectory-safe candidate passed the 8% request-level gate.

## 2026-09-13 — decision

- Final label: `CHARACTERIZATION-SIGNAL`.
- Retained the current-step-versus-future-trajectory staleness trap as the main
  system finding.
- Killed full-layer refresh, routed-MoE selective refresh, shared-only
  reduction, and periodic stale refresh on safety/headroom/prior-art gates.
- Restarted the repository-owned utilization workload on physical GPUs 0--3
  only after all task measurements completed.

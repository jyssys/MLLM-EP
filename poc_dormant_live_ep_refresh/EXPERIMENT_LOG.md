# Experiment log

- 14:55 KST: audited Window-Diffusion at `d8bb349`; reproduced active/window
  transition with a deterministic synthetic masked sequence.  Full model smoke
  was unavailable because LLaDA-8B-Base was not local.
- 15:30--15:37: six clean independent EP4 launches, three per task.  Physical
  GPUs 4--7 only; mini32 best-static path.
- 15:37--15:40: one CUDA-event timing trace per task.  Observer-heavy request
  wall excluded from all performance claims.
- 15:40--15:45: one all-layer route/shape trace per task.
- 15:45--15:48: one detailed value trace per task at layers 1/8/16/24/31.
- 15:49--16:02: built acceptance plans and future-aware census/oracles.  The
  task-owned utilization process was kept off while GPU measurement was active.
- 16:02--16:15: one loaded-model causal sweep per task, seven policies.  Exact
  work was still computed and outputs were replaced; only quality causality is
  claimed.
- 16:15 onward: restarted task-owned utilization on physical GPUs 4--7 and
  performed CPU-only analysis, proxy fitting, audit, reports, and validation.

The raw campaign is intentionally retained outside Git because it includes
large structural traces.  Derived CSVs, plans, scripts, reports, and an
instrumentation patch are versioned.

# Compute-budget campaign progress

Status: Phase 0 audit and contention calibration setup.
No new performance or quality result exists yet.

The authoritative contract is `ouro_8gpu_compute_budget_joint_cg_optimization_poc_spec.md` in the parent directory.
The new engine checkout preserves the qualified G2/G3 runtime patch.
The previous deadline-study data remain frozen.
The paused native-vLLM PP program remains excluded.

The initial calibration uses the same per-device CPU affinity in solo and parallel conditions.
Quiet interference controls reserve otherwise idle GPUs without burners.
Other idle periods use recognized owned burners.

Next tasks are completed-token fairness, per-request legal C/G selection, and readiness-separated service measurement.
No future decode-token completion predictor is a prerequisite.

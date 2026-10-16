# Burner restoration race

## Observed failure

The first fixed-history control fails during engine initialization.
It produces no client generation or benchmark output.
Logical cuda:0, physical GPU4, has 19.96 GiB free at that point.
The unchanged engine requests 67.31 GiB at memory utilization 0.85.
The earlier clean-start sample shows 81,090 MiB free on GPU4.
The model configuration itself does not change.

## Cause and controls

The original guard restores GPU4 burner worker 2024129 after the HF reference finishes.
The added post-exit check sees no GPU context during that worker's CPU bootstrap.
It incorrectly requests a second restore through launcher 2024202.
That launcher creates GPU4 worker 2024298.
The next guard's initial inventory contains 2024129, but not the later worker.
The later worker acquires GPU4 memory during model startup.

The surviving context PID2024597 resolves to burner worker 2024298.
Both workers belong to the project, not another user.
The scoped cleanup never signals another user's process or an unassigned GPU.
The ranked hypotheses were duplicate restoration, a late burner child, and foreign allocation.
The recorded ancestry and launch order support the first two together.

## Repair

The post-exit check now reads the original restoration ledger.
It recognizes a live project burner or launcher before that process acquires a GPU context.
It does not duplicate a pending restore.
It still restores a genuinely idle device after a late model-context release.
Only the assigned GPUs remain eligible.
The model weights, prompt, sampler, graph policy, and memory-utilization setting remain unchanged.

## Verification

The regression fails before the repair because the normal path creates an extra launcher.
The expected launcher count is zero, but the observed count is one.
The repaired guard passes seven CPU unit checks.
Those checks include pending CPU bootstrap, late context release, exact masks, and unrelated-process protection.
The second fixed-history attempt passes the previously failing initialization stage with identical engine arguments.
Its output comparisons remain a separate scientific result.

The final validation suite contains 15 passing implementation checks.
The historical different-history reproducibility property remains a failed test.
The repair does not change that evidence or imply output determinism.

## Scientific boundary

The full native quality collection ends before this restoration race occurs.
Its 460 recorded generations do not change.
The failed first history attempt remains preserved as `restart_same_history_default_v1`.
The initialization-only retry uses `restart_same_history_default_v2`.
This is experiment-control repair, not a model optimization or a source of speedup.

Evidence: `results/restore_fix_retry_lock.json`, `results/restore_race_regression_red.json`, and `results/validation_execution_v3.json`.

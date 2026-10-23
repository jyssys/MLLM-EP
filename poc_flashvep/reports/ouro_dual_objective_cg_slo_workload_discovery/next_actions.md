# Resume instructions

The task is RUNNING. Do not issue a final result from this checkpoint.
Read the authoritative spec, current research state, run manifest, live PIDs, and coordinator locks first.
The previous parent NO_GO_INCREMENTAL verdict remains unchanged.

The new branch is `flashvep/ouro-dual-objective-cg-slo-workload-poc`.
The first checkpoint is `7a8b8d61`. It contains Phase A, offline objectives, controls, and registered validation.
The full current SHA comes from Git. Do not assume this checkpoint is the final commit.
The old staged binary-diff hash must remain `7018be31bd8a517574711055a72afbd6ae90737e4a585578c98ec255dbf7e1d9`.
Commit only the new `ouro_dual_objective_cg` package and this report directory.

## Completed work

- Read all 400 spec lines and the required reporting skill references.
- Preserve GPU0–3 processes. Only GPUs4–7 may start or stop campaign/burner work.
- Reproduce ordered G-only/Joint plans and outputs: 144/144.
- Reproduce 660 old selector calls exactly. Of 180 fresh calls, 76 have multiple feasible C choices.
- U2 selects fixed-C2048 behavior in all 76 cases. Frontier/clipping does not explain everything.
- Evaluate all O0–O13 in 112 weight settings on historical TRAIN selection geometry.
- Complete 24 contention starts/144 cohorts and 32 isolated numerical comparisons.
- Diagnose a WorkLedger engine-reference leak with a failing weak-reference test.
- Preserve all four failed adaptive-smoke v1 attempts. All four reviewed v2 attempts pass.
- Run the 22-arm VALIDATION screen. Keep objective runtime code frozen during this queue.

## Pipeline

The first validation coordinator finished.
The follow-up coordinator completed TRAIN tuning and froze O6 before FINAL.
It currently owns the coordinator lock and executes the registered subplans.
The sequence is alternative-state audit, event audit, FINAL characterization, serial quiet confirmation, then full-answer quality.
The first two audits are complete.
The quality plan has a separate short preflight through `execution_dependencies.json`.
This preflight does not change the frozen4096-token quality cap or question set.
Check the process state before launching a second coordinator.
It must not race the current dispatcher or overwrite failed/complete attempts.

The follow-up pipeline now has `final_plan.py` and `grade_quality.py`.
It registers30 parallel characterization jobs, six serial quiet blocks, and the nine-arm quality pilot after selection.
The frozen FINAL cap rule raises every arm to4096 if VALIDATION shows more than5% caps in any cohort.
This rule applies before FINAL outcomes. The original1024-cap results remain unchanged.
Both the four-worker and serial quiet lanes must remain separate in all tables and intervals.
Do not weaken the fixed-C G-only controls. They share objectives, telemetry, model, fairness, and budget.
Keep both cost-tuned C2048 and C1024 G-only where required by the frozen finalist.

## Remaining work

1. Check `state/followup.json`, live PIDs, and completed markers. Do not launch a duplicate queue.
2. Complete direct candidate-versus-control quality comparisons after the preserved evaluator finishes.
3. Preserve the completed audits:8/8 exact initial states,87/96 observer token parity,770 measured budget units.
4. Rerun mechanism analysis and plotting after FINAL. The offline interaction and constructed-state audits already exist.
5. Analyze actual FINAL quality-matched serving. Do not infer wins from score diversity.
6. Review capacity gates. No 1,000-request capacity result exists yet.
7. Write final reports and exact scope limitations. Keep measured and modeled lanes separate.
8. Archive raw attempts, publish hashes, restore owned GPU4–7 burners, and commit/push only this PoC.

Never resume the suspended native-vLLM Loop PP project.
Do not start GPU0–3 experiments or change their burners.

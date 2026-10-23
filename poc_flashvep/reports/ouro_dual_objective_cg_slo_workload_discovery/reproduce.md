# Reproduce and resume

Repository: `/home/esjung/MLLM-EP-github`.
Branch: `flashvep/ouro-dual-objective-cg-slo-workload-poc`.
The parent result stays fixed at `821b11653695d0a470bd88bfcc997ae1bbcf5bc9`.
`model_source_locks.json` records the unchanged RLT revision and G2/G3 patch hash.
The model snapshot is `574fa66cb8bf5abdc979642d01cf2b79b16bfab1`.
No native-vLLM Loop PP implementation is part of this work.

## CPU diagnosis and software checks

```bash
cd /home/esjung/MLLM-EP-github
/home/esjung/anaconda3/bin/python -m poc_flashvep.ouro_dual_objective_cg.action_diagnostics
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 /home/esjung/anaconda3/bin/python -m poc_flashvep.ouro_dual_objective_cg.objective_search
/home/esjung/anaconda3/bin/python -m unittest discover -s poc_flashvep/ouro_dual_objective_cg/tests -t . -p 'test_*.py'
```

The first analysis validates parent hashes before reading immutable quiet traces.
It writes only this new report directory.
The objective survey uses consumed historical TRAIN states for development, not a new blind test.
Do not refit the old cost model on previous HELDOUT residuals.

## GPU attempts

Check `state/coordinator.json`, live PIDs, and the dispatcher lock before starting any queue.
Do not start a second coordinator or dispatcher while its owner remains active.
Each completed attempt has hashes. Resume verifies them and skips that attempt.
A failure or interrupted identity requires a reviewed new attempt ID.
Never remove the original raw directory to force a rerun.

```bash
/home/esjung/anaconda3/bin/python -m poc_flashvep.ouro_dual_objective_cg.resources status
/home/esjung/anaconda3/bin/python -m poc_flashvep.ouro_dual_objective_cg.dispatch --manifest poc_flashvep/reports/ouro_dual_objective_cg_slo_workload_discovery/p0_contention_plan.json --manage-burners
/home/esjung/anaconda3/bin/python -m poc_flashvep.ouro_dual_objective_cg.dispatch --manifest poc_flashvep/reports/ouro_dual_objective_cg_slo_workload_discovery/p1_numerical_plan.json --manage-burners
/home/esjung/anaconda3/bin/python -m poc_flashvep.ouro_dual_objective_cg.dispatch --manifest poc_flashvep/reports/ouro_dual_objective_cg_slo_workload_discovery/p1_adaptive_smoke_retry_v2_plan.json --manage-burners
```

The dispatcher uses the existing GPU venv automatically.
It sets the RLT worktree on PYTHONPATH and masks one physical device before Torch loads.
Only IDs4–7 are accepted. GPU0–3 state remains untouched.

The broad validation handoff is:

```bash
/home/esjung/anaconda3/bin/python -u -m poc_flashvep.ouro_dual_objective_cg.campaign --validation
```

It waits for all reviewed adaptive smoke attempts to complete.
It then runs the frozen 22-arm, eight-job VALIDATION plan.
The final objective and capacity decision must not use FINAL outcomes for tuning.

## Resource handoff

```bash
/home/esjung/anaconda3/bin/python -m poc_flashvep.ouro_dual_objective_cg.resources restore --gpus 4,5,6,7
```

The wrapper calls the user's burner script with an explicit authorized mask.
It preserves unrelated work and does not change GPU0–3 processes.

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

The first analysis checks parent hashes before reading immutable quiet traces.
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

## Registered handoff and final analysis

Do not rerun `register`, `online_plan`, `select_final`, or `final_plan` over an existing freeze.
The follow-up coordinator now detects existing selection and FINAL locks before a resumed dispatch.
It never selects another objective after FINAL exposure.
Acquire its lock before any coordinator restart. A live owner means the queue is still active.
Resume granularity is a completed immutable job, not a mid-generation KV checkpoint.
An interrupted job needs a reviewed new attempt ID. Preserve its original identity, outputs, and failure evidence.

```bash
/home/esjung/anaconda3/bin/python -u -m poc_flashvep.ouro_dual_objective_cg.followup
```

The registered subplan sequence contains paired alternative diagnostics, observer audits, FINAL characterization, and serial confirmation.
The quality dependency adds a short callback/teardown preflight before the full-answer question pilot.
The preflight's256-token cap is not an accuracy result.
The main quality cap remains4,096 in all nine configurations.

After all measurement jobs finish:

```bash
taskset -c 128-143 /home/esjung/anaconda3/bin/python -m poc_flashvep.ouro_dual_objective_cg.analyze_serving --require-validation-complete
taskset -c 128-143 /home/esjung/anaconda3/bin/python -m poc_flashvep.ouro_dual_objective_cg.report_tables
taskset -c 128-143 /home/esjung/anaconda3/bin/python -m poc_flashvep.ouro_dual_objective_cg.mechanism_analysis
taskset -c 128-143 /home/esjung/anaconda3/bin/python -m poc_flashvep.ouro_dual_objective_cg.same_output_diagnostic
taskset -c 128-143 /home/esjung/anaconda3/bin/python -m poc_flashvep.ouro_dual_objective_cg.audit_analysis
taskset -c 128-143 /home/esjung/anaconda3/bin/python -m poc_flashvep.ouro_dual_objective_cg.analyze_resources
taskset -c 128-143 /home/esjung/anaconda3/bin/python -m poc_flashvep.ouro_dual_objective_cg.grade_quality
taskset -c 128-143 /home/esjung/anaconda3/bin/python -m poc_flashvep.ouro_dual_objective_cg.plot_results
taskset -c 128-143 /home/esjung/anaconda3/bin/python -m poc_flashvep.ouro_dual_objective_cg.validate_evidence
taskset -c 128-143 /home/esjung/anaconda3/bin/python -m poc_flashvep.ouro_dual_objective_cg.export_portable
taskset -c 128-143 /home/esjung/anaconda3/bin/python -m poc_flashvep.ouro_dual_objective_cg.archive_evidence
```

Do not run heavy analysis during the isolated measurement lane.
The preserved quality extractor imports from `/home/esjung/MLLM-EP-ouro-request-pp/poc_looped_exact_depth_chunk_pipeline/scoring.py`.
Import the campaign package before adding that older repository to Python's search path.
`quality_evaluator_lock.json` records the actual imported file and verifies its SHA256.
This does not implement a new evaluator or relax the preserved answer rules.

The portable gzip JSONL files contain completed natural outputs, executed prompt plans, and decision records.
`portable_evidence_manifest.json` records their byte counts and hashes.
The larger local raw archive also contains failed attempts, logs, full source snapshots, and telemetry.
Archive creation retains every original source file.
The archive command refuses to overwrite an existing archive. Check its recorded hash instead of recreating it in place.

## Supplementary observed budget coverage

The late CALIBRATION supplement preserves the unlaunched-v1 manifest before registering observed-v2 attempts.
It does not change primary FINAL settings or claims.
The completed six-job matrix covers20/80/240ms after the main40/120ms event audit.
Its CUDA-event timings are diagnostics, not clean serving headline results.

```bash
taskset -c 128-143 /home/esjung/anaconda3/bin/python -m poc_flashvep.ouro_dual_objective_cg.budget_supplement_analyze
taskset -c 128-143 /home/esjung/anaconda3/bin/python -m poc_flashvep.ouro_dual_objective_cg.profiling_analysis
taskset -c 128-143 /home/esjung/anaconda3/bin/python -m poc_flashvep.ouro_dual_objective_cg.slo_sensitivity
```

No GPU rerun is necessary to read the completed negative result.
Only an explicit reproduction or new study should start a completed plan again.

## Resource handoff

```bash
/home/esjung/anaconda3/bin/python -m poc_flashvep.ouro_dual_objective_cg.resources restore --gpus 4,5,6,7
```

The wrapper calls the user's burner script with an explicit authorized mask.
It preserves unrelated work and does not change GPU0–3 processes.

# Reproduction and resume

## Source and environment

Repository: `/home/esjung/MLLM-EP-github`.
Branch: `flashvep/ouro-8gpu-budget-joint-cg-optimization-poc`.
Model snapshot: `574fa66cb8bf5abdc979642d01cf2b79b16bfab1`.
External RLT commit: `4a350344027960a48384626aa646f1765fda6d4f`.
The removable runtime patch is `runtime_patch.diff` in this report directory.
Its working-tree diff hash is recorded in `model_source_locks.json`.
The external checkout is `/home/esjung/external/ouro-budget-joint-rlt`.

Use the existing GPU environment:

```bash
cd /home/esjung/MLLM-EP-github
export PYTHONPATH=/home/esjung/external/ouro-budget-joint-rlt:/home/esjung/MLLM-EP-github
/home/esjung/.venvs/ouro-chunk-loop-prefill/bin/python -m unittest discover -s poc_flashvep/ouro_budget_joint_cg/tests -t . -p 'test_*.py'
```

`environment_lock.json` records the installed backend versions and device topology.
The GPU environment has Torch and FlashAttention4.
The CPU analysis environment does not have Torch.
Do not use the CPU environment for modules that import the model or GPU benchmark.

## Inspect before execution

```bash
/home/esjung/anaconda3/bin/python -m poc_flashvep.ouro_budget_joint_cg.progress
/home/esjung/anaconda3/bin/python -m poc_flashvep.ouro_budget_joint_cg.resources status
```

Read `research_state.json`, the coordinator process-state files, and the active dispatcher lock.
Check live PIDs before creating another coordinator.
Do not restart a dispatcher while another dispatcher owns its lock.
Do not kill unrelated GPU work.

## Existing registered queues

```bash
/home/esjung/anaconda3/bin/python -m poc_flashvep.ouro_budget_joint_cg.dispatch --manifest poc_flashvep/reports/ouro_8gpu_compute_budget_joint_cg_discovery/p0_contention_plan.json --manage-burners
/home/esjung/anaconda3/bin/python -m poc_flashvep.ouro_budget_joint_cg.dispatch --manifest poc_flashvep/reports/ouro_8gpu_compute_budget_joint_cg_discovery/p1_numerical_plan.json --manage-burners
/home/esjung/anaconda3/bin/python -m poc_flashvep.ouro_budget_joint_cg.dispatch --manifest poc_flashvep/reports/ouro_8gpu_compute_budget_joint_cg_discovery/p1_fairness_plan.json --manage-burners
/home/esjung/anaconda3/bin/python -m poc_flashvep.ouro_budget_joint_cg.dispatch --manifest poc_flashvep/reports/ouro_8gpu_compute_budget_joint_cg_discovery/p2_service_plan.json --manage-burners
/home/esjung/anaconda3/bin/python -m poc_flashvep.ouro_budget_joint_cg.dispatch --manifest poc_flashvep/reports/ouro_8gpu_compute_budget_joint_cg_discovery/p2_controlled_service_plan.json --manage-burners
```

Each completed job has an immutable marker and artifact hashes.
Resume verifies those hashes and skips the completed job.
An interrupted identity or failure marker requires a reviewed new attempt ID.
Never overwrite or delete the original attempt.
The unused paired-service registration is not a required queue.
The controlled-endpoint queue supersedes it.

The cost-model freeze is immutable.
Do not run `--fit` over an existing frozen artifact.
Use a new model version and new blind families for any changed model.

## Budget pilot and quiet confirmation

```bash
/home/esjung/anaconda3/bin/python -m poc_flashvep.ouro_budget_joint_cg.dispatch --manifest poc_flashvep/reports/ouro_8gpu_compute_budget_joint_cg_discovery/p3_budget_pilot_plan.json --manage-burners
/home/esjung/anaconda3/bin/python -m poc_flashvep.ouro_budget_joint_cg.dispatch --manifest poc_flashvep/reports/ouro_8gpu_compute_budget_joint_cg_discovery/p5_quiet_confirmation_plan.json --manage-burners
```

The first queue uses all eight independent workers for a TRAIN diagnostic.
The second queue reserves all devices and measures one GPU at a time.
This avoids treating host interference as a policy gain.
All compared policies cross on the same device/process block.
Quiet confirmation uses CALIBRATION warm-up inputs and fresh FINAL_ONLINE measured inputs.

## Analysis

```bash
taskset -c 128-143 /home/esjung/anaconda3/bin/python -m poc_flashvep.ouro_budget_joint_cg.analyze
taskset -c 128-143 /home/esjung/anaconda3/bin/python -m poc_flashvep.ouro_budget_joint_cg.cost_interaction
taskset -c 128-143 /home/esjung/anaconda3/bin/python -m poc_flashvep.ouro_budget_joint_cg.budget_analysis
taskset -c 128-143 /home/esjung/anaconda3/bin/python -m poc_flashvep.ouro_budget_joint_cg.live_pairs
taskset -c 128-143 /home/esjung/anaconda3/bin/python -m poc_flashvep.ouro_budget_joint_cg.figures
```

Run heavy analysis after quiet serving windows.
The raw records remain the evidence source.
Recomputed tables do not become independent experiments.

## Quality and idle resources

The confirmation coordinator registers a B40 full-answer quality pilot after the quiet queue.
Its separate plan uses the preserved 64 GSM8K and 50 MATH questions.
It includes stock Atomic and stock Mixed controls.
It uses a 4,096-token cap with natural EOS and the preserved evaluator stops.
Quality timing is not a clean serving benchmark.

```bash
/home/esjung/anaconda3/bin/python -m poc_flashvep.ouro_budget_joint_cg.resources restore --gpus 0,1,2,3,4,5,6,7
```

The wrapper calls the existing burner script with an explicit eight-device mask.
It restores only unoccupied devices and preserves unrelated processes.
Quiet controls intentionally keep isolation-reserved devices free of burners.
All unallocated periods and the final handoff restore owned burners.

## Final evidence exports

The GPU campaign is complete. Do not restart completed queues to regenerate tables.
Use the CPU environment for these commands after all workers finish:

```bash
/home/esjung/anaconda3/bin/python -m poc_flashvep.ouro_budget_joint_cg.export_results
/home/esjung/anaconda3/bin/python -m poc_flashvep.ouro_budget_joint_cg.archive
```

The archive command preserves raw attempts and worker logs in local phase archives.
`raw_archive_manifest.json` records their absolute paths, sizes, and SHA256 hashes.
These large archives remain local. They are not part of the Git push.
The committed request tables, quality cases, run manifests, and source files identify their raw evidence.
`artifact_hash_manifest.json` records the published analysis artifacts and source files.
It excludes its own file, live resource state, logs, raw attempts, and local archives.
Do not modify analysis artifacts after this step without regenerating the manifest.

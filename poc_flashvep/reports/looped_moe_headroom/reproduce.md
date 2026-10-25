# Reproduce the three-model study

## Scope and safety

Run commands from `/home/esjung/MLLM-EP-github`.
The commands below are the actual module interfaces.
They are not the illustrative interfaces in the supplied spec.
Do not rerun a failed or interrupted ID by appending data.
Review the failure first.
Use a new ID for a corrected run.

The dispatcher acquires a campaign lease and checks GPU process ownership.
It only stops verified owned burner sessions.
Each worker receives one physical GPU in CUDA_VISIBLE_DEVICES.
The worker also checks its own NVML PID-to-UUID mapping.
Unused GPUs retain owned burners.
The burner script needs explicit CUDA_VISIBLE_DEVICES and VLLM_UTILIZE_GPUS values for the requested device scope.
Its current default device list is not all eight GPUs.

The resource helper reuses `poc_flashvep/ouro_budget_joint_cg/resources.py` from the existing repository.
It does not resume any prior scheduling or PP experiment.

## Inspect existing evidence without a GPU

Raw files are compressed per run in archives/.
The local raw/ and logs/ directories remain unchanged.
The archive manifest records every file hash.
Extraction checks those hashes and refuses conflicting existing files.

```bash
/home/esjung/anaconda3/bin/python -m poc_flashvep.looped_moe_headroom.archive --extract
taskset -c 128-143 /home/esjung/anaconda3/bin/python -m poc_flashvep.looped_moe_headroom.verify
taskset -c 128-143 /home/esjung/anaconda3/bin/python -m poc_flashvep.looped_moe_headroom.analyze
taskset -c 128-143 /home/esjung/anaconda3/bin/python -m poc_flashvep.looped_moe_headroom.reports
taskset -c 128-143 /home/esjung/anaconda3/bin/python -m poc_flashvep.looped_moe_headroom.plots
```

The CPU analysis environment needs NumPy and Matplotlib.
It does not load a model or initialize CUDA.
The analyzer can use workload_manifest_public.json without the full text manifest.
The full language passages remain local rather than in Git.

## Environment and model assets

The validated worker Python is `/home/esjung/.venvs/looped-moe-headroom-py312/bin/python`.
It uses Python 3.12.13 and PyTorch 2.11.0+cu129.
The native extension compiles with CUDA 12.8 for SM90.
`requirements.lock.txt` records installed direct versions, not a complete portable container image.
`environment.json` records the local read-only package-base configuration.
Do not modify that shared package base.

The official xLLM checkout is `/home/esjung/external/xllm-loop-ep-poc`.
Its pinned commit is `df0ed1ea415b55f0a944a291f90da9d47c98b5d5`.
The Foil source audit uses `SR-A-W/how-to-loop-moe` commit `0f9cc7351658c2437f75379e492f3920f0a0ec19`.
The actual Foil loader uses the pinned Hugging Face remote code.
Model revisions appear in model_pins.json.
Foil uses revision `b3d128b44cbefd54e2c8784224a110186c8931f1` and only its Foil-1 checkpoint.
All model weights remain outside Git under `/home/esjung/looped-moe-headroom-assets`.
The native IFM loader checks artifact-manifest hashes on every load.
foil_artifact_verification.json checks the downloaded Foil weight hash against its LFS object.

```bash
CUDA_VISIBLE_DEVICES='' /home/esjung/.venvs/looped-moe-headroom-py312/bin/python -m poc_flashvep.looped_moe_headroom.prepare --download
bash poc_flashvep/looped_moe_headroom/scripts/check_native_import.sh
CUDA_VISIBLE_DEVICES='' /home/esjung/.venvs/looped-moe-headroom-py312/bin/python -m poc_flashvep.looped_moe_headroom.tests.test_contract
```

The installation script documents the native extension build.
It is not a universal one-command environment installer.
Use the validated local dependency pins when optional packages conflict.
Do not replace the native IFM loader with Transformers.

## Reconstruct workload text

The public workload manifest contains dataset revisions, row IDs, and text hashes.
The reconstruction script fetches only those pinned public sources.
Its regression check reproduced all 508 frozen text hashes.
The pinned MATH source uses test.jsonl, not parquet.

```bash
CUDA_VISIBLE_DEVICES='' /home/esjung/.venvs/looped-moe-headroom-py312/bin/python -m poc_flashvep.looped_moe_headroom.reconstruct_workloads --verify-only
```

Without --verify-only, the script writes a missing full-text manifest.
It does not overwrite a different existing manifest.
Dataset licensing and pretraining overlap remain separate questions.

## Fresh run without overwriting the published evidence

Create a new result directory.
Set LOOPED_MOE_RESULTS to its absolute path.
Copy model_pins.json, workload_manifest_public.json, and the desired plan files into that directory.
Use the same environment variable for every command in that campaign.
The resource helper can identify burners from the published result directory.
Run preflight and reconstruct_workloads before inference.
The default ROOT still points to the published directory, so do not omit the override.

The copied plans retain the original run IDs inside the new root.
Saved job files are regenerated under that root's state/ directory.
Completion markers make a completed plan resumable without duplicate measurements.

## GPU execution order

The following commands resume the original root and skip completed runs.
For a fresh root, export LOOPED_MOE_RESULTS before running them.
Primary timing, corrected microbenchmarks, and heavy profiling use physical GPU0 sequentially.
Quality and discovery jobs rotate across eight independent workers.
No model-parallel communication is used.

```bash
/home/esjung/anaconda3/bin/python -m poc_flashvep.looped_moe_headroom.preflight
/home/esjung/anaconda3/bin/python -m poc_flashvep.looped_moe_headroom.dispatch poc_flashvep/reports/looped_moe_headroom/plan_p0_diagnostic_v5.json
/home/esjung/anaconda3/bin/python -m poc_flashvep.looped_moe_headroom.dispatch poc_flashvep/reports/looped_moe_headroom/plan_p1.json
/home/esjung/anaconda3/bin/python -m poc_flashvep.looped_moe_headroom.dispatch poc_flashvep/reports/looped_moe_headroom/plan_calibration.json
/home/esjung/anaconda3/bin/python -m poc_flashvep.looped_moe_headroom.dispatch poc_flashvep/reports/looped_moe_headroom/plan_heldout.json
/home/esjung/anaconda3/bin/python -m poc_flashvep.looped_moe_headroom.dispatch poc_flashvep/reports/looped_moe_headroom/plan_timing.json --gpus 0
/home/esjung/anaconda3/bin/python -m poc_flashvep.looped_moe_headroom.dispatch poc_flashvep/reports/looped_moe_headroom/plan_generation_fidelity.json
/home/esjung/anaconda3/bin/python -m poc_flashvep.looped_moe_headroom.dispatch poc_flashvep/reports/looped_moe_headroom/plan_generation_quality.json
/home/esjung/anaconda3/bin/python -m poc_flashvep.looped_moe_headroom.dispatch poc_flashvep/reports/looped_moe_headroom/plan_micro_v2.json --gpus 0
/home/esjung/anaconda3/bin/python -m poc_flashvep.looped_moe_headroom.dispatch poc_flashvep/reports/looped_moe_headroom/plan_heavy_profile.json --gpus 0
/home/esjung/anaconda3/bin/python -m poc_flashvep.looped_moe_headroom.dispatch poc_flashvep/reports/looped_moe_headroom/plan_independent_controls_successful.json
/home/esjung/anaconda3/bin/python -m poc_flashvep.looped_moe_headroom.dispatch poc_flashvep/reports/looped_moe_headroom/plan_strict_foil_load.json --gpus 0
```

For a new root, substitute the copied plan paths in these commands.
The plan named independent_controls_successful combines the 16 IFM controls and eight corrected Foil controls.
Do not run retry_controls against a fresh campaign.
That script explicitly requires the historical serialization failure.

The six contention plans use a synchronized wall-clock release after all workers report ready.
Run plan_contention_p1_r0/r1/r2 with --gpus 0.
Run plan_contention_p8_r0/r1/r2 with all eight GPUs.
Do not reuse an old state barrier for a different campaign.

## Final verification and preservation

```bash
CUDA_VISIBLE_DEVICES='' /home/esjung/.venvs/looped-moe-headroom-py312/bin/python -m poc_flashvep.looped_moe_headroom.provenance
taskset -c 128-143 /home/esjung/anaconda3/bin/python -m poc_flashvep.looped_moe_headroom.verify
taskset -c 128-143 /home/esjung/anaconda3/bin/python -m poc_flashvep.looped_moe_headroom.analyze
taskset -c 128-143 /home/esjung/anaconda3/bin/python -m poc_flashvep.looped_moe_headroom.reports
taskset -c 128-143 /home/esjung/anaconda3/bin/python -m poc_flashvep.looped_moe_headroom.plots
/home/esjung/anaconda3/bin/python -m poc_flashvep.looped_moe_headroom.archive
/home/esjung/anaconda3/bin/python -m poc_flashvep.looped_moe_headroom.resources restore --gpus 0,1,2,3,4,5,6,7
```

The final Git commit must include only this study's code, reports, and copied specification.
Existing dirty and staged files must remain unchanged.
Do not commit model weights, virtual environments, full language passages, or credentials.

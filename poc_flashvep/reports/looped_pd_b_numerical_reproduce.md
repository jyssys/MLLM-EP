# Reproduction guide for numerical and B transfer diagnosis

## Safety and immutable inputs

Run commands from `/home/esjung/MLLM-EP-github`.
Use Python `/home/esjung/.venvs/looped-pd-abc/bin/python`.
Use `PYTHONPATH=/home/esjung/external/looped-pd-abc-rlt:/home/esjung/MLLM-EP-github` for standalone operator commands.
The bounded PD launcher creates that environment itself.
It sets `OMP_NUM_THREADS=1` and disables tokenizer thread parallelism.

The baseline commit is `fea20edef840ed357ee0493c2f0dd69668c66c1f`.
The new branch is `flashvep/looped-pd-b-numerical-diagnosis`.
The frozen model pins are:

| Model | Revision | Precision | Fixed depth |
|---|---|---|---:|
| `ByteDance/Ouro-1.4B` | `574fa66cb8bf5abdc979642d01cf2b79b16bfab1` | BF16 | 4 |
| `tomg-group-umd/huginn-0125` | `bb6621b65e90b6a4b9b29ef88dc83866d450470c` | BF16 | 32 |

The installed RLT source revision is `6b164d3844d35ab8e753df1864f0c87dc180401a`.
Existing local changes remain part of that runtime.
The source lock and per-run source archives identify the actual code, not only the clean Git revision.
The environment uses Torch2.11.0+cu129, CUDA12.9, FA4 4.0.0b30 and NIXL1.4.1.
Huginn retains `ABC_HUGINN_REFERENCE=1` for the existing reference RNG convention.
The registry also contains a Dory pin. Dory is not loaded or tested in this task.

## GPU ownership

Check UUIDs and current process owners before every lease.
The helpers stop only recognized owned burner sessions.
They refuse an occupied device instead of terminating an unrelated model process.
Never run the old and new controllers together on the same pair.

```bash
/home/esjung/.venvs/looped-pd-abc/bin/python -m poc_flashvep.looped_pd_b_numerical_diagnosis.resources status --gpus 0,1,2,3,4,5,6,7
```

Before a standalone probe, stop burners and wait for the command to succeed.
Then launch the probe. Do not run acquisition and launch concurrently.
Restore burners only after that probe exits.
The quiet comparison reserves all eight GPUs empty, although its model uses one physical pair.
Burners on unused GPUs would contaminate its shared-resource comparison.

## CPU regression and preservation

```bash
/home/esjung/.venvs/looped-pd-abc/bin/python -m pytest -q poc_flashvep/looped_pd_b_numerical_diagnosis/test_diagnostics.py
/home/esjung/.venvs/looped-pd-abc/bin/python -m poc_flashvep.looped_pd_b_numerical_diagnosis.validation
/home/esjung/.venvs/looped-pd-abc/bin/python -m poc_flashvep.looped_pd_b_numerical_diagnosis.slowdown
/home/esjung/.venvs/looped-pd-abc/bin/python -m poc_flashvep.looped_pd_b_numerical_diagnosis.summarize
/home/esjung/.venvs/looped-pd-abc/bin/python -m poc_flashvep.looped_pd_b_numerical_diagnosis.finalize_analysis
```

`validation` checks1637 locked source files and the unrelated staged index.
The frozen red command below is expected to exit1.
It records the old failures. It is not expected to turn green after new experiments.

```bash
/home/esjung/.venvs/looped-pd-abc/bin/python -m poc_flashvep.looped_pd_b_numerical_diagnosis.audit --assert-parity
```

## Minimal GPU regression examples

These examples use a new run namespace.
They do not overwrite the recorded experiments.
Standalone operator replay uses visible device0 under the explicit physical mask.

```bash
CUDA_VISIBLE_DEVICES=4 PYTHONPATH=/home/esjung/external/looped-pd-abc-rlt:/home/esjung/MLLM-EP-github /home/esjung/.venvs/looped-pd-abc/bin/python -m poc_flashvep.looped_pd_b_numerical_diagnosis.teacher_replay --case ouro_residual --sample 233 --sizes 1,2,4,8 --prefill-chunk 256 --variant down_norm --track-all --kv-hash --name local_reproduce_ouro233_fixed
CUDA_VISIBLE_DEVICES=5 PYTHONPATH=/home/esjung/external/looped-pd-abc-rlt:/home/esjung/MLLM-EP-github ABC_HUGINN_REFERENCE=1 /home/esjung/.venvs/looped-pd-abc/bin/python -m poc_flashvep.looped_pd_b_numerical_diagnosis.teacher_replay --case huginn_repeat --sample 92 --sizes 1,2,4 --variant down_norm --track-all --kv-hash --name local_reproduce_huginn92_fixed
```

The script checks KV capacity before each batch size.
`NOT_RUN_HBM` means the requested geometry was not measured.
It must not be counted as an exact comparison.
Teacher forcing is not natural-generation answer accuracy.

Fresh finite PD reproduction:

```bash
/home/esjung/.venvs/looped-pd-abc/bin/python -m poc_flashvep.looped_pd_b_numerical_diagnosis.run --model ouro --variant down_norm --gpus 0,1 --count 8 --cap 128 --rate 1 --seed 2026103001 --tag local_pd_reproduce
```

This launches PD_G1, an independent PD_G1 repeat, then B.
Use `--probe` only for instrumented host timings.
Use `--kv-verify` only for byte checks.
Do not pool either instrumented path with clean serving latency.
Each run stores its contract, worker mapping, model/source revisions, request records and launch state.
The launcher reuses a completed matching contract and refuses a changed contract under the same name.

The exact qualification plans are in `qualification.py` and `huginn_400.py`.
The recorded Huginn400 gate stops before B because the baseline repeat fails.
Do not rerun that failed expansion as an accepted performance experiment.

The later Huginn adapter correction has a separate controller and namespace.
Inspect its PID and state before any restart.
It runs numerical200/400 qualification on four independent physical pairs.
It does not run a serving-capacity search.

```bash
/home/esjung/.venvs/looped-pd-abc/bin/python -m poc_flashvep.looped_pd_b_numerical_diagnosis.adapter_qualification
/home/esjung/.venvs/looped-pd-abc/bin/python -m poc_flashvep.looped_pd_b_numerical_diagnosis.adapter_postcheck
```

The postcheck waits for full acceptance before its small byte, order and HF-triage probes.
Do not start another copy of either controller while its recorded process remains alive.

The optional postcheck controllers were canceled at the user's earlier scope change.
The later RCC prerequisite requires explicit closure of byte and order evidence.
`release_checks.py` completes that closure in fresh namespaces without overwriting the canceled status.

```bash
/home/esjung/.venvs/looped-pd-abc/bin/python -m poc_flashvep.looped_pd_b_numerical_diagnosis.release_checks
```

The recorded terminal status is `INTERNAL_CORRECTNESS_RELEASE_PASS`.
Do not restart this command on GPUs assigned to a subsequent experiment.

## Clean transfer comparisons and generic control

`configs/quiet_plan.json` specifies the finite three-start comparison.
`configs/generic_queued_plan.json` specifies the later ordinary-PD readiness control.
`quiet_experiments.py` and `followup.py` provide the original controllers.
Do not start them again while an earlier controller remains alive.
Inspect `state/quiet_experiments.json` and `state/residual_followup.json` first.

`B_NUMERICAL_QUEUED_COMPUTE=1` changes B's producer readiness only.
`B_NUMERICAL_QUEUED_COMPUTE_GENERIC=1` applies the same diagnostic rule to ordinary PD_G1.
Both retain fixed-depth, producer-event and byte-inflight constraints.
Neither variant has the full200/400 acceptance qualification.
Their small paired results do not establish a loop-specific serving method.

## Nsight Systems

The tested profiler version is Nsight Systems2024.6.2.225.
The launch path adds `--nsys` and captures the measured NVTX range only.
It preserves child-process CUDA graph nodes and actual P2P copy events.
Heavy profiling remains separate from clean latency.

After a completed profile, export its report to SQLite.
Use a new destination if an export already exists.

```bash
nsys export --type sqlite --output /absolute/run/directory/timeline.sqlite /absolute/run/directory/timeline.nsys-rep
/home/esjung/.venvs/looped-pd-abc/bin/python -m poc_flashvep.looped_pd_b_numerical_diagnosis.nsys_analysis --run ouro_quiet_nsys_B
```

The parser computes interval unions and overlaps.
It checks actual P2P bytes against measured transport-counter deltas.
It does not add overlapping CUDA API, kernel and copy durations.
Actual HBM bandwidth counters remain unavailable.

## Raw evidence publication

The namespace retains uncompressed local raw files.
Git stores large teacher JSON, dataset JSON, Nsight reports and SQLite exports as lossless `.gz` files.
`configs/artifact_storage.json` records both SHA256 hashes and exact decompression checks.
The compression procedure does not delete the original files.

```bash
/home/esjung/.venvs/looped-pd-abc/bin/python -m poc_flashvep.looped_pd_b_numerical_diagnosis.artifacts
/home/esjung/.venvs/looped-pd-abc/bin/python -m poc_flashvep.looped_pd_b_numerical_diagnosis.artifacts --restore
```

The restore command refuses to replace different existing evidence.
The published dataset archives contain the exact replay inputs, not a new dataset sample.
Per-run source archives retain earlier diagnostic revisions when later instrumentation changes.

## Interpretation limits

Do not compare pre-correction and corrected latency as equal-work speedup.
Do not call the small HF sequence comparison benchmark answer accuracy.
Do not call finite N8/cap8 latency sustainable serving capacity.
Do not label host transfer observations as pure GPU DMA.
Do not restart A/C, RCC-PD or the stopped native-vLLM Loop PP project from this controller.

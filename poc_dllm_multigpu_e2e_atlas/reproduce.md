# Reproduction

## Scope and preservation

Run commands from `/home/esjung/MLLM-EP-e2e-atlas`.
Use only physical GPUs 4, 5, 6, and 7.
Check allocation and process ownership before any GPU command.
Do not terminate another user's job. Do not reuse a measured output directory.

The experiment uses the existing dInfer checkout at
`/home/esjung/external/dinfer-llada2-flash-poc`.
Base revision is `1ffeb961cd258bede74fcf5ca8a416ae6d57b18f`.
The pre-existing diff hash is
`7b148f52ab97e176ba84c57f28a0c40bacdc488e139135b8664569dbbab56e27`.
Its snapshot is committed in the parent PoC at
`poc_dllm_prefix_canvas_cp/patches/dinfer_pre_existing_native.diff`.
Do not apply that patch again on this server.
The current PoC does not edit external runtime source.

Use the cached non-preview checkpoint revision
`inclusionAI/LLaDA2.0-mini@dad945cac317da394b390f82c7b40691d8a881ed`.
Model weights are not committed. Package versions and source hashes are in the environment manifest.
No package upgrade or new engine is required.

```bash
cd /home/esjung/MLLM-EP-e2e-atlas
export CUDA_VISIBLE_DEVICES=4,5,6,7
atlas_python=/home/esjung/.venvs/llada2-flash-sglang-053/bin/python
```

The guard reuses exact-mask, same-user burner restoration.
It refuses competing non-burner processes on the assigned GPUs.
The local burner script defaults to another mask.
If manual restoration is necessary, explicitly set `VLLM_UTILIZE_GPUS=4,5,6,7`.
Do not start duplicate burners.

## Frozen data and initial audit

Use the committed calibration and held-out manifests.
Token IDs come from the parent PoC's hashed `frozen_tokens.json.gz`.
Do not regenerate or reselect the cohort from output quality or timing.

`scripts/inspect_runtime_and_collectives.py` created the initial audit.
It refuses to overwrite the frozen environment manifest.
`scripts/freeze_control.py` froze one control after new Stage A and architecture screening.
It also refuses to overwrite its record.
These scripts document chronology. They are not commands for replacing the committed freeze.

## Stage A native runs

Use new output names for independent reproduction.
The following names intentionally differ from the recorded campaign.

```bash
"$atlas_python" -m poc_dllm_multigpu_e2e_atlas.scripts.run_guarded \
  --output results/reproduction/native_p512 --timeout 900 -- \
  "$atlas_python" -m poc_dllm_multigpu_e2e_atlas.scripts.run_clean_baseline \
  --output /home/esjung/MLLM-EP-e2e-atlas/poc_dllm_multigpu_e2e_atlas/results/reproduction/native_p512 \
  --prefix 512 --limit 8 --repeats 3
```

Repeat this command at P4096, P8192, and P16384 with distinct output names.
The recorded sweep used `scripts/run_native_sweep.py` for those three contexts.
Only one graph set exists per Stage A process.
Full graph prefill succeeded at every measured context.
An eager-prefill fallback exists for a bounded P16K capture failure, but it was not used.

Each request has a 180-second watchdog. Each guarded job has a 900-second deadline.
Three warmups precede measured repeats.
Clean runs have no Atlas hooks. Coarse runs preserve every actual replay occurrence.
Parity-only runs retain native token updates after the request.
The worker completes GPU work and a four-rank barrier before its bounded process exit.
The parent guard performs scoped cleanup and restores burners.
Legacy resource-tracker warnings remain in logs.

## Nsight diagnostic lanes

The installed Nsight version is 2024.6. Use supported options only.
CPU sampling is unavailable. No permission change is needed.
External reports live in `/home/esjung/external/dllm-e2e-atlas-profiles/`.
Create a new profile basename for reproduction.

```bash
"$atlas_python" -m poc_dllm_multigpu_e2e_atlas.scripts.run_guarded \
  --output results/reproduction/nsys_graph --timeout 900 -- \
  nsys profile --trace=cuda,nvtx,osrt --cuda-graph-trace=graph:host-only \
  --sample=none --cpuctxsw=none --capture-range=cudaProfilerApi \
  --capture-range-end=stop --kill=none \
  -o /home/esjung/external/dllm-e2e-atlas-profiles/reproduction_graph \
  "$atlas_python" -m poc_dllm_multigpu_e2e_atlas.scripts.run_clean_baseline \
  --output /home/esjung/MLLM-EP-e2e-atlas/poc_dllm_multigpu_e2e_atlas/results/reproduction/nsys_graph \
  --prefix 16384 --limit 1 --profile
```

The recorded node runs use `--cuda-graph-trace=node` instead.
The source-correlated node run also uses `--eager-diagnostic`.
That option executes one additional eager forward from the same input.
It checks logits and records source/destination storage metadata.
Its work is not part of native serving latency.
Do not subtract its duration to invent an uninstrumented result.

Export each report with `nsys export --type=sqlite --output=NEW.sqlite REPORT.nsys-rep`.
Then run `scripts/parse_timeline.py --sqlite PATH --output NEW_DIRECTORY`.
The parser uses process plus correlation ID, not correlation ID alone.
Graph envelopes and node service are different accounts.
Large originals stay external. Their paths, sizes, and SHA-256 hashes are committed.

The separate low-overhead Torch run uses `--profile --torch-profile`, without Nsight.
It records every CUDA occurrence. It does not extrapolate the final replay.

## One exact full-model control

`compact_scatter.py` scopes the replacement to attention KV updates.
It clones logical values into independent contiguous storage and writes fresh canvas rows.
It never changes native source-cache values in place.
It restores Python methods after candidate graph capture.
Candidate clean replay has no wrapper in the hot path.
Unsupported tensor shapes fail assertions instead of silently using an approximation.

```bash
"$atlas_python" -m poc_dllm_multigpu_e2e_atlas.scripts.run_guarded \
  --output results/reproduction/compact_calibration --timeout 900 -- \
  "$atlas_python" -m poc_dllm_multigpu_e2e_atlas.scripts.run_optional_physical_control \
  --output /home/esjung/MLLM-EP-e2e-atlas/poc_dllm_multigpu_e2e_atlas/results/reproduction/compact_calibration \
  --prefix 16384 --split calibration --limit 8 --profile
```

Run held-out only after calibration correctness and timing pass.
Use a new output name and `--split held_out --limit 24`.
Omit `--profile` for that held-out run.
The secondary calibration uses P4096 with the same control and thresholds.
No other candidate or operating point is selected from held-out results.

Each request checks native/candidate parity before headline timings.
It checks final token IDs, retained acceptance updates, NFE, all 40 regular-graph KV tensors, and logits.
Held-out additionally asserts independent native/candidate output storage.
Three clean repeats use counterbalanced arm order.
One coarse repeat per arm supplies the separate warm-block envelope.

The optional paired Torch profiles include a distinct int16 clock anchor.
The anchor executes before the timed request.
Native anchor width exceeds the frozen 100-microsecond point gate.
Its fine attribution remains BOUNDED/PARTIAL. No threshold is relaxed.

## Offline analysis

These commands use committed evidence and do not allocate GPU work.
They replace derived tables and figures, not frozen manifests or measured raw records.

```bash
CUDA_VISIBLE_DEVICES='' "$atlas_python" -m unittest discover \
  -s poc_dllm_multigpu_e2e_atlas/tests -v
CUDA_VISIBLE_DEVICES='' "$atlas_python" -m poc_dllm_multigpu_e2e_atlas.scripts.analyze_critical_path
CUDA_VISIBLE_DEVICES='' "$atlas_python" -m poc_dllm_multigpu_e2e_atlas.scripts.analyze_repeated_overheads
CUDA_VISIBLE_DEVICES='' "$atlas_python" -m poc_dllm_multigpu_e2e_atlas.scripts.analyze_control
CUDA_VISIBLE_DEVICES='' "$atlas_python" -m poc_dllm_multigpu_e2e_atlas.scripts.finalize_atlas
```

The architecture-screening script records analytical capacity and dependency filters.
Alternative-layout performance remains unresolved without lower-TP/PP calibration.
Do not rerun a selection script to imply a new preregistration after results.
The original decision hash remains in `manifests/frozen_control.json`.

`finalize_atlas.py` uses committed compact trace exports and the process mapping.
External originals are needed only for an independent re-export or initial mapping reconstruction.
The server-specific final audit checks original dirty-state hashes and current burner ancestry.

## Evidence map

- `manifests/`: initial environment, geometry, frozen IDs, timing validity, and control chronology.
- `results/per_request.csv`: new standalone native baseline, not pooled historical timings.
- `results/per_refinement.csv`: actual all-occurrence outer intervals, live/accepted counts, and explicit unresolved fields.
- `results/e2e_phase_budget.csv`: disjoint host-wall scopes, not GPU compute percentages.
- `results/critical_path_intervals.csv`: disjoint rank-local GPU scopes, not a complete cross-rank dependency graph.
- `results/gpu_kernel_service.csv`: service counts and time, not E2E fractions.
- `results/copy_and_memory_classification.csv`: source-correlated physical D2D payload proof.
- `results/per_layer_or_graph_node.csv`: auxiliary heavy-trace service with checked source-order mapping.
- `results/per_rank_collectives.csv`: real NCCL type/count and labelled payload proxies.
- `results/control_per_request.csv`: paired clean native/compact measurements.
- `results/profiler_clock_alignment.json`: bounded clock offsets and unchanged validity decisions.
- `results/external_profile_inventory.json`: original Nsight report/SQLite hashes and paths.
- `results/completion_audit.json`: tests, preservation, and final GPU ownership.

Official benchmark quality and B2/B4 throughput are NOT_RUN.
The measured B1 control gain is real. Novel architecture performance is not established.

# Reproduction

## Preconditions

Use the pinned model revision and the two existing interpreter environments from `environment.md`.
Run commands from the repository root.
All GPU launchers must use a unique output directory under `poc_recurrent_asymmetry/outputs`.
The launcher refuses an existing directory.
Do not delete a failed run to repeat its name.

The recorded parent commit is `458c1fe9b43f08863308d3b661a4163a854cb9aa`.
The current branch contains the reused loaders, guard, ownership probe, and scoring helpers.
Frozen inputs are in `metrics/frozen_workload.json`.
Do not re-freeze calibration IDs after observing results.

## Commands

The following commands create fresh run names.
Use a new suffix if any target already exists.
The `gpu_session` guard sets and checks the exact GPU mask.
It also restores verified utilization workers after execution.

```bash
cd /home/esjung/MLLM-EP-recurrent-asymmetry
export CUDA_VISIBLE_DEVICES=4,5,6,7
REF=/home/esjung/.venvs/llada2-mini-reference/bin/python
NATIVE=/home/esjung/.venvs/cp-composition-vllm026/bin/python

$REF -m poc_recurrent_asymmetry.scripts.gpu_session \
  --gpus 4 --output outputs/baseline_repro1 --timeout 7200 -- \
  $REF -m poc_recurrent_asymmetry.scripts.validate_baseline

$REF -m poc_recurrent_asymmetry.scripts.gpu_session \
  --gpus 4 --output outputs/profile_P1024_repro1 --timeout 7200 -- \
  $REF -m poc_recurrent_asymmetry.scripts.run_sweep --context 1024 \
  --baseline poc_recurrent_asymmetry/outputs/baseline_repro1

$REF -m poc_recurrent_asymmetry.scripts.gpu_session \
  --gpus 5 --output outputs/profile_P4096_repro1 --timeout 7200 -- \
  $REF -m poc_recurrent_asymmetry.scripts.run_sweep --context 4096 \
  --baseline poc_recurrent_asymmetry/outputs/baseline_repro1

$REF -m poc_recurrent_asymmetry.scripts.gpu_session \
  --gpus 6 --output outputs/profile_P8192_repro1 --timeout 7200 -- \
  $REF -m poc_recurrent_asymmetry.scripts.run_sweep --context 8192 \
  --baseline poc_recurrent_asymmetry/outputs/baseline_repro1

$REF -m poc_recurrent_asymmetry.scripts.gpu_session \
  --gpus 7 --output outputs/profile_P16384_repro1 --timeout 7200 -- \
  $REF -m poc_recurrent_asymmetry.scripts.run_sweep --context 16384 \
  --baseline poc_recurrent_asymmetry/outputs/baseline_repro1

$REF -m poc_recurrent_asymmetry.scripts.gpu_session \
  --gpus 4,5,6,7 --output outputs/physical_repro1 --timeout 3600 -- \
  $REF -m poc_recurrent_asymmetry.scripts.run_physical_bundle

$REF -m poc_recurrent_asymmetry.scripts.gpu_session \
  --gpus 4 --output outputs/light_repro1 --timeout 3600 -- \
  $REF -m poc_recurrent_asymmetry.scripts.light_components

$REF -m poc_recurrent_asymmetry.scripts.gpu_session \
  --gpus 6 --output outputs/chunk_correctness_repro1 --timeout 1800 -- \
  $REF -m poc_recurrent_asymmetry.scripts.chunk_correctness

$REF -m poc_recurrent_asymmetry.scripts.gpu_session \
  --gpus 4 --output outputs/native_TP1_repro1 --timeout 5400 -- \
  $NATIVE -m poc_recurrent_asymmetry.scripts.native_control --tp 1 --allocator cuda_malloc \
  --reference-directory poc_recurrent_asymmetry/outputs/physical_repro1/physical

$REF -m poc_recurrent_asymmetry.scripts.gpu_session \
  --gpus 4,5 --output outputs/native_TP2_repro1 --timeout 5400 -- \
  $NATIVE -m poc_recurrent_asymmetry.scripts.native_control --tp 2 --allocator cuda_malloc \
  --reference-directory poc_recurrent_asymmetry/outputs/physical_repro1/physical

$REF -m poc_recurrent_asymmetry.scripts.gpu_session \
  --gpus 4,5,6,7 --output outputs/native_TP4_repro1 --timeout 5400 -- \
  $NATIVE -m poc_recurrent_asymmetry.scripts.native_control --tp 4 --allocator cuda_malloc \
  --reference-directory poc_recurrent_asymmetry/outputs/physical_repro1/physical
```

Do not run the four-GPU command concurrently with a subset command.
The original profile controller runs independent context workers on disjoint GPUs.
Its command and exit codes remain in `metrics/profile_controller.json`.
The reference sweeps depend on the recorded `baseline_v1` validation gate.
The native comparisons depend on the recorded `physical_v1` HF reference payloads.
To replace those source runs, use the explicit input-path options below.
Do not overwrite the recorded evidence.

## Analysis and tests

The default analysis reproduces the published run selection.
It reads recorded `*_v1` runs and successful `native_TP4_v3`.
The input files remain immutable.

```bash
CUDA_VISIBLE_DEVICES='' $REF -m unittest discover \
  -s poc_recurrent_asymmetry/tests -v

CUDA_VISIBLE_DEVICES='' $REF -m poc_recurrent_asymmetry.scripts.analyze \
  --output poc_recurrent_asymmetry/metrics/analysis_repro1

CUDA_VISIBLE_DEVICES='' $REF -m poc_recurrent_asymmetry.scripts.present \
  --analysis poc_recurrent_asymmetry/metrics/analysis_repro1 \
  --output poc_recurrent_asymmetry/metrics/presentation_repro1

CUDA_VISIBLE_DEVICES='' $REF -m poc_recurrent_asymmetry.scripts.trace_figure \
  --output poc_recurrent_asymmetry/figures_repro1
```

`analysis_v1` is an earlier partial analysis before the TP1/TP2 controls finished.
`analysis_final_v1` is the final complete analysis.
The report uses only the latter for headline tables.

The original heavy capture used the following bounded command.
It is optional because the compact correlated events already reproduce the timeline figure.

```bash
$REF -m poc_recurrent_asymmetry.scripts.gpu_session \
  --gpus 4 --output outputs/nsys_repro1 --timeout 1800 -- \
  /usr/local/bin/nsys profile --trace=cuda,nvtx --sample=none --cpuctxsw=none \
  --capture-range=cudaProfilerApi --capture-range-end=stop \
  --output /home/esjung/MLLM-EP-recurrent-asymmetry/poc_recurrent_asymmetry/outputs/nsys_repro1/capture \
  $REF -m poc_recurrent_asymmetry.scripts.trace_hf
```

`scripts/parse_trace.py` identifies the original SQLite capture and preserves its compact export.
Its fixed input prevents accidental replacement of that frozen export.
An independent recapture needs a separate export directory and source-path change.
Do not replace the original capture or its hashes.

## Artifact map

| Artifact | Path |
|---|---|
| Frozen contract and dataset shapes | `metrics/frozen_contract.json`, `metrics/frozen_workload.json` |
| Exact reference and natural smoke outputs | `outputs/baseline_v1/` |
| Raw clean shape measurements | `outputs/profile_P*_v1/clean_B*.json` |
| Per-loop observer intervals | `outputs/profile_P*_v1/*_timeline.json` |
| Serial chunk measurements | `outputs/profile_P*_v1/chunk_*.json` |
| Full-cache chunk correctness | `outputs/chunk_correctness_v1/` |
| P2P, overlap, and exact CP operator | `outputs/physical_v1/physical/` |
| Actual four-rank NCCL | `outputs/physical_v1/nccl/` |
| Counter permission failure | `outputs/physical_v1/counter/` |
| Native clean controls and profiler events | `outputs/native_TP1_v1`, `native_TP2_v1`, `native_TP4_v3` |
| Preserved native startup failures | `outputs/native_TP4_v1`, `native_TP4_v2` |
| Controlled allocator diagnosis | `outputs/ipc_probe_v1/cases.json` |
| Compact actual Nsight events | `metrics/nsight/events.json.gz` |
| Heavy trace local pointers and hashes | `metrics/nsight/heavy_manifest.json` |
| Final CSV / JSON summaries | `metrics/analysis_final_v1/` |
| Plots and per-loop/component CSV | `metrics/presentation_v1/` |

Heavy `.nsys-rep` and `.sqlite` files remain local with SHA256 pointers.
The commit includes compact correlated events and summaries.
Small reference/parity arrays have a separate local-file hash manifest.
No checkpoint weights or GPU profiling database belong in Git.

Raw CSV files keep their original CRLF line endings.
Console logs and the source spec keep original whitespace.
The code/prose whitespace check excludes these frozen files.

## Burner restoration

The shared script can have a different default GPU set.
Always supply both masks explicitly.

```bash
CUDA_VISIBLE_DEVICES=4,5,6,7 VLLM_UTILIZE_GPUS=4,5,6,7 \
  bash /home/esjung/vllm-ep/run_utilize.sh
```

Do not execute that command while an experiment still owns those GPUs.
The recorded guarded runs already restore their assigned subsets.
The final read-only audit checks actual GPU contexts and their burner ancestry.

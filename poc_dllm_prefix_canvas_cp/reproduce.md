# Reproduction

## Scope and safety

The final verdict is NO-GO. No more GPU runs are required for this result.
The commands below document independent reproduction. They are not instructions
to resume experiments on a shared server without checking its current allocation.

Use physical GPUs 4, 5, 6, and 7 only. Check ownership before each run.
Do not terminate another user's process. Do not reuse a frozen output directory.
The guard refuses competing non-burner GPU processes.
It restores only the same-user, exact-mask burner policy that existed before the run.

The installed `run_utilize.sh` defaults to GPUs 2,3,6,7.
Do not run that default for this PoC. If burners need restoration without the guard,
set `VLLM_UTILIZE_GPUS=4,5,6,7` first. Do not create duplicate burners.

## Repository and dependencies

The experiment branch is `flashvep/dllm-prefix-canvas-cp-poc`.
Its parent is `d2cb1de1fc64d7cf7488754425c9675efde6c157`.
The local experiment worktree is `/home/esjung/MLLM-EP-prefix-canvas-cp`.
The original dirty checkout is separate and remains unchanged.

The environment paths below describe this server. No package installation ran
during the experiment. Pin versions before reproduction on another machine.

```bash
cd /home/esjung/MLLM-EP-prefix-canvas-cp
export CUDA_VISIBLE_DEVICES=4,5,6,7
poc_python=/home/esjung/.venvs/llada2-flash-sglang-053/bin/python
```

Required versions are PyTorch 2.8.0+cu128, SGLang 0.5.3.post1, transformers 4.57.0,
and Triton 3.4.0. The checkpoint revision is
`inclusionAI/LLaDA2.0-mini@dad945cac317da394b390f82c7b40691d8a881ed`.
The local checkpoint remains in the Hugging Face cache. Weights are not committed.

The existing dInfer path is `/home/esjung/external/dinfer-llada2-flash-poc`.
Its base revision is `1ffeb961cd258bede74fcf5ca8a416ae6d57b18f`.
The pre-existing diff appears in `patches/dinfer_pre_existing_native.diff`.
On this server, do not apply that patch again.
For another machine, create a separate checkout at the base revision.
Check the patch with `git apply --check` before applying it there.
Its expected hash appears in `results/environment.json`.
The final audit checks the four runtime source hashes too.

`runtime.py` reuses the previously checked distributed model loader.
It changes only setup constants for the native TP4/EP1 path.
It retains native weights, attention, cache, diffusion sampler, and block transitions.
It does not activate the older DeepEP/source-SP bridge.

## Frozen inputs

Use `results/frozen_tokens.json.gz` and `results/frozen_manifest.json` directly.
They contain eight calibration and 24 untouched held-out requests.
The source revisions, source hashes, token hashes, and selection rules are recorded.

`inspect_checkpoint_runtime.py` and `freeze_requests.py` created the original audit
and cohort. They refuse to overwrite the frozen primary records.
Do not regenerate the cohort to reproduce these numbers.
The original GSM8K resolution step selected a revision once, then recorded it.
Future reproduction uses that recorded revision, not current upstream main.

## Native sanity probe

Use a new output name. The guard creates its directory and records the command.

```bash
"$poc_python" -m poc_dllm_prefix_canvas_cp.scripts.run_guarded \
  --output results/reproduction/native_graph \
  --timeout 900 -- \
  "$poc_python" -m poc_dllm_prefix_canvas_cp.scripts.validate_native_multigpu \
  --output /home/esjung/MLLM-EP-prefix-canvas-cp/poc_dllm_prefix_canvas_cp/results/reproduction/native_graph \
  --backend tp4 --graphs
```

The probe checks three deterministic repeats and cross-rank output equality.
It records actual QKV and MoE tensor shapes on every rank.
The original eager and graph smoke files remain under `results/raw/`.
Generic CUDA-graph acceleration is not this PoC's gain.

## Stage A native sweep

The original `stage_a_native_v4` completed P512 through P8192.
Its P16,384 prefill graph failed. The matched fallback campaign
`stage_a_eager_prefill_v6` completed P512 and P16,384 with eager prefill.
Refinement used native graphs in both campaigns.

These commands reproduce the successful configurations without requesting the
known failing P16,384 prefill graph. Separate processes also reduce graph-pool accumulation.

```bash
"$poc_python" -m poc_dllm_prefix_canvas_cp.scripts.run_guarded \
  --output results/reproduction/native_mid \
  --timeout 1800 -- \
  "$poc_python" -m poc_dllm_prefix_canvas_cp.scripts.profile_native \
  --output /home/esjung/MLLM-EP-prefix-canvas-cp/poc_dllm_prefix_canvas_cp/results/reproduction/native_mid \
  --backend tp4 --limit 8 --prefixes 1024 2048 4096 8192

"$poc_python" -m poc_dllm_prefix_canvas_cp.scripts.run_guarded \
  --output results/reproduction/native_short_long \
  --timeout 1800 -- \
  "$poc_python" -m poc_dllm_prefix_canvas_cp.scripts.profile_native \
  --output /home/esjung/MLLM-EP-prefix-canvas-cp/poc_dllm_prefix_canvas_cp/results/reproduction/native_short_long \
  --backend tp4 --limit 8 --prefixes 512 16384 --eager-prefill --profiler
```

Each request has a 180-second watchdog. The outer guard also has a deadline.
Per length, the script runs three warmups, three clean repeats, and two coarse repeats.
The first frozen calibration request supplies separate trace and collective diagnostics.
The optional profiler runs only at the largest requested prefix.

The v6 experiment persisted every measurement but stalled during native process teardown.
Its interrupted guard status remains unchanged. Current code synchronizes GPU work,
executes a final barrier, records completion, then exits the worker subprocess directly.
This cleanup correction does not change inference. It was not GPU-rerun after the NO-GO.
Reproduction must check all four completion files and the parent guard status.

External graph events retain the last replay only. Analysis uses call-count
multiplication for earlier executions and labels that assumption explicitly.
Do not interpret the resulting attention fractions as directly timed whole-request shares.

## Stage B exact component test

This command recreates the final v3 order: three full traversals of 20 layers.
It tests P512, P8192, and P16,384. It does not execute task generation.
It uses native Q strides, three fresh canvas states, and real four-GPU collectives.

```bash
"$poc_python" -m poc_dllm_prefix_canvas_cp.scripts.run_guarded \
  --output results/reproduction/attention_component \
  --timeout 900 -- \
  "$poc_python" -m poc_dllm_prefix_canvas_cp.scripts.microbench_exact_split_attention \
  --output /home/esjung/MLLM-EP-prefix-canvas-cp/poc_dllm_prefix_canvas_cp/results/reproduction/attention_component
```

All three arms use optimized native attention kernels and CUDA graphs.
The distributed arm includes all per-call collectives, layout work, and softmax merge.
Actual prefix all-to-all preparation is measured separately.
Frozen BF16 tolerances are atol 0.03125, rtol 0.01, and relative L2 at most 0.01.
Masked prefill and cross-block refresh are not candidate acceleration claims.

## Offline analysis and tests

These commands use committed raw evidence. They do not allocate GPU workloads.
Analysis overwrites derived summaries and figures, never raw or frozen inputs.

```bash
CUDA_VISIBLE_DEVICES='' "$poc_python" -m poc_dllm_prefix_canvas_cp.scripts.analyze
CUDA_VISIBLE_DEVICES='' "$poc_python" -m poc_dllm_prefix_canvas_cp.scripts.finalize
CUDA_VISIBLE_DEVICES='' "$poc_python" -m unittest discover \
  -s poc_dllm_prefix_canvas_cp/tests -v
```

`analyze.py --input` can select newly recorded Stage A directories.
The committed default selects v4, then prefers completed v6 records for P512/P16,384.
`finalize.py` selects the final `stage_b_refinement_order_v3` component evidence.
For an independent campaign, change only its analysis input path in a new worktree.
Do not replace the committed raw evidence.

`scripts/final_audit.py` checks preserved local files and current burner ownership.
Its paths and original dirty-state hashes are server-specific.
It records, rather than modifies, GPU state. It must not run during a new experiment.

## Evidence map

| File | Scope |
|---|---|
| `results/baseline_per_request.csv` | Clean native phases, NFE/TPF, and request-level estimates |
| `results/baseline_per_step.csv` | Actual native forward events and iteration labels |
| `results/baseline_per_layer.csv` | Last-replay operator samples, explicitly not all replay histories |
| `results/critical_path_summary.json` | Request bootstrap and conditional Amdahl bounds |
| `results/KV_lifetime_summary.json` | Sampled prefix/canvas tensor checks |
| `results/profiler_classification.json` | Heavy kernel classes, not additive headline components |
| `results/microbench.csv` | Final synthetic physical component measurements |
| `results/correctness.json` | Native observer parity, not candidate full-generation parity |
| `results/raw_sha256.json` | Raw evidence inventory |
| `results/completion_audit.json` | Preservation, CPU tests, reference checks, and final burners |

Stage C and held-out task accuracy remain NOT_RUN. No speedup can be reconstructed
by treating conditional Amdahl bounds or isolated component ratios as measured E2E gains.

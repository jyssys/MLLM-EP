# Reproduction

## Fixed scope

Use physical GPUs 4, 5, 6, and 7 only.
Run commands from this repository's root.
Keep the original checkout and its staged changes untouched.
Use a new worktree or a new output directory for every GPU run.
The GPU guard refuses an existing target directory.
Never remove archived evidence to make a command run.

The experiment uses these existing interpreters without package upgrades:

```text
Native Ouro: /home/esjung/.venvs/cp-composition-vllm026/bin/python
Official HF reference: /home/esjung/.venvs/llada2-mini-reference/bin/python
Analysis and GPU guard: python with NumPy and Matplotlib
```

The [environment record](results/env_and_revisions.json) contains exact versions.
The [source pin](results/runtime_source_pin.json) contains implementation hashes.
Checkpoint revisions are constants in [contract.py](contract.py).
Pretrained weights remain in the external Hugging Face cache.
No model weights enter Git.

## Asset acquisition on a new machine

Use an isolated matching environment. Do not upgrade the existing dInfer environment.
The asset helper downloads pinned public metadata and weights outside Git:

```bash
/home/esjung/.venvs/cp-composition-vllm026/bin/python -m poc_looped_multigpu_kv_placement.scripts.bootstrap_assets --weights
```

The official Huginn plugin checkout is external:

```text
Repository: https://github.com/seal-rg/recurrent-pretraining
Revision: 1ea7220ec7eb42d13e89db0663df254d0bcdc28e
Path used: /home/esjung/external/looped-kv-huginn-official
```

The frozen prompt IDs already accompany the evidence.
Do not regenerate them for a matched rerun.
Their hash must match `token_artifact_sha256` in `results/frozen_workload.json`.
The freeze scripts document selection and source acquisition.
They refuse to replace an existing manifest.

## Native TP measurements

These examples use fresh `reproduce_*` directories.
The guard stops only verified, same-user burners on the selected devices.
It restores burners after success or failure.
It refuses an unidentified competing process.

```bash
python -m poc_looped_multigpu_kv_placement.scripts.gpu_session \
  --gpus 4 --timeout 2400 --output results/raw/reproduce_tp1 \
  -- /home/esjung/.venvs/cp-composition-vllm026/bin/python \
  -m poc_looped_multigpu_kv_placement.scripts.run_native_layout_sweep \
  --tp 1 --concurrency 1 2 4 8 \
  --output "$PWD/poc_looped_multigpu_kv_placement/results/raw/reproduce_tp1"

python -m poc_looped_multigpu_kv_placement.scripts.gpu_session \
  --gpus 4,5 --timeout 2400 --output results/raw/reproduce_tp2 \
  -- /home/esjung/.venvs/cp-composition-vllm026/bin/python \
  -m poc_looped_multigpu_kv_placement.scripts.run_native_layout_sweep \
  --tp 2 --concurrency 1 2 4 8 \
  --output "$PWD/poc_looped_multigpu_kv_placement/results/raw/reproduce_tp2"

python -m poc_looped_multigpu_kv_placement.scripts.gpu_session \
  --gpus 4,5,6,7 --timeout 2400 --output results/raw/reproduce_tp4 \
  -- /home/esjung/.venvs/cp-composition-vllm026/bin/python \
  -m poc_looped_multigpu_kv_placement.scripts.run_native_layout_sweep \
  --tp 4 --concurrency 1 2 4 8 --capacity-prefixes 1024 4096 8192 16384 32704 \
  --output "$PWD/poc_looped_multigpu_kv_placement/results/raw/reproduce_tp4"
```

TP1 and TP2 test B1 at all five lengths.
Their concurrent sweeps use the two endpoint lengths.
The archived TP4 campaign tests every offered concurrency at all five lengths.
The explicit `--capacity-prefixes` option reproduces that larger TP4 grid.
The current default limits concurrent sweeps to endpoint lengths.
This change does not change archived measurements.

Add `--smoke` in a separate fresh directory for the four-token reference probe.
Use `--profile-only --trace-prefix 16384` for the native kernel lane.
Add `--eager` only for the separate source diagnostic.
Use `--events-only --trace-prefix 16384 --limit 8` for the paired outer-event lane.
Use `--quality-only` for the three frozen natural-EOS sanity cases.
Do not interpret the eager diagnostic as the optimized native baseline.

## Independent native replica controls

Use all four assigned GPUs for each control.
The script creates independent official engines with round-robin request assignment.
It does not create a new placement method.

```bash
python -m poc_looped_multigpu_kv_placement.scripts.gpu_session \
  --gpus 4,5,6,7 --timeout 2400 --output results/raw/reproduce_tp1_dp4 \
  -- /home/esjung/.venvs/cp-composition-vllm026/bin/python \
  -m poc_looped_multigpu_kv_placement.scripts.run_native_replicas --tp 1 \
  --output "$PWD/poc_looped_multigpu_kv_placement/results/raw/reproduce_tp1_dp4"
```

Change the output suffix and use `--tp 2` for TP2 × DP2.
The native API rejects a single `LLM(data_parallel_size>1)` in this version.
Independent native engines avoid an unsupported API and preserve actual controller wall measurement.

## Huginn reference

```bash
python -m poc_looped_multigpu_kv_placement.scripts.gpu_session \
  --gpus 5 --timeout 1800 --output results/raw/reproduce_huginn \
  -- /home/esjung/.venvs/llada2-mini-reference/bin/python \
  -m poc_looped_multigpu_kv_placement.scripts.run_huginn_reference \
  --steps 16 --prefixes 1024 4032 \
  --output "$PWD/poc_looped_multigpu_kv_placement/results/raw/reproduce_huginn"
```

Use `--smoke --steps 32` in another directory for the H32 cache probe.
Use `run_huginn_quality` with the same guard and a fresh output directory for natural-EOS sanity.
Do not request Huginn TP2 or TP4. The native partition is unsupported.

## Rebuild tables and figures from archived evidence

These commands read archived raw evidence and rewrite only derived results and figures.
Their default selectors target the archived campaign names, not the `reproduce_*` examples.
New GPU runs retain their own directly readable JSON records.

```bash
python -m poc_looped_multigpu_kv_placement.scripts.summarize_results
python -m poc_looped_multigpu_kv_placement.scripts.analyze_evidence
python -m poc_looped_multigpu_kv_placement.scripts.placement_ledger
python -m poc_looped_multigpu_kv_placement.scripts.make_figures
python -m unittest discover -s poc_looped_multigpu_kv_placement/tests -v
```

`run_remaining_checks` executes missing archived diagnostic arms sequentially.
It preserves existing directories instead of retrying or replacing them.
The failed eager v1 log remains archived. The revised observer uses a separate v2 directory.

## GPU restoration

The launcher defaults to another GPU set on this server.
Always override its device list explicitly:

```bash
VLLM_UTILIZE_GPUS=4,5,6,7 CUDA_VISIBLE_DEVICES=4,5,6,7 \
  bash /home/esjung/vllm-ep/run_utilize.sh
```

Do not launch a second burner over an active model or burner.
The final restoration script validates the existing workers before replacement.
Its audit records the scoped launcher and actual GPU activity.

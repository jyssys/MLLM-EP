# Reproduction and resume

## Safety and scope

Use the isolated worktree `/home/esjung/MLLM-EP-ouro-deep-discovery`.
Do not reset or clean another worktree.
Do not change a shared environment or an external repository.
Use only physical GPUs 4/5/6/7.
Check process ownership before a GPU run.

The guard stops only identified owned utilization workers.
It refuses unknown GPU processes.
It restores a burner on each released device after its child process exits.
Use a new output name for every attempt.
An existing output directory must not receive another run.

## Existing environment

The campaign interpreter is `/home/esjung/.venvs/ouro-deep-discovery/bin/python`.
It installs no package into the shared binary environment.
The wrapper imports existing native vLLM or CDB dependencies read-only.
The exact dependency and model revisions are in `artifacts/environment_manifest.json`.

The guard uses `/home/esjung/.venvs/llada2-mini-reference/bin/python`.
This interpreter only manages processes and artifacts for these commands.
The native worker environment uses PyTorch 2.11.0+cu129 and vLLM 0.26.0.
The pinned CDB revision is `abdfae0adbdd12be8c6123ccb34eee98a38afb39`.

Native graph/custom-All-Reduce requires scoped `expandable_segments:False` in this environment.
The reduced red/green IPC evidence is under `outputs/custom_ar_*`.
The native scripts apply that setting before Torch imports.
Do not apply it globally to unrelated jobs.

## Preserve the original work

```bash
cd /home/esjung/MLLM-EP-ouro-deep-discovery
/home/esjung/.venvs/llada2-mini-reference/bin/python \
  -m poc_ouro_parallelism_deep_discovery.scripts.checkpoint verify
```

This command checks the saved predecessor HEAD and staged/unstaged hashes.
It also checks the specification snapshot.

## CPU validation

```bash
/home/esjung/.venvs/llada2-mini-reference/bin/python -m unittest \
  poc_ouro_parallelism_deep_discovery.scripts.verify_dag.Tests -v

/home/esjung/.venvs/llada2-mini-reference/bin/python -m compileall -q \
  poc_ouro_parallelism_deep_discovery/scripts
```

Additional CPU scripts retain their exact invocation in the companion artifacts.
These scripts test algebra or ownership. They do not measure model speed or quality.

## Optimized native TP2 baseline

```bash
/home/esjung/.venvs/llada2-mini-reference/bin/python \
  -m poc_ouro_parallelism_deep_discovery.scripts.gpu_session \
  --gpus 4,6 --output outputs/native_base26_R4_P2_repro1 --timeout 3600 -- \
  /home/esjung/.venvs/ouro-deep-discovery/bin/python \
  -m poc_ouro_parallelism_deep_discovery.scripts.native_baseline \
  --size 2.6B --tp 2 --R 4
```

The guard maps physical 4/6 to logical CUDA 0/1.
For TP4, use `--gpus 4,5,6,7` and `--tp 4`.
For TP1, select one eligible device and `--tp 1`.
The script asserts UUIDs, cache slots, and head ownership.
Its output work is teacher-forced. It is not an accuracy benchmark.

The optional `--fuse-allreduce-rms` flag tests an existing generic optimization.
Keep that run in a `control_` output directory.
Do not silently replace a default baseline row with this variant.
Check its numerical and quality effects separately.

## Physical component control

```bash
/home/esjung/.venvs/llada2-mini-reference/bin/python \
  -m poc_ouro_parallelism_deep_discovery.scripts.gpu_session \
  --gpus 4,6 --output outputs/components_base14_P2_repro1 --timeout 1800 -- \
  /home/esjung/.venvs/ouro-deep-discovery/bin/python \
  -m poc_ouro_parallelism_deep_discovery.scripts.launch_components \
  --standalone --nnodes 1 --nproc-per-node 2 \
  -m poc_ouro_parallelism_deep_discovery.scripts.component_probe \
  --size 1.4B --graphs \
  --capture /home/esjung/MLLM-EP-ouro-deep-discovery/poc_ouro_parallelism_deep_discovery/outputs/capture_base14_v2/1.4B/actual_mlp_inputs.pt
```

The captured tensor contains bounded activations, not model weights.
The run records its SHA256.
The remote path includes real GPU communication and a complete pretrained MLP.
The ready branch rows are supplied inputs, not uncomputed recurrent successors.

Replace `component_probe` with `communication_probe` to reproduce the collective fixture.
That fixture does not use model weights and needs no component arguments.
Replace it with `residency_probe` to compare aliases and intervening accesses.
The latter uses `--size` and `--capture`, but no `--graphs` flag.

## Baseline natural quality

```bash
/home/esjung/.venvs/llada2-mini-reference/bin/python \
  -m poc_ouro_parallelism_deep_discovery.scripts.gpu_session \
  --gpus 7 --output outputs/quality_base26_adaptive_repro1 --timeout 7200 -- \
  /home/esjung/.venvs/ouro-deep-discovery/bin/python \
  -m poc_ouro_parallelism_deep_discovery.scripts.quality_pilot \
  --size 2.6B --mode adaptive
```

The frozen calibration cohort contains eight requests per domain.
The held-out cohort must remain untouched until a qualifying candidate needs confirmation.
Quality generation permits EOS and uses the common 8192-token cap.
It does not force continuation.

For the native fixed baseline, use `native_quality_pilot --size 2.6B --tp 1`.
Do not combine native and staged results into one accuracy denominator.
The scorer separates runtime, model size, mode, TP degree, and domain.

## Analysis

```bash
/home/esjung/.venvs/llada2-mini-reference/bin/python \
  -m poc_ouro_parallelism_deep_discovery.scripts.analysis_and_plots \
  --label reproducible_analysis_01

/home/esjung/.venvs/llada2-mini-reference/bin/python \
  -m poc_ouro_parallelism_deep_discovery.scripts.native_parity_analysis \
  --output poc_ouro_parallelism_deep_discovery/artifacts/repro_parity_01.json

/home/esjung/.venvs/llada2-mini-reference/bin/python \
  -m poc_ouro_parallelism_deep_discovery.scripts.score_quality \
  --output poc_ouro_parallelism_deep_discovery/artifacts/repro_quality_01.json
```

Repeated quality reproductions need a separate analysis scope.
The scorer rejects duplicate request IDs within one runtime/model/mode/TP arm.
This prevents silent denominator inflation.
The historical analysis snapshot remains unchanged.

The source-index script also requires a new `--label`.
Its original observations preserve partial reads, failed identities, and missing code access.

## Resume

Read `research_state.json`, `reports/next_actions.md`, and the latest actual-time segment.
Inspect live owned jobs before another GPU command.
Update the state after each substantive work block.
Do not add concurrent worker minutes to coordinator elapsed hours.
Do not declare normal completion before the ten-hour contract is satisfied.

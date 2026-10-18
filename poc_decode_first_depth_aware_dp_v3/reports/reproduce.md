# Reproduction

## Source and safety

The branch is `flashvep/decode-first-depth-aware-dp-v3`.
The parent evidence commit is `e6a350e0a862d94251c5cefa0ef38038b017b8c4`.
The report identifies the selected checkpoint and CDB source SHA.
Use the pinned environment. Do not upgrade shared dependencies.

The commands below record the executed workflow.
Use an unused output path for a new physical run.
The GPU guard rejects an existing run directory.
Do not remove old evidence to rerun a command.
Keep the immutable manifests and all failed logs.

Run from `/home/esjung/MLLM-EP-decode-first-depth-dp-v3`.
Set the allowed physical mask:

```bash
export CUDA_VISIBLE_DEVICES=4,5,6,7
```

The native executable is `/home/esjung/.venvs/cp-composition-vllm026/bin/python`.
The analysis executable is `/home/esjung/.venvs/llada2-mini-reference/bin/python`.
Each `run_contract.json` records the exact physical command and timeout.
Its `source_snapshot.zip` preserves the runner source at launch.

## Metadata and source selection

These commands ran before the new discovery inference:

```bash
/home/esjung/.venvs/llada2-mini-reference/bin/python -m poc_decode_first_depth_aware_dp_v3.scripts.prepare
/home/esjung/.venvs/llada2-mini-reference/bin/python -m poc_decode_first_depth_aware_dp_v3.scripts.freeze_discovery
/home/esjung/.venvs/llada2-mini-reference/bin/python -m poc_decode_first_depth_aware_dp_v3.scripts.select_baseline
```

The source-selection file is immutable.
The incumbent already imports the CDB authors' exact source.
The Ragged-KV author repository was not verified. No imitation of that unpublished implementation was built.

## Quality and trace collection

This is one of the four quality partitions:

```bash
/home/esjung/.venvs/llada2-mini-reference/bin/python -m poc_decode_first_depth_aware_dp_v3.scripts.gpu_session \
  --gpus 5 --output outputs/quality8192_part1_v1 --timeout 10800 -- \
  /home/esjung/.venvs/cp-composition-vllm026/bin/python -m poc_decode_first_depth_aware_dp_v3.scripts.probe \
  --quality --cap 8192 --part 1 --parts 4
```

Parts 0/1/2/3 used physical GPUs 4/5/6/7 respectively.
The discovery command was:

```bash
/home/esjung/.venvs/llada2-mini-reference/bin/python -m poc_decode_first_depth_aware_dp_v3.scripts.gpu_session \
  --gpus 4 --output outputs/discovery128_graph_b1_v1 --timeout 7200 -- \
  /home/esjung/.venvs/cp-composition-vllm026/bin/python -m poc_decode_first_depth_aware_dp_v3.scripts.probe \
  --graphs --manifest poc_decode_first_depth_aware_dp_v3/artifacts/discovery_frozen.json --cap 256 --trace
```

`observer_runs.py` ran graph-clean, graph-trace, and eager-clean checks on eight frozen calibration requests.
`runtime_environment.py` records the reused Python and library versions.

## Physical DP4

The trace and clean plans contain every case, arrival process, request ID, and causal feature.
The controller uses four native workers. Each worker receives one physical GPU.
The local recurrence and KV code remain upstream code.

```bash
/home/esjung/.venvs/llada2-mini-reference/bin/python -m poc_decode_first_depth_aware_dp_v3.scripts.gpu_session \
  --gpus 4,5,6,7 --output outputs/live_trace_v1 --timeout 10800 -- \
  /home/esjung/.venvs/llada2-mini-reference/bin/python -m poc_decode_first_depth_aware_dp_v3.scripts.live_dp \
  --plan poc_decode_first_depth_aware_dp_v3/artifacts/live_trace_plan.json --trace
```

The clean run used `outputs/live_clean_v1` and `artifacts/live_clean_plan.json`, without `--trace`.
`cost_surface.py` ran the bounded synthetic-context diagnostic on physical GPU7.
`after_cost_ablation.py` ran the fixed/refill/no-refill controls.
`batched_kv_probe.py` ran the corrected heavy correctness probe.
The failed v1 counting assertion remains in its original log.

## Offline analysis

Large rank traces have lossless `.json.gz` publication copies.
The new JSON reader accepts these copies when the uncompressed file is absent.
The archive manifest records original SHA256 hashes.

```bash
/home/esjung/.venvs/llada2-mini-reference/bin/python -m poc_decode_first_depth_aware_dp_v3.scripts.analyze_quality
/home/esjung/.venvs/llada2-mini-reference/bin/python -m poc_decode_first_depth_aware_dp_v3.scripts.analyze_discovery
/home/esjung/.venvs/llada2-mini-reference/bin/python -m poc_decode_first_depth_aware_dp_v3.scripts.validate_transitions
/home/esjung/.venvs/llada2-mini-reference/bin/python -m poc_decode_first_depth_aware_dp_v3.scripts.analyze_live \
  --trace-run poc_decode_first_depth_aware_dp_v3/outputs/live_trace_v1 \
  --clean-run poc_decode_first_depth_aware_dp_v3/outputs/live_clean_v1
/home/esjung/.venvs/llada2-mini-reference/bin/python -m poc_decode_first_depth_aware_dp_v3.scripts.oracle
/home/esjung/.venvs/llada2-mini-reference/bin/python -m poc_decode_first_depth_aware_dp_v3.scripts.refine_oracle
/home/esjung/.venvs/llada2-mini-reference/bin/python -m poc_decode_first_depth_aware_dp_v3.scripts.causal_predictor
```

These analyses write versioned v3 artifacts. Preserve their earlier outputs before a new analysis revision.
The predictor runs only after the frozen oracle gate passes.
It uses supervised ridge models, not RL.

## Live hindsight check

`confirm_hindsight.py` freezes the oracle assignments before the physical check.
It runs the live gate. It does not replay synthetic exits.

```bash
/home/esjung/.venvs/llada2-mini-reference/bin/python -m poc_decode_first_depth_aware_dp_v3.scripts.confirm_hindsight
/home/esjung/.venvs/llada2-mini-reference/bin/python -m poc_decode_first_depth_aware_dp_v3.scripts.confirm_hindsight --concurrency 64 --tag c64
/home/esjung/.venvs/llada2-mini-reference/bin/python -m poc_decode_first_depth_aware_dp_v3.scripts.analyze_confirmation
/home/esjung/.venvs/llada2-mini-reference/bin/python -m poc_decode_first_depth_aware_dp_v3.scripts.secondary_analysis
/home/esjung/.venvs/llada2-mini-reference/bin/python -m poc_decode_first_depth_aware_dp_v3.scripts.report_results
/home/esjung/.venvs/llada2-mini-reference/bin/python -m poc_decode_first_depth_aware_dp_v3.scripts.plots
/home/esjung/.venvs/llada2-mini-reference/bin/python -m unittest poc_decode_first_depth_aware_dp_v3.tests.test_contract
```

## Burner restoration

The guard restores recognized owned burners on every exit path.
It does not signal unidentified users' processes.
The shared script's default GPU list is not trusted.
If a permitted GPU is idle after all experiments, use both explicit masks:

```bash
CUDA_VISIBLE_DEVICES=4,5,6,7 VLLM_UTILIZE_GPUS=4,5,6,7 bash /home/esjung/vllm-ep/run_utilize.sh
```

Do not launch a second burner when the scoped guard already restored one.
The final audit checks the four permitted GPUs and burner ownership.

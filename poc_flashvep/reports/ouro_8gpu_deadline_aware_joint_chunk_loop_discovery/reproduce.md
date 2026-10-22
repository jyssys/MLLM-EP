# Reproduction and resume

This campaign does not restart native-vLLM PP.
Every worker sees one authorized GPU.
The dispatcher checks physical UUIDs and preserves completed jobs.
Failed attempts require a new attempt ID.

## Environment

Repository: `/home/esjung/MLLM-EP-github`.
Python: `/home/esjung/.venvs/ouro-chunk-loop-prefill/bin/python`.
The source locks record exact package versions and checkpoint hashes.
The model is Base Ouro-1.4B at revision `574fa66cb8bf5abdc979642d01cf2b79b16bfab1`.
The source pin is RLT `4a350344027960a48384626aa646f1765fda6d4f`.

The original calibration checkout is `/home/esjung/external/ouro-deadline-adaptive-rlt`.
The G3 candidate checkout is `/home/esjung/external/ouro-deadline-g3-candidate-rlt`.
The candidate adds a homogeneous loop-group boundary check.
`qualified_runtime_candidate.json` preserves its complete diff and hash.
The controller selects it only after Phase 0 finishes.
`active_runtime.json` then records the selected path.
The previous research checkouts remain unchanged.

## Commands

Run these commands from the repository root.
Do not start a second dispatcher on occupied experiment devices.

```bash
/home/esjung/anaconda3/bin/python -m poc_flashvep.ouro_adaptive_chunk_loop.resources status
/home/esjung/anaconda3/bin/python -m poc_flashvep.ouro_adaptive_chunk_loop.checkpoint
```

The original calibration command is:

```bash
/home/esjung/anaconda3/bin/python -m poc_flashvep.ouro_adaptive_chunk_loop.dispatch --manifest poc_flashvep/reports/ouro_8gpu_deadline_aware_joint_chunk_loop_discovery/p0_contention_plan.json --manage-burners
```

The controller waits for that dispatcher to finish.
It then checks the G3 source hash and executes the registered Phase 1 diagnostic queue.
It stops after those tests for root-cause review.
It does not implement Adaptive Joint automatically.

```bash
/home/esjung/.venvs/ouro-chunk-loop-prefill/bin/python -u -m poc_flashvep.ouro_adaptive_chunk_loop.controller --after-p0 --stop-after-p1
```

The controller state is `state/controller.json`.
Each GPU also has a separate `state/gpuN.json`.
`run_manifest.jsonl` records job start/end events.
`raw/JOB_ID/completed.json` records result hashes.
The dispatcher checks those hashes before it skips a completed job.

The historical failure command remains red:

```bash
/home/esjung/anaconda3/bin/python -m poc_flashvep.ouro_adaptive_chunk_loop.legacy_feedback --assert-exact
```

That command checks frozen outputs, not a new GPU run.
The G3 regression command preserves red and green CPU lanes:

```bash
/home/esjung/.venvs/ouro-chunk-loop-prefill/bin/python -m poc_flashvep.ouro_adaptive_chunk_loop.qualify_candidate
```

## Subsequent gates

Review first-divergence tensors before dispatching affected new scheduling paths.
`p1_controlled_plan.json` adds common-prefix numerical controls.
Those forced-prefix diagnostics are not quality or performance results.
`p1_quality_plan.json` contains the full-answer 114-case pilot per policy.
`p2_surface_plan.json` contains the registered cost grid.
Neither plan is evidence that those jobs executed.

The canonical fit freezes validation selection before blind test metrics.
It refuses to overwrite an existing blind result.
Use a new registered output root for a new fit.
Do not delete frozen results to bypass this protection.

```bash
/home/esjung/.venvs/ouro-chunk-loop-prefill/bin/python -m poc_flashvep.ouro_adaptive_chunk_loop.fit_models
```

The earlier `predictor` and `decode_prediction` entry points are development tools.
They do not implement the complete frozen selection protocol.
Do not use them to replace the published blind result.

Completed analyses can run without a GPU:

```bash
/home/esjung/.venvs/ouro-chunk-loop-prefill/bin/python -m poc_flashvep.ouro_adaptive_chunk_loop.surface_analysis
/home/esjung/.venvs/ouro-chunk-loop-prefill/bin/python -m poc_flashvep.ouro_adaptive_chunk_loop.prediction_exports
/home/esjung/.venvs/ouro-chunk-loop-prefill/bin/python -m poc_flashvep.ouro_adaptive_chunk_loop.headroom
/home/esjung/.venvs/ouro-chunk-loop-prefill/bin/python -m poc_flashvep.ouro_adaptive_chunk_loop.summarize_controls
```

The main completed dispatch is `quality_and_surface_dispatch_v1.json`.
It contains eight quality jobs and 96 main cost jobs.
The dispatcher checks immutable result hashes and skips completed jobs.
Original failed observer jobs remain failed.
The v2 retry plan contains only their reviewed replacement attempts.

For the local test suite, set the runtime import path explicitly:

```bash
PYTHONPATH=/home/esjung/external/ouro-deadline-g3-candidate-rlt /home/esjung/.venvs/ouro-chunk-loop-prefill/bin/python -m unittest discover -s poc_flashvep/ouro_adaptive_chunk_loop -t . -p 'test_*.py'
```

`raw_archive_manifest.json` lists portable raw archives and every member SHA256.
Restore archive members relative to this report root, not the repository root.
The archive writer checks each decompressed member before publication.
`local_large_artifacts.json` lists captures that remain local.
Model weights and compilation caches are not Git artifacts.
`portable_mlp_witness.pt` contains only three 2,048-element BF16 vectors.
It is not a model checkpoint.
The operator replay can use it when the large local captures are absent.

## Burners

Use the scoped resource helper.
It passes an explicit GPU list to `run_utilize.sh`.
The shell script's default mask still names 2,3,6,7.
Do not rely on that default for this campaign.

```bash
/home/esjung/anaconda3/bin/python -m poc_flashvep.ouro_adaptive_chunk_loop.resources restore --gpus 0,1,2,3,4,5,6,7
```

The helper restores only empty devices.
It refuses to stop unrelated jobs.
Quiet interference controls reserve idle devices as measurement resources.
Burners remain stopped during those controls.

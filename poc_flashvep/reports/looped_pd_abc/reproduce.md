# Reproduction and resume

This experiment uses the current repository and a separate patched RLT clone.
Do not apply its patch to a shared runtime with active jobs.
Do not delete existing raw data to repeat a run.
Use a new run name for a new measurement.

## Pinned inputs

| Component | Value |
|---|---|
| Repository | `/home/esjung/MLLM-EP-github` |
| Experiment branch | `flashvep/looped-pd-abc-disaggregation-poc` |
| Parent commit | `b12fedb68a0861034f3ca684358acccfd5f84b0e` |
| Private RLT clone | `/home/esjung/external/looped-pd-abc-rlt` |
| Upstream RLT commit | `6b164d3844d35ab8e753df1864f0c87dc180401a` |
| Main Python | `/home/esjung/.venvs/looped-pd-abc/bin/python` |
| HF reference Python | `/home/esjung/.venvs/looped-pd-abc-hf457/bin/python` |
| Main Transformers | 5.16.1 |
| Official HF reference Transformers | 4.57.6 |
| Torch / CUDA | 2.11.0+cu129 / 12.9 |
| vLLM | 0.26.0+cu129 |
| Attention | FlashAttention 4, package 4.0.0b30 |
| NIXL | 1.4.1, CU12 wheel, UCX backend |
| Triton | 3.6.0 |
| NumPy | 2.4.6 |

`configs/models.json` records the checkpoint revisions and configuration files.
`common.py:MODEL_PINS` records the same immutable revisions.
`configs/hardware.json` records GPU UUIDs and topology.
`commands.jsonl` records executed commands, GPU masks, PIDs, and launch times.

The main environment reads validated package paths through its own `.pth` file.
It does not change packages in the previous environments.
This local reuse is not a portable environment lock.
A separate host must install the pinned packages and qualify its backend again.
The NIXL metapackage declares a CU13 dependency that this CU12 experiment did not install.
Do not interpret the environment inventory as a successful `pip check`.

## Runtime changes

The report preserves a current RLT patch in `sources/rlt_changes.patch`.
Each worker also saves an independent source ZIP before inference.
The ZIP contains this PoC's Python files and that attempt's RLT patch.
`configuration.json` and each request row contain the attempt ID.
The latest-attempt analysis never merges outputs from an earlier failed attempt.
Original append-only outputs and failure logs remain available.

Dory loads the official BF16 checkpoint with a minimal RLT adapter.
Its reference plugin uses the unchanged official-card-linked vLLM source.
The exact reference source is retained in `sources/dory_vllm_876edc.py`.
The plugin compatibility wrapper is `dory_plugin.py`.
The adapter is `dory_adapter.py`.
No other checkpoint substitutes for Dory.

## Safe resource checks

Run these commands from the repository root.

```bash
nvidia-smi -L
nvidia-smi topo -m
/home/esjung/.venvs/looped-pd-abc/bin/python -m poc_flashvep.looped_pd_abc.resources status
```

The resource helper checks UID, process session, and approved burner logs.
It does not terminate an unrelated GPU process.
Quiet measurements stop owned burners on all eight GPUs.
The contention experiment justifies that isolation policy.
The helper restores idle owned GPUs after a dispatcher releases its pool.

## CPU checks

```bash
CUDA_VISIBLE_DEVICES='' PYTHONPATH=/home/esjung/external/looped-pd-abc-rlt:/home/esjung/MLLM-EP-github /home/esjung/.venvs/looped-pd-abc/bin/python -m pytest -q poc_flashvep/looped_pd_abc/test_contracts.py /home/esjung/external/looped-pd-abc-rlt/tests/test_pd.py /home/esjung/external/looped-pd-abc-rlt/tests/test_tensor_transfer.py
```

That earlier subset passed 54 CPU tests and skipped 77 GPU-marked cases.
The extended corrected suite passed 97 CPU tests and deselected 86 GPU cases.
It included upstream StageEngine, PD, and tensor-transfer tests.
Its 12 warnings concern unregistered GPU test marks.
The PoC-specific suite separately passed 11 tests.
These tests do not replace actual model qualification.
The original failed extended-suite log remains available.
The corrected log is `logs/frozen_source_cpu_tests_corrected.log`.

The extended command is:

```bash
CUDA_VISIBLE_DEVICES='' PYTHONPATH=/home/esjung/external/looped-pd-abc-rlt:/home/esjung/MLLM-EP-github /home/esjung/.venvs/looped-pd-abc/bin/python -m pytest -q -m 'not gpu' poc_flashvep/looped_pd_abc/test_contracts.py /home/esjung/external/looped-pd-abc-rlt/tests/test_pd.py /home/esjung/external/looped-pd-abc-rlt/tests/test_tensor_transfer.py /home/esjung/external/looped-pd-abc-rlt/tests/test_stage_engine.py
```

New accounting tests use `poc_flashvep/looped_pd_abc/test_analysis.py`.
Their result is separate from the earlier frozen-runtime qualification.

## Small actual PD run

```bash
CUDA_VISIBLE_DEVICES=0,1 OMP_NUM_THREADS=1 ABC_HUGINN_REFERENCE=1 PYTHONPATH=/home/esjung/external/looped-pd-abc-rlt:/home/esjung/MLLM-EP-github /home/esjung/.venvs/looped-pd-abc/bin/python -m poc_flashvep.looped_pd_abc.worker --model ouro --mode pd --gpus 0,1 --name ouro_manual_new_run --requests 4 --length 128 --cap 32 --concurrency 1 --warmup --graphs --asynchronous --chunk 256
```

Stop verified owned burners on that pair before a manual run.
Do not launch this command over an existing experiment.
For clean latency, also reserve a quiet interval on the rest of the node.

Add A with `--wavefront --enable-recurrence-split --split-cut 2` and a chunk large enough for the complete prompt.
Add B instead with `--wavefront --enable-frontier-handoff`.
Use neither A nor B for standalone C.
C uses `--placement loop` and multiple same-role workers.
The matched C ablation uses `--placement service`.
Huginn and Dory require `--enable-boundary-adapter` for ordinary PD.
Dory additionally requires `ABC_RLT_DORY=1` in the process environment.
The dispatcher supplies these environments automatically.

## Resume the actual campaign

First check `state/main_campaign.json`, `state/dispatcher.json`, and live PIDs.
Do not start a second controller while the recorded controller remains active.
The dispatcher uses a file lock and skips runs whose status is `COMPLETE`.

```bash
/home/esjung/.venvs/looped-pd-abc/bin/python -m poc_flashvep.looped_pd_abc.run_benchmarks
```

The campaign performs byte-identity checks, contention calibration, and independent A/B/C comparisons.
Its final state requests human review. It does not claim success automatically.
The first launched controller predates the explicit gate-summary update.
Its six byte-identity runs passed before it continued.
Later resumes also write independent model/track gate summaries.

## Analyze existing results without GPU work

```bash
/home/esjung/.venvs/looped-pd-abc/bin/python -m poc_flashvep.looped_pd_abc.analyze
/home/esjung/.venvs/looped-pd-abc/bin/python -m poc_flashvep.looped_pd_abc.compare --comparisons
```

`results_raw.csv` contains only completed runs.
`paired_comparisons.csv` includes output agreement and the number of independent starts.
Its bootstrap resamples starts and paired request IDs.
An interval from fewer than three starts cannot establish an accepted gain.

`e2e_phase_breakdown.csv` partitions coordinator wall time into non-overlapping host phases.
It is not an attention/MLP GPU-time decomposition.
`transfer_completion_observation_seconds` can overlap with computation and other transfers.
Do not add that counter to request latency.

## Fidelity and quality limits

The natural parity pilot has a 32-token cap and permits EOS.
It is not forced-token replay. It is also not complete-answer accuracy evaluation.
The pilot source contains GSM8K training-validation questions, not the test set.
The harness uses each question only. It does not use the source's few-shot raw prompt or answer field.

Official-reference output agreement is incomplete for all three models.
The reports distinguish that limitation from exact internal RLT handoff parity.
No GSM8K test score or HumanEval score is established by this campaign.
HumanEval execution requires a separately qualified code sandbox.

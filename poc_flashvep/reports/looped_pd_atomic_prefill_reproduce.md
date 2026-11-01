# Atomic Prefill PD: Reproduction and Resume

Status: the GPU search is still in progress.
This file lists the actual implementation and executed entry points.
It does not imply that every planned configuration has run.

## Environment

```bash
cd /home/esjung/MLLM-EP-github
PYTHON=/home/esjung/.venvs/looped-pd-abc/bin/python
```

The branch is `flashvep/looped-pd-atomic-prefill-qps-crossover`.
The first implementation checkpoint is `fa344722b4adaebc6d058875325047289825dec3`.
Its remote SHA was verified.
The final commit is not available until closeout.

The native RLT source remains `/home/esjung/external/looped-pd-abc-rlt`.
Its exact HEAD and working-diff hash are in the runtime audit.
Each executed run has a source archive and configuration SHA256.
Use those records to distinguish a measured version from later CPU analysis.

## Executed preparation and tests

```bash
$PYTHON -m poc_flashvep.looped_pd_atomic_prefill.bootstrap
$PYTHON -m poc_flashvep.looped_pd_atomic_prefill.prepare_data
$PYTHON -m poc_flashvep.looped_pd_atomic_prefill.configs initial
$PYTHON -m unittest poc_flashvep.looped_pd_atomic_prefill.test_contracts
```

The original ShareGPT-only preparation failed before the data freeze.
Its failure record remains preserved.
The frozen pool includes pinned public LongBench documents.
Do not replace its Prompt IDs or silently resample the calibration/test split.

## Run one immutable namespace

```bash
$PYTHON -m poc_flashvep.looped_pd_atomic_prefill.dispatch qual_ouro_full_l8192_v1.json
```

This example already exists.
The dispatcher skips a completed namespace.
It refuses to overwrite a failed or incomplete namespace.
Use a new name for an explicitly justified retry.
Do not run two dispatchers on the same GPU pair.

The dispatcher verifies GPU ownership and the preservation contract.
Qualification jobs can share the host only on disjoint GPU pairs.
They are not clean timing runs.
Service, calibration, and final measurements take an exclusive quiet-lane lease.
Do not bypass that lease to increase utilization.

## Resume the automatic search

First inspect `state/controller.json`, `budget.json`, and actual process command lines.
Do not launch a second controller while one is alive.
If no controller or owned GPU dispatcher remains, use:

```bash
$PYTHON -m poc_flashvep.looped_pd_atomic_prefill.campaign
```

The controller resumes `configs/study_plan.json`.
It skips terminal namespaces instead of repeating them.
It reloads CPU stage generation at each boundary.
The progress guard can restart only a stopped controller with no incomplete namespace.
It does not repair code, reinterpret a failed equality test, or overwrite a failure.

The new start is 2026-11-01 04:03:08 UTC.
The unattended operational checkpoint is 24 hours after that start.
The previous study's expired 12-hour budget is not used.

## Actual raw evidence

The evidence root is:

`poc_flashvep/reports/looped_pd_atomic_prefill_qps_crossover/`

| Path | Content |
|---|---|
| `configs/` | Frozen run configurations, Poisson offsets, and selected model settings |
| `data/` | Exact token IDs, source text/provenance, and disjoint calibration/test pools |
| `raw/<run>/` | Per-request output IDs, times, state checks, counters, source archive, and terminal status |
| `logs/<run>.log` | Complete stdout/stderr, including failures |
| `commands.jsonl` | Executed commands, GPU mapping, commit, and archive/config hashes |
| `resource_ledger.jsonl` | Verified burner stop/restore operations |
| `state/` | Correctness gates, controller history, and explicit blockers |
| `figures/` | Measured plots and their provenance manifest |

An existing configuration file is not proof of an executed experiment.
Use `dispatcher_result.json`, worker `status.json`, and actual request evidence together.

## CPU analysis entry points

These commands read actual evidence.
They do not launch model inference.

```bash
$PYTHON -m poc_flashvep.looped_pd_atomic_prefill.analyze
$PYTHON -m poc_flashvep.looped_pd_atomic_prefill.matrix
$PYTHON -m poc_flashvep.looped_pd_atomic_prefill.profiles
$PYTHON -m poc_flashvep.looped_pd_atomic_prefill.classify
$PYTHON -m poc_flashvep.looped_pd_atomic_prefill.uncertainty
$PYTHON -m poc_flashvep.looped_pd_atomic_prefill.figures
```

The first Nsight compatibility run captures actual CUDA activity on GPU6/7.
An empty arrival span in its all-ready smoke uses the full measured-cohort window.
Final open-loop profiles use the central 20–90% arrival window.
Nsight global PIDs contain a host tag.
The parser checks PID bits 24–47 against actual measured worker PIDs.

Graph-level traces omit graph-internal kernels.
They must not be reported as complete D kernel-busy evidence.
SM occupancy is NOT_MEASURED under the current profiling permission.

## Publication

```bash
$PYTHON -m poc_flashvep.looped_pd_atomic_prefill.publish --push
```

This command makes an implementation checkpoint, not a final-result claim.
Final closeout uses `--final --push` only after evidence review.
Publication uses a temporary Git index and only this task's paths.
It checks that unrelated index entries remain unchanged.
It never force-pushes.
Files above GitHub's size limit require lossless archive handling before publication.

At closeout, restore only verified approved burners on genuinely unused devices.
Explicitly pass GPU0–7 to the approved resource helper.
The burner shell script's current default subset is not all eight GPUs.

## Automatic CPU closeout

A separate CPU monitor waits for the GPU search to reach its terminal state.
It does not submit, stop, or modify GPU jobs.
It checks that no managed GPU job or incomplete namespace remains.
It then compiles the actual evidence, tests the code, and restores approved idle burners.
It commits only owned paths and checks the remote SHA after the normal push.
An error produces `state/closeout_status.json` with `BLOCKED_CLOSEOUT`.
Such a record is not successful completion.

```bash
$PYTHON -m poc_flashvep.looped_pd_atomic_prefill.closeout --wait --push
```

Do not launch a second monitor while its CPU lease is held.
The final report compiler refuses to run while GPU search is active.

# Atomic Prefill PD: Final Reproduction Commands

Compiled at 2026-11-02T01:28:43.086700+00:00. Elapsed time: 21.426 hours.

Branch: `flashvep/looped-pd-atomic-prefill-qps-crossover`.

Compilation source commit: `147c4dc48215dd82cc85f77b044ff27325ea84fc`.

Each measured run records its own source archive, source commit, and configuration hash.
The final publication commit can differ from the measurement commit.
Use `git log -1 --format=%H` on this branch to obtain the publication SHA.

Actual analysis-tree SHA256: `a392ed79e603c437d00c3575d39166f104f681e2144fe757f6ef0524e7bc5c18`.

The compilation HEAD can precede uncommitted analysis changes.
See `state/analysis_source_manifest.json` for the exact file hashes.

```bash
cd /home/esjung/MLLM-EP-github
PYTHON=/home/esjung/.venvs/looped-pd-abc/bin/python
$PYTHON -m poc_flashvep.looped_pd_atomic_prefill.bootstrap
$PYTHON -m unittest poc_flashvep.looped_pd_atomic_prefill.test_contracts
```

Executed commands, worker PIDs, source commits, archive hashes, and GPU IDs are in `commands.jsonl`.
Each raw namespace includes its exact immutable configuration and source ZIP.

## Resume without duplicate execution

Inspect `budget.json`, `state/controller.json`, and actual process command lines first.
Never launch another controller while an existing controller owns the queue.
The dispatcher skips completed namespaces and refuses to overwrite incomplete or failed namespaces.
```bash
$PYTHON -m poc_flashvep.looped_pd_atomic_prefill.campaign
```

A justified retry needs a new namespace.
Do not replace Prompt IDs, calibration thresholds, or arrival offsets in an old configuration.

## CPU-only analysis

```bash
$PYTHON -m poc_flashvep.looped_pd_atomic_prefill.analyze
$PYTHON -m poc_flashvep.looped_pd_atomic_prefill.matrix
$PYTHON -m poc_flashvep.looped_pd_atomic_prefill.profiles
$PYTHON -m poc_flashvep.looped_pd_atomic_prefill.native_capacity_controls
$PYTHON -m poc_flashvep.looped_pd_atomic_prefill.margin
$PYTHON -m poc_flashvep.looped_pd_atomic_prefill.classify
$PYTHON -m poc_flashvep.looped_pd_atomic_prefill.uncertainty
$PYTHON -m poc_flashvep.looped_pd_atomic_prefill.sensitivity
$PYTHON -m poc_flashvep.looped_pd_atomic_prefill.figures
$PYTHON -m poc_flashvep.looped_pd_atomic_prefill.reservation_accounting
$PYTHON -m poc_flashvep.looped_pd_atomic_prefill.architecture_accounting
$PYTHON -m poc_flashvep.looped_pd_atomic_prefill.reporting
```

CPU analysis cannot turn a configuration into an executed GPU result.
Profile overhead must pass its matched control before activity transfers to clean timing.

## Large evidence and publication

Original large files remain local.
Lossless gzip parts and `archives/archive_manifest.json` preserve them for GitHub.
Verify the manifest SHA256 after restoration.
```bash
$PYTHON -m poc_flashvep.looped_pd_atomic_prefill.archive
$PYTHON -m poc_flashvep.looped_pd_atomic_prefill.publish --final --push
git rev-parse HEAD
git ls-remote origin refs/heads/flashvep/looped-pd-atomic-prefill-qps-crossover
```

Publication uses a temporary index and owned paths only.
It checks the unrelated staged-entry contract before and after the commit.
It never force-pushes.

## Resource restoration

The approved helper explicitly selects physical GPUs 0–7.
It starts burners only on unused devices after all GPU measurements end.
It does not signal an unrelated process.
```bash
$PYTHON -m poc_flashvep.looped_pd_atomic_prefill.resources status
```

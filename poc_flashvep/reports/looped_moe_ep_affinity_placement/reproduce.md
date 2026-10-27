# Reproduction and resume

## Scope

Use the three original pinned checkpoints in `model_architecture_and_sources.md`.
Use physical GPUs 0–7 only.
One CUDA device belongs to each rank process.
The launcher records the physical device, rank, command, and source hashes.
The environment record includes UUIDs, NVLink topology, driver, and source revisions.

The worker interpreter is `/home/esjung/.venvs/looped-moe-headroom-py312/bin/python`.
Foil benchmark and packed-data jobs use `/home/esjung/.venvs/looped-moe-affinity-eval-py312/bin/python`.
The isolated FA3 import directory is `/home/esjung/looped-moe-affinity-deps/site`.
`scripts/build_fa3.sh` records the separate FA3 build recipe.
Do not replace the existing environment to reproduce that binary.

## Existing active campaigns

Inspect these state files before any GPU launch:

1. `research_state.json` records the original EP and official-quality campaign.
2. `post_state.json` records matched native, independent-process, and generic metadata controls.
3. `attribution_campaign_state.json` records complete-rollout percentage measurements.
4. `finalization_state.json` records the local analysis and release watcher.
5. `release_state.json` records a real commit and verified remote SHA, if publication succeeds.

Also inspect exact process command lines and GPU ownership.
Do not duplicate a dispatcher that still owns devices.
The resume sequence preserves completed attempts and their original identifiers.
An incomplete attempt requires diagnosis and a new identifier.
Do not remove its raw files to force a retry.

The normal durable GPU sequence is:

```bash
cd /home/esjung/MLLM-EP-github
python -u -m poc_flashvep.looped_moe_affinity.full_campaign
python -u -m poc_flashvep.looped_moe_affinity.post_campaign
python -u -m poc_flashvep.looped_moe_affinity.attribution_campaign
```

The latter two campaigns can wait as independent background processes.
They launch no GPU work until the preceding campaign reaches its required state.
The final analysis watcher waits for the attribution campaign:

```bash
python -u -m poc_flashvep.looped_moe_affinity.finalize --wait --publish
```

Do not execute this sequence again while the existing processes remain active.
The current commands and logs appear in `runs/commands.jsonl` and `next_actions.md`.

## CPU analysis without a GPU rerun

```bash
cd /home/esjung/MLLM-EP-github
python -m poc_flashvep.looped_moe_affinity.aggregate_quality
python -m poc_flashvep.looped_moe_affinity.aggregate_timing
python -m poc_flashvep.looped_moe_affinity.aggregate_generic
python -m poc_flashvep.looped_moe_affinity.compare_baselines
python -m poc_flashvep.looped_moe_affinity.paired_quality
python -m poc_flashvep.looped_moe_affinity.oracle_summary
python -m poc_flashvep.looped_moe_affinity.aggregate_attribution
python -m poc_flashvep.looped_moe_affinity.reporting --preview
python -m unittest poc_flashvep.looped_moe_affinity.test_cpu -v
```

The preview explicitly identifies incomplete evidence.
Do not remove `--preview` to manufacture a completed result.
The finalizer checks completed GPU campaigns and mandatory result counts first.

The route analyses use these commands after their input corpora complete:

```bash
python -m poc_flashvep.looped_moe_affinity.ep_affinity
python -m poc_flashvep.looped_moe_affinity.affinity --decode
python -m poc_flashvep.looped_moe_affinity.count_prediction
python -m poc_flashvep.looped_moe_affinity.kernel_aggregate
python -m poc_flashvep.looped_moe_affinity.verify_fidelity
python -m poc_flashvep.looped_moe_affinity.figures
```

The original prefill analysis also uses `affinity`, `surrogate_controls`, `prediction_controls`, and `placement`.
Its derived arrays already exist.
Do not rerun that pipeline unnecessarily during clean GPU measurements.

## Exact generic metadata ablation

The generic control uses current router IDs already replicated on every rank.
It calculates the source/expert count matrix without the redundant count All-Gather.
It does not predict routing, move weights, change expert work, or use loop history.
Its diagnostic compares each count matrix against a real collective result.
The clean run disables that diagnostic collective.
Separate warm-up observers count actual Python collective submissions.
The metadata-only control must preserve logits and generated tokens exactly.
The combined source-owner variant retains separate numerical and output-parity fields.

All mandatory batch sizes and prefix lengths receive this ablation.
These results remain generic runtime evidence, not a new loop-aware architecture.

## Raw-data preservation and publication

Full raw outputs, traces, failed attempts, and complete benchmark files remain under `raw/` locally.
The repository excludes those large raw directories and runtime caches.
`raw_evidence_manifest.json` records raw file paths, lengths, and hashes after completion.
`archives/` contains compact numeric baseline records and representative CUPTI traces.
`router_trace_sample.jsonl.gz` is a safe numeric router sample.
The publication does not redistribute the full benchmark question sets.
No original raw file is deleted for publication.
The large `ep_affinity_metrics.csv` remains local in full.
Its tracked `ep_affinity_metrics.csv.gz` contains identical uncompressed bytes.
`quality/compressed_table_provenance.json` records that identity.

## Safe Git release

The release helper accepts only this study's three approved paths.
It requires the expected branch and base commit.
It hashes the unrelated staged diff before and after the commit.
An intervening commit or a changed unrelated index blocks automatic release.
The helper never resets, cleans, rebases, or force-pushes.
It compares the remote branch SHA with the actual new commit.
A failed push remains `COMMITTED_PUSH_PENDING`, not a claimed success.
The isolated local-repository regression checks this index-preservation behavior.

Unused authorized GPUs retain verified owned burners.
No foreign process is eligible for termination.

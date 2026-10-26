# Reproduction

## Prerequisites

Use the repository root as the working directory.
Use the exact model revisions in `preflight.json`.
Do not place model weights in Git.
Preserve IFM artifact manifests and native file layout.

GPU worker Python:
`/home/esjung/.venvs/looped-moe-headroom-py312/bin/python`.

CPU analysis Python:
`/home/esjung/anaconda3/bin/python`.

The worker environment uses Python 3.12.13 and PyTorch 2.11.0+cu129.
The compiled CUDA version is 12.9.
Each fidelity record includes the actual NCCL version.
IFM uses supported FA4 in the pinned native checkout.
Foil uses its original remote code, grouped GEMMs, and SDPA path.

The new code reuses the previous study's checkpoint adapter and public corpus manifest.
That dependency belongs to baseline commit `532cd9023ce900272a5403464d54e9da45a9fe72`.
No upstream runtime file requires modification.

For a fresh environment, recreate the pinned native checkout and worker environment first.
The previous study records package versions in `poc_flashvep/looped_moe_headroom/requirements.lock.txt`.
Its `scripts/install_native.sh` records native installation steps.
Review those commands before using a different CUDA or PyTorch release.

The public workload manifest contains dataset revisions, request IDs, and text hashes.
Reconstruct its local text only when the full local manifest is absent:

```bash
/home/esjung/.venvs/looped-moe-headroom-py312/bin/python -m poc_flashvep.looped_moe_headroom.reconstruct_workloads
```

Use `--verify-only` when the full manifest already exists.
The reconstruction checks every frozen text hash.
Do not replace missing public samples with generated proxy text.

## Restore archived evidence

Run this command after a fresh clone:

```bash
python -m poc_flashvep.looped_moe_ep.archive --extract
```

The extractor checks archive hashes and existing file hashes.
It refuses to overwrite different evidence.
The original local raw files remain intact after archive creation.
The archive excludes checkpoints, environments, and compiler caches.

## CPU checks

```bash
/home/esjung/.venvs/looped-moe-headroom-py312/bin/python -m poc_flashvep.looped_moe_ep.tests
/home/esjung/anaconda3/bin/python -m compileall -q poc_flashvep/looped_moe_ep
```

The unit fixtures test accounting only.
They are not model proxies or performance evidence.

## Launch a new native/EP check

Inspect GPU ownership first.
Choose an unused run ID.
The launcher rejects partial existing attempts.
It skips only fully completed IDs.

```bash
/home/esjung/anaconda3/bin/python -m poc_flashvep.looped_moe_ep.launch \
  --model ifm_ouro --gpus 0,1,2,3 --id ifm_ouro_ep4_p0_new --phase p0
```

Use `ifm_huginn` or `foil1` for the other checkpoints.
Use `0,1,2,3,4,5,6,7` for EP8.
The launcher checks ownership before it stops approved burner sessions.
It restores approved burners after the job, including on failure.
It never treats permission to use a GPU as permission to terminate unrelated work.

## Recorded campaign

The immutable configs reproduce every measured cell.
The launcher records source hashes, device mappings, process IDs, and separate rank logs.
The recorded campaign uses six P0 jobs, eighteen baseline jobs, six quality jobs, six ready observers, and six warmed controls.
EP4 start 1 uses GPUs 4–7.
EP4 starts 0 and 2 use GPUs 0–3.
EP8 uses all eight GPUs.
Prompt-cohort offsets are 0/4/8 across those starts.
The resulting intervals include workload variation, not only restart variation.

The baseline phase waits for successful EP8 P0 completion.
The quality phase waits for the baseline phase.
The ready observer waits for all quality jobs.
The warmed control waits for the ready observer campaign.
Do not launch those phases independently against the same GPUs.

```bash
/home/esjung/anaconda3/bin/python -m poc_flashvep.looped_moe_ep.campaign ep8_p0
/home/esjung/anaconda3/bin/python -m poc_flashvep.looped_moe_ep.campaign baseline
/home/esjung/anaconda3/bin/python -m poc_flashvep.looped_moe_ep.campaign final_quality
/home/esjung/anaconda3/bin/python -m poc_flashvep.looped_moe_ep.campaign ready_probe
/home/esjung/anaconda3/bin/python -m poc_flashvep.looped_moe_ep.steady
```

The original EP4 P0 jobs used the individual launcher.
Their configs remain under `configs/`.
Changing a config requires a new run ID.
Do not overwrite a failed or partial attempt to resume it.

## Analyze completed evidence

```bash
taskset -c 128-143 /home/esjung/anaconda3/bin/python -m poc_flashvep.looped_moe_ep.analyze
taskset -c 128-143 /home/esjung/anaconda3/bin/python -m poc_flashvep.looped_moe_ep.diagnostics
taskset -c 128-143 /home/esjung/anaconda3/bin/python -m poc_flashvep.looped_moe_ep.oracles
taskset -c 128-143 /home/esjung/anaconda3/bin/python -m poc_flashvep.looped_moe_ep.characterize
taskset -c 128-143 /home/esjung/anaconda3/bin/python -m poc_flashvep.looped_moe_ep.finalize
taskset -c 128-143 /home/esjung/anaconda3/bin/python -m poc_flashvep.looped_moe_ep.integrity
```

The finalizer refuses incomplete campaigns.
It writes summaries from actual completed evidence only.
Oracle files remain labeled as modeled estimates.
No command reports a candidate A/B speedup without a real optimized implementation.

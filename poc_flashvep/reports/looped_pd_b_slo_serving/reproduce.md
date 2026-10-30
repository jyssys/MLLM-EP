# Reproduction and safe resume

## Restore the exact extended dataset after cloning

GitHub rejected the first unpublished commit because `data/W1_long.json` contains126,308,939 bytes.
The local original remains unchanged. The published tree contains its verified gzip representation instead.
This storage change occurs after GPU measurements. It changes no experiment input.

```bash
CUDA_VISIBLE_DEVICES='' /home/esjung/.venvs/looped-pd-abc/bin/python -m poc_flashvep.looped_pd_b_slo_serving.data_artifacts --restore
```

`configs/dataset_storage.json` contains both SHA-256 values and byte counts.
The restore command refuses to overwrite a different existing dataset.
The recorded worker source and its input path remain unchanged.
The original wall-clock deadline does not reset. Do not restart this campaign after its deadline.

The source is `poc_flashvep/looped_pd_b_slo_serving/`.
The parent experiment remains at commit `b82eabd20a4ec168a5ad9a119c8401f97a565e78`.
The private RLT source is `/home/esjung/external/looped-pd-abc-rlt`, upstream commit `6b164d3844d35ab8e753df1864f0c87dc180401a`.
The existing runtime patch SHA-256 is `4904a311358c5cf6467678d0e1c30af38b5a96064cae4c96ed680943d7e694c7`.
The follow-up changes process-local output paths and the arrival harness. It does not edit that model/runtime patch.

## Dataset and CPU tests

```bash
CUDA_VISIBLE_DEVICES='' PYTHONPATH=/home/esjung/external/looped-pd-abc-rlt:/home/esjung/MLLM-EP-github /home/esjung/.venvs/looped-pd-abc/bin/python -m poc_flashvep.looped_pd_b_slo_serving.prepare
CUDA_VISIBLE_DEVICES='' PYTHONPATH=/home/esjung/external/looped-pd-abc-rlt:/home/esjung/MLLM-EP-github /home/esjung/.venvs/looped-pd-abc/bin/python -m pytest -q poc_flashvep/looped_pd_b_slo_serving/test_contracts.py
```

The initial CPU suite passes five tests.
The raw source dataset remains in the HF download cache. The report retains its revision and hash.
The selected text/token manifests remain in `data/`.

## Qualification and campaign

```bash
CUDA_VISIBLE_DEVICES='' PYTHONPATH=/home/esjung/external/looped-pd-abc-rlt:/home/esjung/MLLM-EP-github /home/esjung/.venvs/looped-pd-abc/bin/python -m poc_flashvep.looped_pd_b_slo_serving.qualify
CUDA_VISIBLE_DEVICES='' PYTHONPATH=/home/esjung/external/looped-pd-abc-rlt:/home/esjung/MLLM-EP-github /home/esjung/.venvs/looped-pd-abc/bin/python -m poc_flashvep.looped_pd_b_slo_serving.campaign
```

Do not start duplicate controllers. Check `state/qualification_controller.json` and `state/campaign.json` first.
The current controller records the original clock. A resume does not reset its deadline.
Every terminal run name is reused rather than silently rerun.
A changed configuration requires a new name. Failed logs remain available.

Each command launches the existing qualified RLT models through a separate follow-up wrapper.
The wrapper preserves the prior source and redirects only its process-local output directory.
The old source ZIP and new `followup_sources.zip` jointly identify each executed implementation.
`serving_contract.json` and `arrival_trace.json` define the actual arrival workload.
The first two calibration jobs retain an unused legacy argparse rate field of zero.
Their replacement serving callback reads the nonzero rate from the saved follow-up contract.
Before held-out serving, that metadata field also receives the actual rate.
This metadata correction does not change the arrival function or GPU model computation.

## Diagnostics and measurement

The registered nonblocking recorder remains enabled in workers and spawned stage processes.
It has JSON/enqueue overhead and background I/O cost. It is not a zero-cost recorder.
Heavy KV checks and event traces stay separate from clean capacity runs.
`configs/01_looped_pd_b_timing.pth` is installed only in the private main environment.
It imports diagnostic hooks only when `B_EVENT_TRACE=1`.
Normal serving does not enable that flag.

The event hooks record first observed CUDA readiness, NIXL submission, and observed completion.
These are host observations. They are not exact GPU DMA start/end timestamps.
No observed event interval is added to request E2E as an independent cost.

Separate two-request native profiles reuse RLT's existing `start_profile` and `stop_profile` operations.
The diagnostic launcher adds the existing `--profile` flag after the frozen wrapper constructs its arguments.
It does not change the worker source, model operators, or clean serving settings.
Native profile capture starts after warmup. Its latency is never a clean serving result.
Compressed Chrome traces remain in each diagnostic run's `profile/` directory.
Device-event unions measure recorded GPU activity only. Their idle denominator excludes unrecorded leading and trailing time.

Mismatch diagnostics compare one repeated control and two sequential matched arms on selected prompts.
Both sequential arms use the same subset-index seeds.
Huginn's initial-state random draws can therefore differ from the original full-trace sample-index draws.
Those diagnostics cannot establish answer accuracy or reopen the failed serving gate.

## Resource ownership

The resource helper recognizes only approved owned burner process sessions and log destinations.
It stops all owned burners before clean timing to prevent shared-resource interference.
It restores idle GPUs outside quiet timing windows and during final cleanup.
The user permits all physical GPUs0–7. No other device is used.
`worker_map.json` records actual stage UUIDs and physical GPU assignment.

A read-only all-GPU process observer starts at23:28 KST after the initial W1 points.
It samples process UID and registered experiment sessions every10s.
Earlier points have no continuous full-fabric observer coverage. Their setup and periodic manual process checks remain separate evidence.
The observer never signals a process. Unregistered activity marks the corresponding observed window as contended in the final capacity check.
This does not prove exclusive CPU, memory, storage, or network use.

## Analysis

```bash
CUDA_VISIBLE_DEVICES='' PYTHONPATH=/home/esjung/external/looped-pd-abc-rlt:/home/esjung/MLLM-EP-github /home/esjung/.venvs/looped-pd-abc/bin/python -m poc_flashvep.looped_pd_b_slo_serving.analyze
```

This command updates measured tables. It does not accept a scientific GO automatically.
Final publication uses only the new source/report paths. Unrelated dirty and staged work stays unchanged.

The commands describe this frozen campaign. Its absolute deadline does not reset on a later calendar date.
Do not execute a raw worker command against an existing run directory.
The guarded controller reuses terminal jobs. Direct worker execution does not provide that resume guard.
A new measurement campaign needs a new output namespace and an explicitly renewed budget.
CPU analysis of preserved evidence does not require a new GPU experiment.

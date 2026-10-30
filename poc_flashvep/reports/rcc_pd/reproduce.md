# Reproduction

## Safety and pins

Use the RCC branch `flashvep/rcc-pd-cross-depth-cobatching-poc`.
Its parent numerical release is `3b7a722898dab5f6dd797b474c1ff461659d8fa1`.
Read `dependency_contract.md` before model execution.
Keep the private RLT checkout and its preserved diff unchanged.
Do not restore an older numerical implementation.

Use a new run tag for a new experiment.
The dispatcher skips completed runs with the same tag.
It refuses to overwrite a partial run directory.
An incomplete run therefore needs investigation and a new tag.
Do not delete its evidence to force a restart.

## CPU preflight

Run these commands from `/home/esjung/MLLM-EP-github`.

```bash
python -m poc_flashvep.rcc_pd_poc.preflight
/home/esjung/.venvs/looped-pd-abc/bin/python -m poc_flashvep.rcc_pd_poc.audit
python -m compileall -q poc_flashvep/rcc_pd_poc
python -m poc_flashvep.rcc_pd_poc.resources status
```

The audit records the initial Git index for this campaign.
Do not replace a campaign's audit after staging its new files.
Use a new namespace if the source or prerequisite gate changes.
The output gate must mark each selected model `CORRECTNESS_READY`.

## Diagnostic execution

The following commands describe the measured campaign.
They are not instructions to interrupt another user's GPU job.
Check process ownership before a device lease.
The resource helper only stops recognized owned burners.

```bash
python -m poc_flashvep.rcc_pd_poc.resources stop --gpus 0,1,2,3
python -m poc_flashvep.rcc_pd_poc.discovery --model ouro --gpus 0,1 --tag diag128_v1 --length 128 --probe --verify-kv --eager --cap 32
python -m poc_flashvep.rcc_pd_poc.discovery --model huginn --gpus 2,3 --tag diag128_v1 --length 128 --probe --verify-kv --eager --cap 32
python -m poc_flashvep.rcc_pd_poc.discovery --model ouro --gpus 0,1 --tag diag256_v2 --length 256 --probe --verify-kv --eager --cap 16 --target-trace
python -m poc_flashvep.rcc_pd_poc.discovery --model huginn --gpus 2,3 --tag diag256_v2 --length 256 --probe --verify-kv --eager --cap 16 --target-trace
```

The actual command manifest records which diagnostic pairs ran concurrently.
These executions cannot provide clean latency claims.
Version 1 source is frozen in `configs/runtime_overlay_v1.zip`.
Current code includes the Version 2 Linear counter correction.
Rerunning Version 1 with current code is therefore not an exact instrumentation replay.
Use its archived overlay when exact instrumentation reproduction is required.

## Quiet comparison

Wait for every diagnostic worker to exit.
Check all eight physical devices before the quiet lane starts.
Temporarily stop owned burners on the shared fabric.
Do not stop any unrelated process.

```bash
python -m poc_flashvep.rcc_pd_poc.resources stop --gpus 0,1,2,3,4,5,6,7
nvidia-smi --query-compute-apps=pid,gpu_uuid --format=csv,noheader
python -m poc_flashvep.rcc_pd_poc.quiet
```

The quiet controller runs 18 independent server starts serially on GPU 0 and GPU 1.
It checks for an empty fabric before each start.
It stops a model if outputs differ.
It does not convert an output mismatch into accepted performance evidence.

## Heavy state comparison

These runs follow the quiet comparison.
They do not supply clean serving times.

```bash
python -m poc_flashvep.rcc_pd_poc.resources restore --gpus 4,5,6,7
python -m poc_flashvep.rcc_pd_poc.discovery --model ouro --gpus 0,1 --tag state128_v1 --length 128 --state-probe --verify-kv --eager --cap 16
python -m poc_flashvep.rcc_pd_poc.discovery --model huginn --gpus 2,3 --tag state128_v1 --length 128 --state-probe --verify-kv --eager --cap 16
```

## Analysis and figures

```bash
python -m poc_flashvep.rcc_pd_poc.analyze
python -m poc_flashvep.rcc_pd_poc.operator_evidence
python -m poc_flashvep.rcc_pd_poc.paired
python -m poc_flashvep.rcc_pd_poc.host_breakdown
python -m poc_flashvep.rcc_pd_poc.state_analysis
/home/esjung/.venvs/looped-pd-abc/bin/python -m poc_flashvep.rcc_pd_poc.figures
python -m poc_flashvep.rcc_pd_poc.validate
```

The validator expects this frozen 36-run campaign.
A different experiment matrix needs a separately declared validation contract.
The figures are editable SVG files.
The Mermaid architecture source is `architecture.mmd`.

## Restore the resource policy

After all workers exit, restore only the available owned burner devices.

```bash
python -m poc_flashvep.rcc_pd_poc.resources restore --gpus 0,1,2,3,4,5,6,7
python -m poc_flashvep.rcc_pd_poc.resources status
```

The helper invokes `/home/esjung/vllm-ep/run_utilize.sh` with the explicit physical GPU list.
It preserves the burner environment and ownership checks.
Burners were absent from the clean comparison.
They were restored after the diagnostic workers released their devices.

## Artifact map

| File | Purpose |
|---|---|
| `full_results.csv` | Per-run measured latency, memory, traffic, and work accounting |
| `paired_comparisons.csv` | Independent-start paired ratios and uncertainty |
| `correctness_matrix.json` | Token, depth, input, and stop agreement |
| `prompt_state_comparisons.json` | Selected prompt KV and last-hidden identity |
| `operator_fusion_evidence.json` | Real combined operands and selected kernel counts |
| `e2e_component_shares.csv` | Serial host-observed shares of summed request E2E |
| `experiment_matrix.csv` | Completed run settings |
| `unmeasured_serving_matrix.csv` | Explicit 2/4/8GPU and W1–W4 gaps |
| `validation.json` | Evidence integrity and preserved Git index |
| `command_manifest.jsonl` | Actual commands and configuration hashes |
| `raw/<run>/` | Output IDs, state checks, statistics, source archives, and logs |

The namespace contains no copied model weights.
Its token prefixes derive from the frozen previous ShareGPT dataset.
Each prefix manifest records its source archive hash and transformation.

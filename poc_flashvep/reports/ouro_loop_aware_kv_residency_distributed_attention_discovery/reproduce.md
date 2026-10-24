# Reproduction

## Evidence classes

`LIVE_MEASURED_SERVING` means an actual model process generated and delivered tokens.
The primary discovery lane ignores EOS and uses a constant output cap.
It measures bounded hardware inference, not sustainable natural serving capacity.
`LIVE_MEASURED_OPERATOR` means actual CUDA/NCCL execution of the recorded operator.
`MODEL_UPPER_BOUND` denotes tensor accounting or a conditional latency model.
`NOT_RUN` is not a zero measurement.

## Environment

Use `/home/esjung/.venvs/ouro-chunk-loop-prefill/bin/python` for GPU workers.
Use `/home/esjung/anaconda3/bin/python` for dispatch and analysis.
The runtime checkout must be `/home/esjung/external/ouro-kv-residency-rlt` at the registered revision.
It must have no tracked source diff.
Keep the pinned model snapshot and tokenizer.
The worker checks the physical GPU UUID.

Run from `/home/esjung/MLLM-EP-github`.
The following commands create a separate evidence root.
They do not overwrite the frozen study.

```bash
export OURO_KV_OUTPUT=/home/esjung/MLLM-EP-github/poc_flashvep/reports/ouro_kv_reproduction
export OURO_KV_FIXTURES=/home/esjung/.cache/ouro-kv-reproduction-fixtures
/home/esjung/anaconda3/bin/python -m poc_flashvep.ouro_kv_residency.register
/home/esjung/anaconda3/bin/python -m poc_flashvep.ouro_kv_residency.plans
/home/esjung/anaconda3/bin/python -m poc_flashvep.ouro_kv_residency.dispatch --manifest "$OURO_KV_OUTPUT/plans/smoke.json" --manage-burners
/home/esjung/anaconda3/bin/python -m poc_flashvep.ouro_kv_residency.campaign grid capture calibration profiles layout
/home/esjung/anaconda3/bin/python -m poc_flashvep.ouro_kv_residency.operator_campaign --tag reproduction --full --starts 3 --repeats 20
```

The source contains the corrected corpus-path expression.
The original failed 64K attempt remains in the study archive.
The reproduction does not need that failed attempt.
The first exploratory SplitKV failure also remains archived.
The current H100 path records unsupported local SplitKV without retrying it.

The dispatcher skips a completed immutable job after its hashes pass.
It refuses to overwrite failed or interrupted attempts.
Use a new output root for a fresh GPU run.
Do not delete a frozen attempt to force a rerun.

## CPU reanalysis

The raw archives retain original relative paths.
Extract them into the report root in a separate analysis workspace.
Keep the source hashes and archive manifest.
The external tensor fixtures contain model activations, not weights.
Their full hashes and reconstruction inputs remain in each capture job.
They reside at `/home/esjung/.cache/ouro-kv-residency-fixtures` on this server.
Large tensor fixtures are not embedded in Git.

```bash
/home/esjung/anaconda3/bin/python -m poc_flashvep.ouro_kv_residency.tests
/home/esjung/anaconda3/bin/python -m poc_flashvep.ouro_kv_residency.profile_analysis
/home/esjung/anaconda3/bin/python -m poc_flashvep.ouro_kv_residency.analyze
/home/esjung/anaconda3/bin/python -m poc_flashvep.ouro_kv_residency.figures
```

The profile parser does not convert service sums into clean E2E percentages.
It identifies complete R4 GPU windows between consecutive sampling kernels.
Each retained window contains 96 actual attention kernels.
Incomplete capture boundaries do not enter that table.

## GPU ownership and burners

The resource module checks process ownership and approved burner logs.
It rejects an unrelated occupied GPU.
Never replace that check with a broad process-name kill.

```bash
/home/esjung/anaconda3/bin/python -m poc_flashvep.ouro_kv_residency.resources status
/home/esjung/anaconda3/bin/python -m poc_flashvep.ouro_kv_residency.resources restore 0,1,2,3,4,5,6,7
```

Quiet comparison groups reserve unused devices without burners.
The final restoration explicitly selects all eight authorized GPUs.
The server script's default device list must not be assumed.

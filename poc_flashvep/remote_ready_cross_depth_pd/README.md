# Remote-ready Cross-depth PD audit

This audit does not run a new GPU experiment.
It joins recorded source cuts, imports, and selected D batches by transfer generation.
It excludes warmup requests and keeps missing ready queues unknown.

The candidate equals the existing A import path plus Native Mixed.
The requested equivalence gate therefore stops Phase B and Phase C.
This result does not prove that every possible ready-work opportunity was used.

## Reproduce the CPU analysis

Run these commands from `/home/esjung/MLLM-EP-github`.
Use the preserved external RLT source at `/home/esjung/external/looped-pd-abc-rlt`.
The evidence manifest records the required historical input files.
The source audit checks the external runtime hashes.

```bash
CUDA_VISIBLE_DEVICES='' /home/esjung/.venvs/looped-pd-abc/bin/python -m poc_flashvep.remote_ready_cross_depth_pd.audit
CUDA_VISIBLE_DEVICES='' /home/esjung/.venvs/looped-pd-abc/bin/python -m unittest poc_flashvep.remote_ready_cross_depth_pd.test_audit
CUDA_VISIBLE_DEVICES='' /home/esjung/.venvs/looped-pd-abc/bin/python -m poc_flashvep.remote_ready_cross_depth_pd.figures
```

The initialization guard preserves the original checkout, index, and runtime diff.
It rejects an unexpected change to that saved state.
The analysis does not modify frozen input evidence.
The CPU descriptor replay does not measure GPU timing.

## Closeout

`verify` checks input hashes, nine tests, local report links, and approved burner ownership.
It also requires fresh generation records from all eight approved burners.
Run it only in the original guarded experiment environment.

`publish --push` publishes only this namespace and its two report files.
It uses a private Git index and a separate branch.
It does not change the user's checkout or staged index.
Its publication receipt remains local because it contains the final commit SHA.

```bash
CUDA_VISIBLE_DEVICES='' /home/esjung/.venvs/looped-pd-abc/bin/python -m poc_flashvep.remote_ready_cross_depth_pd.verify
CUDA_VISIBLE_DEVICES='' /home/esjung/.venvs/looped-pd-abc/bin/python -m poc_flashvep.remote_ready_cross_depth_pd.publish --push
```

[Final report](../reports/ouro_huginn_remote_ready_cross_depth_pd_poc_final_summary.md)

# GPU-resumption delivery record

Date:2026-09-08. Branch:`flashvep/top-tier-successor-mining`.
Final decision:FOUND_INCREMENTAL_ONLY. Best diagnosed baseline:SERE,not promoted.
Main report:`poc_flashvep/reports/top_tier_successor_mining.md`.
Results:`poc_flashvep/deepep_revalidation/results/top_tier_successor_mining_20260907_135950/`.

GPU experiments and post-run checks:approximately11:03–15:23 KST.
Reporting/tests/versioning continue afterwards;final wall time is reported at
handoff rather than counted as GPU-active time. Resumed recorded GPU-resident
hours14.360,including loads/CPU control;historical+resumed25.309. Burn contributes0.

All measurement workers are stopped. Research telemetry PID1969380 was stopped
after its identity was checked. Other users'0–3 jobs are untouched.

## User-requested final utilization

The local shell script actually defaulted to1–4,so its GPU variable was made
environment-overridable while preserving that default. The explicit launch is:

```bash
VLLM_UTILIZE_GPUS=4,5,6,7 CUDA_VISIBLE_DEVICES=4,5,6,7 \
  bash /home/esjung/vllm-ep/run_utilize.sh
```

The first detached launch did not remain alive and was not counted as successful.
The verified replacement is running in execution session58255:
parent utilize PID2407937;worker launchers2408259/2408260/2408261/2408262.
At15:30:52 KST all four physical UUIDs match the authorized4–7 mapping and each
reports100% GPU utilization. Log confirms generation started on4,5,6,7.
Log:`raw/final_utilize_20260908.log` under the result root.
Script duration is108000seconds(30h),not an indefinite service guarantee.
This process is intentionally left running at handoff as requested. It is not
an experiment and no result is collected while it overlaps a research run.

## Versioning

Task code/docs and curated summaries/plots are committed;model weights,private
Libra patches,cloned references,images/raw answers and large Nsight binaries are
kept local. The unrelated dirty execution-regime spec is not staged.
Use `git log -1 --format='%H %s'` on this branch for the delivered commit;the
commit hash/push outcome are also supplied in the final response. This avoids
putting a self-referential commit hash inside its own content.

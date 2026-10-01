# Reproduction

Use the existing validated environment; no runtime upgrade is required. The
upstream semantic reference is pinned in `spec.md` and
`upstream_semantics_audit.md`. This is real stateful model execution, not replay.

```bash
cd /home/esjung/MLLM-EP-native-lp
export CUDA_VISIBLE_DEVICES=0,1,4,5
export PYTHONPATH="$PWD"
/home/esjung/.venvs/flashvep-deepep-v020/bin/python -m unittest discover \
  -s poc_native_lp_ep4/tests -v
/home/esjung/.venvs/flashvep-deepep-v020/bin/python \
  -m poc_native_lp_ep4.scripts.blocked_campaign --version v6
```

The campaign first checks prior native correctness and the atomic-admission
three-repeat boundary test. It creates fresh engine sessions and new run
directories; completed blocks can be reused without rewriting raw evidence.
Failed blocks are not silently retried. Use a newly versioned campaign/output
plan for a genuinely changed implementation. The exact runtime commands,
GPU mask, model revision, scheduler arms, fixed continuations and source hashes
are recorded in `artifacts/campaign_ledger.jsonl` and per-run manifests.

Per-model wave watchdog: 90 s after model readiness; startup watchdog: 420 s.
Every runtime command additionally has a hard scoped GPU-session timeout.
The model stays BF16/K8 and all natural correctness continuations are bounded
to eight tokens; victim system timing is fixed96 discovery/fixed128 final.
No full benchmark evaluation, simulator, profiler stream or new communication
slicing method runs.

For a completely fresh correctness run, the working startup-hook invocation
is in `README.md`; add `--atomic-wave-admission` and use a new output directory.
The blocked campaign's precondition manifests can then be updated to that
completed independently validated run. Do not relabel the legacy interrupted
v4 sweep as complete.

Independent reanalysis after completed v6 blocks:

```bash
/home/esjung/.venvs/flashvep-deepep-v020/bin/python \
  -m poc_native_lp_ep4.scripts.metrics --version v6 \
  --output poc_native_lp_ep4/artifacts/reanalysis_new
/home/esjung/anaconda3/bin/python -m poc_native_lp_ep4.scripts.figures \
  --metrics poc_native_lp_ep4/artifacts/reanalysis_new/metrics.json \
  --output poc_native_lp_ep4/figures/reanalysis_new
```

Plotting uses the already-installed CPU-only Matplotlib environment, not an
upgrade of the experiment environment. Tables and figures contain measured
points only. Bootstrap unit is the wave/repeat, never individual MoE layers.
The pool is intentionally fixed; this is not a CI for all possible images.

The GPU wrapper restores the existing burner policy on success/failure. The
server script's actual default mask differs from the requested devices, so the
wrapper explicitly supplies `VLLM_UTILIZE_GPUS=0,1,4,5`. Only authorized own
experiment/burner processes are managed; other GPUs are not used or terminated.

After cloning the evidence branch, independent CPU reanalysis can restore the
verified compressed JSON without model weights or GPU use. The restore target
must not already exist:

```bash
/home/esjung/.venvs/flashvep-deepep-v020/bin/python \
  -m poc_native_lp_ep4.scripts.verify_compact poc_native_lp_ep4/artifacts/compact_raw
/home/esjung/.venvs/flashvep-deepep-v020/bin/python \
  -m poc_native_lp_ep4.scripts.restore_compact poc_native_lp_ep4/artifacts/compact_raw \
  --output poc_native_lp_ep4/artifacts/restored_runs
/home/esjung/.venvs/flashvep-deepep-v020/bin/python \
  -m poc_native_lp_ep4.scripts.metrics --version v6 \
  --raw-root poc_native_lp_ep4/artifacts/restored_runs \
  --output poc_native_lp_ep4/artifacts/independent_reanalysis
```

Large `.pt` correctness tensors stay local. Compact raw JSON/JSONL is exported
with SHA256 and verified by `export_compact.py` / `verify_compact.py`. These are
actual per-token engine clocks, routing/counter records, scheduler progress,
runtime manifests and numerical differences, not synthesized measurements.

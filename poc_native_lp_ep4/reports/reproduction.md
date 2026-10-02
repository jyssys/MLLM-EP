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
  -m poc_native_lp_ep4.scripts.verify_compact poc_native_lp_ep4/artifacts/compact_raw
```

The historical v6 campaign produced immutable clean discovery/final data; v7
continues with trace-only repair and collection. Do not invoke the historical
v6 source-hash guard against the changed current tracer. Its original source
is checkpoint `6e318988`. `complete_campaign.py` documents the actual local
continuation, including failed-startup preconditions, not a fresh-clone runner.
The exact executed commands, GPU mask, model revision, scheduler arms, fixed
continuations and source hashes are in the ledger and per-run manifests.

Per-model wave watchdog: 90 s after model readiness; startup watchdog: 420 s.
Every runtime command additionally has a hard scoped GPU-session timeout.
The model stays BF16/K8 and all natural correctness continuations are bounded
to eight tokens; victim system timing is fixed96 discovery/fixed128 final.
No full benchmark evaluation, simulator, profiler stream or new communication
slicing method runs.

For a completely fresh correctness run, use `README.md`, including the
atomic admission, uniform Full resident initialization and event-pool flags.
Use a new output directory and validate it with `check_correctness.py` before
timing. Do not relabel an interrupted historical run as complete.

## Fresh ten-repeat clean confirmation

These are the same bounded runtime arguments as the completed final lane.
Five two-repeat restarts give ten repeats without an unvalidated long session.
Use a new tag/output path for every reproduction. The frozen arms are Full,
CP512, AUTO and G8; do not select a new winner from these final measurements.

```bash
export CUDA_VISIBLE_DEVICES=0,1,4,5
export PYTHONPATH="$PWD/poc_native_lp_ep4/hooks:$PWD"
export NATIVE_LP_ENABLE=1 NATIVE_LP_TRACE=0 NATIVE_LP_CORRECTNESS=0
export NATIVE_LP_LIGHT_TRACE=0 NATIVE_LP_EVENT_POOL=1
export NATIVE_LP_VISION_CACHE=0
export MLLM_COMM_CONTROL="$PWD/poc_native_lp_ep4/artifacts/control.json"
for block in 0 1 2 3 4; do
  export NATIVE_LP_OUTPUT="$PWD/poc_native_lp_ep4/artifacts/runs/reproduce_final_block${block}"
  export MLLM_COMM_OUTPUT="$NATIVE_LP_OUTPUT"
  /home/esjung/.venvs/flashvep-deepep-v020/bin/python \
    poc_native_lp_ep4/scripts/gpu_session.py --timeout 1500 \
    /home/esjung/.venvs/flashvep-deepep-v020/bin/python \
    poc_native_lp_ep4/scripts/run.py --output "$NATIVE_LP_OUTPUT" \
    --arms FULL,CP512,LP_AUTO,LP_G8 \
    --modalities VISUAL_PREFILL,TOKEN_MATCHED_TEXT --repeats 2 \
    --repeat-offset "$((block * 2))" --decode-tokens 128 --inject-at 16 \
    --preprocess-before-injection --pair-baseline \
    --atomic-wave-admission --uniform-resident-prefill || break
done
```

For light components set `NATIVE_LP_TRACE=1`, `NATIVE_LP_LIGHT_TRACE=1` and
add `--trace --interleave-instrumentation --repeats 3`. Use **only two arms**
per paired session: `(FULL,CP512)` or `(LP_AUTO,LP_G8)`; each is 48 waves with
warmup. For extra sweep arms omit `--pair-baseline` and use up to three arms
per block. For rich routes set `NATIVE_LP_LIGHT_TRACE=0`, add
`--trace --route-stats`, omit the overhead/pair-baseline switches, and use
at most three arms per block including Full. The exact completed partitions
are recorded by `campaign_blocks_v7.json`.

The `--repeats 3` argument replaces the two-repeat clean value. Rich trace wall
is not serving latency. On Full/text the component overhead screen still failed
at 10.302%; that stratum remains UNVALIDATED rather than rerunning until it passes.

The executed secondary producer command was:

```bash
PYTHONPATH="$PWD" /home/esjung/.venvs/flashvep-deepep-v020/bin/python \
  -m poc_native_lp_ep4.scripts.producer_control --version v7
```

It passed numerical identity before the two three-repeat diagnostic timing
passes. To reproduce independently without overwriting published evidence,
first execute the uncached correctness command into `new_correctness` (the local
`.pt` hidden/logit files are required), then use a new tag:

```bash
PYTHONPATH="$PWD" /home/esjung/.venvs/flashvep-deepep-v020/bin/python \
  -m poc_native_lp_ep4.scripts.producer_control --version v7 \
  --uncached-reference poc_native_lp_ep4/artifacts/runs/new_correctness \
  --tag independent1
```

This diagnostic includes exact CPU input hashing; it is not a caching method
speedup. Its source-argument defaults match the executed original command.

Independent reanalysis after completed v7 trace collection:

```bash
/home/esjung/.venvs/flashvep-deepep-v020/bin/python \
  -m poc_native_lp_ep4.scripts.metrics --version v7 \
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
  -m poc_native_lp_ep4.scripts.metrics --version v7 \
  --raw-root poc_native_lp_ep4/artifacts/restored_runs \
--output poc_native_lp_ep4/artifacts/independent_reanalysis
/home/esjung/.venvs/flashvep-deepep-v020/bin/python \
  -m poc_native_lp_ep4.scripts.verify_reanalysis \
  --original poc_native_lp_ep4/artifacts/final_metrics.json \
  --rebuilt poc_native_lp_ep4/artifacts/independent_reanalysis/metrics.json \
  --output poc_native_lp_ep4/artifacts/new_reanalysis_check.json
```

Large `.pt` correctness tensors stay local. Compact raw JSON/JSONL is exported
with SHA256 and verified by `export_compact.py` / `verify_compact.py`. These are
actual per-token engine clocks, routing/counter records, scheduler progress,
runtime manifests and numerical differences, not synthesized measurements.

The executed fresh archive check passed **304 compressed/raw SHA256 checks**
and **39,926 numeric field comparisons**, allowing only 1e-7 absolute/1e-10
relative floating accumulation tolerance. Only run-directory prefixes and
metadata-keyed row order are normalized; no numerical field is dropped.
`archive_reanalysis_verification.json` records the actual fresh temporary paths.
The final release records 23 executed unit tests, compileall, source identity,
correctness and authorized-burner UUID/PID verification separately.

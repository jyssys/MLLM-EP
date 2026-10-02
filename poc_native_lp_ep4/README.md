# Native Layered Prefill on TRUE EP4

Authoritative design: [spec.md](spec.md). Pinned upstream:
`scale-snu/layered-prefill@053f80e5201a7c0ab56e468e3d578907a2ca9cc3`.

This is a native, stateful, bounded Qwen3-VL/vLLM extension, not Layered EP
physical replay. Prefill hidden/residual state survives real serving iterations;
resident decode runs all 48 layers and generates a token each iteration.
Only GPUs **0,1,4,5** are permitted. All arms use TP2/DP2/EP4/PP1, BF16,
DeepEP HT, original K8 and linear expert ownership.

The installed vLLM package, upstream checkout, other branches and existing
PoC results are not modified. The isolated startup hook fails closed if the
native integration cannot install. No approximation or new communication
slicing method is implemented.

## Completed finding

See [final_summary.md](reports/final_summary.md), [serving latency](reports/serving_latency.md)
and [measured Pareto analysis](reports/pareto_analysis.md). Native correctness,
real token emission every stage and exact prefill routing pass. AUTO follows the
official rule and is G4. LP retains 1.000x expert reload; CP512 is 3.798x.

Ten paired repeats show a common first-iteration visual producer penalty, **not
an isolated common EP wire/collective bottleneck**. The verdict is
`MODALITY_COMMON_RESIDUAL_FOUND`, limited to this tested producer/runtime path.
Do not use it to promote communication-only slicing. All accepted raw JSON is
published compressed (304 files); independent restoration reproduces 39,926
numeric fields. Hidden/logit tensors remain local, and older failed diagnostics
are retained locally with a published exclusion audit.

## Running

Use the previously validated environment and repository data assets. From the
repository root (a clean worktree is recommended):

```bash
export CUDA_VISIBLE_DEVICES=0,1,4,5
export PYTHONPATH="$PWD/poc_native_lp_ep4/hooks:$PWD"
export NATIVE_LP_ENABLE=1
export NATIVE_LP_TRACE=1
export NATIVE_LP_LIGHT_TRACE=1
export NATIVE_LP_CORRECTNESS=1
export NATIVE_LP_EVENT_POOL=1
export NATIVE_LP_VISION_CACHE=0
export NATIVE_LP_OUTPUT="$PWD/poc_native_lp_ep4/artifacts/runs/new_correctness"
export MLLM_COMM_OUTPUT="$NATIVE_LP_OUTPUT"
export MLLM_COMM_CONTROL="$PWD/poc_native_lp_ep4/artifacts/control.json"
/home/esjung/.venvs/flashvep-deepep-v020/bin/python \
  poc_native_lp_ep4/scripts/gpu_session.py --timeout 2000 \
  /home/esjung/.venvs/flashvep-deepep-v020/bin/python \
  poc_native_lp_ep4/scripts/run.py --output "$NATIVE_LP_OUTPUT" \
  --arms FULL,LP_AUTO,LP_G8,LP_G16,LP_G24,LP_G48 \
  --modalities VISUAL_PREFILL,TOKEN_MATCHED_TEXT --repeats 3 \
  --decode-tokens 96 --inject-at 16 --preprocess-before-injection \
  --correctness --trace --atomic-wave-admission --uniform-resident-prefill
```

The output directory must be new. Numerical/stage validation:

```bash
/home/esjung/.venvs/flashvep-deepep-v020/bin/python \
  poc_native_lp_ep4/scripts/check_correctness.py \
  --run poc_native_lp_ep4/artifacts/runs/new_correctness \
  --output poc_native_lp_ep4/artifacts/correctness/new_validation.json
```

For a **clean** serving run set `NATIVE_LP_TRACE=0`,
`NATIVE_LP_CORRECTNESS=0`, remove `--trace --correctness`, use
`--decode-tokens 128 --pair-baseline --repeats 2`, and a new output directory.
Run five independent two-repeat blocks with `--repeat-offset 0,2,4,6,8`
respectively for ten final repeats. With four arms and two modalities, each
block contains 48 waves including warmup. Do not put ten repeats into one
long external-DP session. Set the frozen arms from discovery, not final results.

Historical `blocked_campaign.py --version v6` collected the clean discovery
and ten-repeat final evidence. Its original source is checkpoint `6e318988`.
Do not run that historical source-hash guard against the repaired current
tracer. `complete_campaign.py` is the recorded **local continuation** after a
diagnosed rendezvous startup failure and event-allocation overhead; it reuses
immutable v6 clean results and collects v7 trace/correctness only. It is not a
fresh-clone entrypoint: its failed-startup audit preconditions are deliberate.
Use the explicit fresh-run commands in [reproduction.md](reports/reproduction.md).
All arms initialize resident KV using the same ordinary Full prefill. Commands,
source hashes, frozen selection, watchdogs and restart boundaries are preserved
in the ledger and per-run manifests. There are no automatic failed-wave retries.
`--interleave-instrumentation` is an overhead diagnostic, not clean latency.

## Boundaries

The primary token clock is actual `EngineCoreOutputs.timestamp`, not external
client polling. CPU rendering is performed before the frozen model-ready
arrival; GPU vision encoding remains enabled. Thus headline TTFT is model-ready
arrival to engine first-token completion, not a full frontend/user TTFT.
Language-entry to first-token wall is reported separately. CPU lifecycle
timestamps are not GPU collective wait-time measurements.

LP Auto is **4 stages** for 2029/2033 prompt tokens under the pinned official
rule. G4 is therefore an identical algorithmic point, not an independent
baseline. G24/G48 are finer-group falsification controls.

Route-rich diagnostics and clean latency are separate. Expert-weight bytes
are derived from loaded tensors; reactivation bytes are a working-set proxy,
not measured HBM traffic. Local CUDA elapsed times are not cross-rank global
timestamps. Empty/unmeasured quantities are N/A, never fabricated zeros.

The burner script's server default is not the authorized mask. The wrapper
explicitly sets `VLLM_UTILIZE_GPUS=0,1,4,5` and restores it from its original
directory. Other GPUs/users are untouched. Run wrappers are independently
session-scoped and serialized by a scoped lock; do not kill a parent campaign
mid-run to modify its plan.

## Tests / compact raw evidence

```bash
/home/esjung/.venvs/flashvep-deepep-v020/bin/python -m unittest discover \
  -s poc_native_lp_ep4/tests -v
/home/esjung/.venvs/flashvep-deepep-v020/bin/python \
  poc_native_lp_ep4/scripts/verify_compact.py poc_native_lp_ep4/artifacts/compact_raw
```

Large hidden/logit tensors and full local run directories remain local.
Published compressed raw traces, per-token clocks, manifests, correctness
differences and hashes permit independent reanalysis/reproduction without
copying model weights, credentials or images.

`stage_latency.py` provides a separately labeled first-versus-later-stage
analysis from actual engine-completion tokens. It neither replaces the primary
whole-window peak nor reselects the best LP arm. Host completion stamps are
never called GPU collective-wait timestamps.

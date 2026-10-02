# Reproduction and actual execution ledger

Worktree: `/home/esjung/MLLM-EP-dispatch-asymmetry`; branch `flashvep/dispatch-combine-asymmetry-discovery`; base `2553b64a7993ff681cafe63b7627b001f4bcd3d1`.
Requested spec basename was absent. Unique matching source was read in full: `/home/esjung/MLLM-EP-github/poc_flashvep/reports/dispatch_combine_asymmetry_discovery_spec_gpu0145.md`; snapshot and SHA saved. This is not a substitution with an older research spec.

The installed vLLM/DeepEP environment and cached BF16 checkpoints were reused without upgrades. Original repository staged/dirty changes are separate and preserved. All new code/evidence is under this project.

## Actual GPU runs (already complete)

| Run | Executed scope | Headline use |
|---|---|---|
| high_baseline_v1 | first high visual attempt | EXCLUDED: external-DP idle forwards mixed into label; strict analyzer rejected |
| mllm_clean_v2 | visual+text × 3 sizes × (3 warmup+10 repeats) | final clean primary |
| llm_controls_v3 | LLM text × 3 sizes × (3 warmup+10), then 3 diagnostic repeats/size and 5 bypass pairs | clean primary only; appended diagnostics separate |
| mllm_internal_v3 | visual+text × 3 sizes × (3 warmup+3 repeats), then 5 bypass pairs/arm | route/accounting evidence, NOT clean headline |
| route_controlled_replay | captured layers 0/24/47 × 3 sizes × 2 arms, warmup3/repeat10 | physical DeepEP payload replay, no model/expert inference |
| mllm_nsight_v4 / llm_nsight_v4 | high-size native profile, warmup3/one diagnostic repeat per arm | kernel classification only |
| mllm_overhead_v5 | high visual/text, common model envelope, 3 warmup+10 bypass/clean pairs | measured instrumentation overhead only |

Every model wave creates one actual request on DP0, with native coordinated idle/dummy DP1 calls. Output cap=1; deterministic, no benchmark reevaluation. `execution_ledger.json` gives actual counts, including excluded/warmup requests. All successful run status/worker records are preserved in `compact_raw`. Per-wave no-progress watchdog=420s; outer GPU-session watchdog is finite. No infinite retry.

The first failed payload-replay launcher referenced a nonexistent venv `torchrun` executable and exited before measurement. It was corrected to `python -m torch.distributed.run`; the corrected replay completed. Neither failure was turned into a research NO-GO or included in headline statistics.

## GPU reproduction — only after a new allocation

**The GPUs were handed off. Do NOT execute this section now.** Commands below describe the already validated launch structure; select a fresh output directory. The old resource wrapper restores the existing burner after the run, so use it only when burner restoration is again authorized by the GPU owner.

```bash
cd /home/esjung/MLLM-EP-dispatch-asymmetry
export CUDA_VISIBLE_DEVICES=0,1,4,5
export PYTHONPATH=/home/esjung/MLLM-EP-dispatch-asymmetry/poc_dispatch_combine_asymmetry/hooks:/home/esjung/MLLM-EP-dispatch-asymmetry
export ASYMMETRY_HOOK=1
export ASYMMETRY_CONTROL=/home/esjung/MLLM-EP-dispatch-asymmetry/poc_dispatch_combine_asymmetry/artifacts/control_repro.json
export ASYMMETRY_OUTPUT=/home/esjung/MLLM-EP-dispatch-asymmetry/poc_dispatch_combine_asymmetry/artifacts/runs/mllm_clean_repro_NEW
export NATIVE_LP_OUTPUT=$ASYMMETRY_OUTPUT
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export VLLM_NO_USAGE_STATS=1
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

/home/esjung/.venvs/flashvep-deepep-v020/bin/python -m poc_native_lp_ep4.scripts.gpu_session --timeout 900 \
 /home/esjung/.venvs/flashvep-deepep-v020/bin/python -m poc_dispatch_combine_asymmetry.scripts.run \
 --model mllm --arms visual,text --sizes high,low,medium --repeats 10 --lane clean \
 --output "$ASYMMETRY_OUTPUT" --control "$ASYMMETRY_CONTROL"
```

For the standalone model, use a DIFFERENT new output directory and `--model llm --arms text`; all other primary settings remain identical. For bounded route capture use `--lane internal --repeats 3`. Routes are downloaded after the measured model forward, with hidden captured only at layers0/24/47. Internal component time is not substituted into clean time.

For the common-envelope overhead check, a fresh output and `ASYMMETRY_ENVELOPE=1`, `--sizes high --lane overhead --repeats 10` were used. Both bypass/clean have identical envelope markers/guard. Do not enable this for the ordinary primary clean baseline.

For physical replay, remove hook-path injection, set `ASYMMETRY_HOOK=0`, use a fresh output directory and matching `NATIVE_LP_OUTPUT`:

```bash
CUDA_VISIBLE_DEVICES=0,1,4,5 ASYMMETRY_HOOK=0 \
PYTHONPATH=/home/esjung/MLLM-EP-dispatch-asymmetry \
NATIVE_LP_OUTPUT=/home/esjung/MLLM-EP-dispatch-asymmetry/poc_dispatch_combine_asymmetry/artifacts/replay_NEW \
/home/esjung/.venvs/flashvep-deepep-v020/bin/python -m poc_native_lp_ep4.scripts.gpu_session --timeout 420 \
 /home/esjung/.venvs/flashvep-deepep-v020/bin/python -m torch.distributed.run --standalone --nproc-per-node 4 \
 -m poc_dispatch_combine_asymmetry.scripts.replay \
 --captures poc_dispatch_combine_asymmetry/artifacts/runs/mllm_internal_v3/captures \
 --output poc_dispatch_combine_asymmetry/artifacts/replay_NEW --repeats 10
```

Replay requires the LOCAL bounded hidden captures or newly generated internal captures. Published compact routes omit hidden vectors intentionally. It executes real DeepEP dispatch/combine but not expert GEMM; it is not native model latency or output-quality validation.

Nsight actual flags/return codes are recorded in `artifacts/nsight/ledger.json`: CUDA+NVTX, CPU sampling/context-switch off, cudaProfilerApi capture only. SQLite-to-kernel classification is `scripts/classify_nsight.py`; raw `.nsys-rep`/SQLite remain local, not required for CPU headline reaggregation.

## CPU-only verification and plots

```bash
cd /home/esjung/MLLM-EP-dispatch-asymmetry
CUDA_VISIBLE_DEVICES= /home/esjung/.venvs/flashvep-deepep-v020/bin/python -m unittest discover -s poc_dispatch_combine_asymmetry/tests -v
CUDA_VISIBLE_DEVICES= /home/esjung/.venvs/flashvep-deepep-v020/bin/python -m poc_dispatch_combine_asymmetry.scripts.verify_bundle
CUDA_VISIBLE_DEVICES= /home/esjung/anaconda3/bin/python -m poc_dispatch_combine_asymmetry.scripts.figures
```

Headlines can also be regenerated from local raw runs with `scripts/analyze.py RUN... --output NEW.json`; accounting with `scripts/account_captures.py RUN... --output NEW.json`. A fresh clone instead uses the compressed raw verifier, with no dependence on the absolute local run paths embedded in old metadata. `scripts/finalize.py`/`publish_evidence.py` are local raw-run export steps, not fresh-clone commands: they require the deliberately unshipped bounded captures and local raw run directories.

The five requested plots are in `figures/`; the internal-composite plot has an explicit UNVALIDATED/perturbed warning. Never add or compare timestamps from different GPU clocks.

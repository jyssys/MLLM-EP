# Reproduction

## Scope and resource rules

Use physical GPU4 for this experiment. GPUs5–7 retain their existing burners.
Do not resume native-vLLM PP work or use GPUs0–3.
All implementation paths below are separate from the paused PP repository.

Model: base `ByteDance/Ouro-1.4B`, revision `574fa66cb8bf5abdc979642d01cf2b79b16bfab1`.
RLT: `4a350344027960a48384626aa646f1765fda6d4f`.
CDB source reference: `abdfae0adbdd12be8c6123ccb34eee98a38afb39`.
The model SHA256 and complete source hashes appear in `environment_initial.json` and the final environment record.

## Source and environment

Create an isolated RLT checkout. Apply the small patch only to that checkout.

```bash
git clone https://github.com/InferMatrixLab/vllm-rlt /home/esjung/external/ouro-chunk-loop-prefill-rlt
git -C /home/esjung/external/ouro-chunk-loop-prefill-rlt checkout --detach 4a350344027960a48384626aa646f1765fda6d4f
git -C /home/esjung/external/ouro-chunk-loop-prefill-rlt apply --check /home/esjung/MLLM-EP-github/poc_flashvep/ouro_chunk_loop_prefill/patches/rlt_grouped_prefill.patch
git -C /home/esjung/external/ouro-chunk-loop-prefill-rlt apply /home/esjung/MLLM-EP-github/poc_flashvep/ouro_chunk_loop_prefill/patches/rlt_grouped_prefill.patch
```

Do not execute clone or patch commands over an existing modified checkout.
Check its revision and diff instead.

The PoC environment is `/home/esjung/.venvs/ouro-chunk-loop-prefill`.
It has a private package overlay and reads the existing native runtime packages through `shared_native_runtime.pth`.
That file contains one line:

```text
/home/esjung/.venvs/cp-composition-vllm026/lib/python3.14/site-packages
```

The original environment remains unchanged.
The overlay uses Python3.14.6, Torch2.11.0+cu129, Triton3.6.0, and Transformers5.16.1.
Its added packages include pytest9.1.1, FlashAttention4 4.0.0b30, CUTLASS DSL4.8.1, and TVM FFI0.1.15.post0.
The final environment record lists the installed versions.
These overlay versions conflict with inherited vLLM package requirements.
Do not run vLLM in this RLT-only overlay. This experiment does not change or execute native vLLM.

```bash
/home/esjung/.venvs/cp-composition-vllm026/bin/python -m venv /home/esjung/.venvs/ouro-chunk-loop-prefill
```

Create the stated `.pth` file in that environment's Python3.14 site-packages before installing the small overlay dependencies.
Then install the recorded versions into the new environment only.
No attention kernel source compilation or source rewrite is part of the G2 patch.

The environment used these explicit overlay pins:

```bash
/home/esjung/.venvs/ouro-chunk-loop-prefill/bin/python -m pip install \
  pytest==9.1.1 flash-attn-4==4.0.0b30 nvidia-cutlass-dsl==4.8.1 \
  apache-tvm-ffi==0.1.15.post0 quack-kernels==0.6.5 matplotlib==3.11.2 ruff==0.17.1
```

The inherited native environment supplies Torch, Triton, Transformers, NumPy, and Safetensors.
See `environment.json` for their exact installed versions.
The command above is an environment reconstruction step, not a reason to modify the native vLLM environment.

## CPU tests

Run the stock test before applying the patch to reproduce the original baseline check.
After applying the patch, run the stock suite and the new grouped tests.

```bash
cd /home/esjung/external/ouro-chunk-loop-prefill-rlt
CUDA_VISIBLE_DEVICES='' /home/esjung/.venvs/ouro-chunk-loop-prefill/bin/python -m pytest -q -m 'not gpu'
cd /home/esjung/MLLM-EP-github
CUDA_VISIBLE_DEVICES='' PYTHONPATH=/home/esjung/external/ouro-chunk-loop-prefill-rlt \
  /home/esjung/.venvs/ouro-chunk-loop-prefill/bin/python -m pytest -q poc_flashvep/ouro_chunk_loop_prefill/test_grouped.py
```

The original source suite reported800 passes,34 skips, and322 GPU deselections.
The separate grouped test file reported eight passes.
Its random tiny models are unit-test fixtures, not performance or quality evidence.

## GPU lease

Inspect ownership first. Stop only the recognized GPU4 burner through the existing lease helper.

```bash
cd /home/esjung/MLLM-EP-ouro-request-pp
/home/esjung/anaconda3/bin/python -m poc_flashvep.ouro_8arm.lease status --gpus 4,5,6,7
/home/esjung/anaconda3/bin/python -m poc_flashvep.ouro_8arm.lease stop --gpus 4
```

This helper is only a resource controller. It does not resume the PP experiment.
It refuses to stop an unrelated process.
Do not invoke `run_utilize.sh` with an unchecked default mask.
The audited default still lists2,3,6,7. The lease helper explicitly restores the requested allowed GPUs.

## Frozen inputs and validation

The committed workload manifest is the authority for request IDs, input IDs, sampling, and SLO thresholds.
Do not rerun `--freeze` over it. That command refuses an existing manifest.
The prompts contain natural problem text, not random token IDs.
The long contexts are constructed passages, not original conversational requests.

```bash
cd /home/esjung/MLLM-EP-github
CUDA_VISIBLE_DEVICES=4 PYTHONPATH=/home/esjung/external/ouro-chunk-loop-prefill-rlt \
  /home/esjung/.venvs/ouro-chunk-loop-prefill/bin/python -m poc_flashvep.ouro_chunk_loop_prefill.smoke \
  --label fresh_triton_correctness --lengths 32,128,512,2048
CUDA_VISIBLE_DEVICES=4 OURO_PREFILL_BACKEND=flash_attn_4 \
  PYTHONPATH=/home/esjung/external/ouro-chunk-loop-prefill-rlt \
  /home/esjung/.venvs/ouro-chunk-loop-prefill/bin/python -m poc_flashvep.ouro_chunk_loop_prefill.smoke \
  --label fresh_fa4_correctness --lengths 32,128,512,2048
```

Use fresh labels. Every run directory refuses overwrite.

## Static discovery grid

The actual grid used trial1, the four chunk sizes, both prompt bins, both concurrency values, and all four policies.
Triton grid label: `static_grid_v1`. FA4 grid label: `fa4_grid_v1`.
Run one backend at a time.

```bash
CUDA_VISIBLE_DEVICES=4 OURO_PREFILL_BACKEND=flash_attn_4 \
  PYTHONPATH=/home/esjung/external/ouro-chunk-loop-prefill-rlt \
  /home/esjung/.venvs/ouro-chunk-loop-prefill/bin/python -u -m poc_flashvep.ouro_chunk_loop_prefill.bench \
  --label fresh_fa4_grid --modes atomic,split,mixed,grouped --chunks 128,256,512,1024 \
  --lengths 2048,8192 --concurrencies 4,8 --trials 1
```

Replace the backend with `triton` for the secondary grid.
Discovery-grid cold effects do not qualify a warm serving claim.
The initial stock calibration used a smaller pool and remains separate.

## Corrected confirmation and controls

The initial confirmation showed one-sided first-use JIT cost. Its partial data remain under `fa4_confirm_s0`.
That initial lane is not a qualified comparison.
The corrected contract is `confirmation_contract_warm_v2.json`.
It uses the same requests, thresholds, and policy menu.
Every configuration receives the same independent calibration cohort before the measured held-out cohorts.
Warm-up results remain in raw data with `warmup=true`.
Any later request-bound graph miss remains inside measured E2E.

```bash
bash poc_flashvep/ouro_chunk_loop_prefill/confirm.sh
```

The script executes three fresh processes and ten paired held-out cohorts.
It also runs observer/graph controls, prefill/decode controls, one bounded Nsight capture, and an8K KV check.
It restores the GPU4 burner through an EXIT trap.
Existing run directories prevent accidental repetition. Use a new label set for a new campaign.

## Analysis, plots, and raw archives

```bash
/home/esjung/.venvs/ouro-chunk-loop-prefill/bin/python -m poc_flashvep.ouro_chunk_loop_prefill.analyze
/home/esjung/.venvs/ouro-chunk-loop-prefill/bin/python -m poc_flashvep.ouro_chunk_loop_prefill.confirmation_analysis
/home/esjung/.venvs/ouro-chunk-loop-prefill/bin/python -m poc_flashvep.ouro_chunk_loop_prefill.diagnostics
/home/esjung/.venvs/ouro-chunk-loop-prefill/bin/python -m poc_flashvep.ouro_chunk_loop_prefill.report_tables
/home/esjung/.venvs/ouro-chunk-loop-prefill/bin/python -m poc_flashvep.ouro_chunk_loop_prefill.figures
/home/esjung/.venvs/ouro-chunk-loop-prefill/bin/python -m poc_flashvep.ouro_chunk_loop_prefill.provenance
/home/esjung/.venvs/ouro-chunk-loop-prefill/bin/python -m poc_flashvep.ouro_chunk_loop_prefill.archive
/home/esjung/.venvs/ouro-chunk-loop-prefill/bin/python -m poc_flashvep.ouro_chunk_loop_prefill.validate_artifacts
```

The raw JSON and JSONL files remain intact on the server.
Git contains deterministic, lossless gzip copies and `archive_manifest.json` with SHA256 checks.
Restore them after cloning:

```bash
/home/esjung/.venvs/ouro-chunk-loop-prefill/bin/python -m poc_flashvep.ouro_chunk_loop_prefill.archive --restore
```

Restoration checks existing files and refuses conflicting content.
Heavy timeline time, model-call event intervals, and clean client latency remain separate evidence classes.
The clock is an in-process client clock. It does not include an HTTP network frontend.

## Nsight physical timeline

The bounded capture uses the installed Nsight Systems 2024.6.2.
The binary report and SQLite remain server-local. Git retains sanitized complete event CSV exports instead.
The source files remain unchanged and their SHA256 hashes appear in `nsys_summary.json`.

```bash
nsys export --type sqlite \
  --output poc_flashvep/reports/ouro_chunk_loop_prefill_joint_discovery/raw/nsys_grouped.sqlite \
  poc_flashvep/reports/ouro_chunk_loop_prefill_joint_discovery/raw/nsys_grouped.nsys-rep
/home/esjung/.venvs/ouro-chunk-loop-prefill/bin/python -m poc_flashvep.ouro_chunk_loop_prefill.nsys_analysis
```

Do not export over an existing database. The completed server already contains it.
After cloning, archive restoration supplies the sanitized kernel/copy/runtime CSV files.
It does not supply the private binary report or create a new GPU measurement.
Re-run archive creation after new timeline exports only when intentionally creating a new artifact set.

## Result interpretation

The final status is `CONDITIONAL_FIXED_QUANTUM_ONLY`.
The registered TTFT/ITL-p95 gate did not justify Adaptive Joint or multi-GPU expansion.
Both phases are NOT_RUN, not successful unmeasured extensions.
The capped natural-serving outputs do not establish full benchmark accuracy.

The first Atomic decode-only concurrency1 control also incurred first-use compilation.
Do not interpret its cold difference as a candidate speedup.

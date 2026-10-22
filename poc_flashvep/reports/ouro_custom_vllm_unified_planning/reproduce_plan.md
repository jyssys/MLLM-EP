# Reproduction and future execution plan

## Scope

The commands in the audit section are read-only.
The serving commands are **future Phase 1 examples, NOT_RUN**.
No unified-engine implementation exists in this planning commit.
Do not treat a proposed harness name as an executable script.

## Repeat the source audit

Run these commands from their stated directories.

```bash
git -C /home/esjung/MLLM-EP-github branch --show-current
git -C /home/esjung/external/ouro-unified-planning-vllm026 rev-parse HEAD
git -C /home/esjung/external/ouro-unified-planning-rlt rev-parse HEAD
git -C /home/esjung/external/decode-first-cdb-upstream rev-parse HEAD
git -C /home/esjung/MLLM-EP-ouro-request-pp rev-parse HEAD
```

Expected revisions appear in [dependency_lock.md](dependency_lock.md).
Check the installed native model against the audit source:

```bash
sha256sum /home/esjung/external/ouro-unified-planning-vllm026/vllm/model_executor/models/ouro.py
sha256sum /home/esjung/.venvs/cp-composition-vllm026/lib/python3.14/site-packages/vllm/model_executor/models/ouro.py
```

Both matched `cf0792c6c8024ded45dba10553b5f0d24b7b6542dd8f8f0e0e6899807ec0c4ff` during the audit.
Verify upstream changes only in a new audit clone. Do not update an experiment pin silently.

Useful source searches:

```bash
rg -n 'class Ouro|early_exit_gate|current_ut|make_layers|unique_layer_idx' /home/esjung/external/ouro-unified-planning-vllm026/vllm/model_executor/models/ouro.py
rg -n '_pp_send_work|irecv_tensor_dict|isend_tensor_dict' /home/esjung/external/ouro-unified-planning-vllm026/vllm/v1/worker/gpu_worker.py
rg -n 'execute_model_state|def execute_model|def get_kv_cache_spec' /home/esjung/external/ouro-unified-planning-vllm026/vllm/v1/worker/gpu_model_runner.py
rg -n 'def should_exit|sigmoid|exit_cdf' /home/esjung/external/decode-first-cdb-upstream/src/looped_cdb/continuous_depth_batching/exit_policy.py
rg -n 'selected.float|def finalize|def finalize_many' /home/esjung/external/ouro-unified-planning-rlt/vllm_rlt/worker/model_runner.py
```

Preserve the old study. Do not rerun its campaign launchers as a shortcut.
Read its `native_quality.py` for the former API setup, not as a new client-latency harness.
Its historical service clocks excluded or treated preparation differently.

## GPU lease and burner rules

Planning does not need a GPU lease change.
Before a future approved execution, inspect allowed devices and ownership:

```bash
nvidia-smi -i 4,5,6,7 --query-gpu=index,uuid,name,memory.total,memory.used,utilization.gpu --format=csv
nvidia-smi topo -m
bash /home/esjung/vllm-ep/burner_4567.sh status
```

Stop only recognized user-owned burners when an approved test needs those devices.
Do not stop an unrelated process on any GPU.
The safe wrapper explicitly targets physical4–7:

```bash
bash /home/esjung/vllm-ep/burner_4567.sh stop
```

After the approved test, restore burners only on idle permitted GPUs:

```bash
bash /home/esjung/vllm-ep/burner_4567.sh start
bash /home/esjung/vllm-ep/burner_4567.sh status
```

Do not invoke an unscoped burner default.
The audited original `run_utilize.sh` default still lists GPUs2,3,6,7.
If that entry point is necessary, set `VLLM_UTILIZE_GPUS=4,5,6,7` explicitly.
The wrapper is preferable because it uses the existing ownership-aware lease controller.

## Minimal Phase 1 implementation checklist

Create a new approved implementation branch and isolated native environment.
Keep the read-only audit clones and prior study unchanged.

Implement a shared harness with these tasks:

1. Export package metadata, extension identities, selected backend, graph policy, UUID mapping, and model/tokenizer hashes.
2. Load the frozen prompt IDs without changing the prompt recipe.
3. Run bounded TP1/TP4 probes and record first numerical divergence.
4. Run 24 natural requests through native HTTP serving at C1/C4.
5. Measure client arrival, token chunks, final completion, failures, and raw output.
6. Run the 114-question pilot after smoke correctness and parsing succeed.
7. Produce G1 qualification before any PP or Dynamic engine patch.

The future harness must not monkey-patch native logits or return recorded outputs in natural generation.
Inspection hooks may copy bounded intermediate tensors in the diagnostic lane only.
Do not place tensor snapshots or per-layer synchronizations in the clean timing lane.

## Future native server examples

These commands use audited native argument names.
Validate them with the isolated environment's `vllm serve --help` before execution.
Graph and backend selection require an archived effective configuration, not an assumption from default flags.
The fixed-pool diagnostic lane preserves 16GiB aggregate KV reserve across Single and TP4.
It is not a maximum-capacity claim.

### Single, physical GPU4

```bash
CUDA_VISIBLE_DEVICES=4 \
/home/esjung/.venvs/cp-composition-vllm026/bin/vllm serve \
  /home/esjung/.cache/huggingface/hub/models--ByteDance--Ouro-1.4B/snapshots/574fa66cb8bf5abdc979642d01cf2b79b16bfab1 \
  --served-model-name ouro-base-r4 \
  --trust-remote-code \
  --dtype bfloat16 \
  --tensor-parallel-size 1 \
  --pipeline-parallel-size 1 \
  --max-model-len 8192 \
  --max-num-seqs 4 \
  --max-num-batched-tokens 8192 \
  --kv-cache-memory-bytes 17179869184 \
  --hf-overrides '{"total_ut_steps":4,"early_exit_threshold":1.0}' \
  --no-enable-prefix-caching \
  --no-enable-chunked-prefill \
  --no-async-scheduling \
  --host 127.0.0.1 --port 18041
```

### TP4, physical GPUs4–7

```bash
CUDA_VISIBLE_DEVICES=4,5,6,7 \
/home/esjung/.venvs/cp-composition-vllm026/bin/vllm serve \
  /home/esjung/.cache/huggingface/hub/models--ByteDance--Ouro-1.4B/snapshots/574fa66cb8bf5abdc979642d01cf2b79b16bfab1 \
  --served-model-name ouro-base-r4 \
  --trust-remote-code \
  --dtype bfloat16 \
  --tensor-parallel-size 4 \
  --pipeline-parallel-size 1 \
  --max-model-len 8192 \
  --max-num-seqs 4 \
  --max-num-batched-tokens 8192 \
  --kv-cache-memory-bytes 4294967296 \
  --hf-overrides '{"total_ut_steps":4,"early_exit_threshold":1.0}' \
  --no-enable-prefix-caching \
  --no-enable-chunked-prefill \
  --no-async-scheduling \
  --host 127.0.0.1 --port 18041
```

Run these servers sequentially, not simultaneously.
The existing environment path is an observed reference. Prefer a verified isolated copy for future modifications.
No `--pipeline-parallel-size 4` Ouro example is provided because native Ouro does not implement the required PP contract.

The primary API request is a base-model completion, not a Thinking chat template.
The historical recipe uses three-shot CoT for GSM8K and zero-shot boxed-answer MATH.
Use `temperature=0`, `max_tokens=4096`, natural EOS, and each request's frozen stop strings.
Use the exact historical scorer implementation and its dependencies.
Implement streaming timestamp collection before claiming TTFT, ITL, or client E2E.

## Future output layout

Write future results under `poc_flashvep/reports/ouro_custom_vllm_unified_serving/` only after approval.
Use a fresh immutable run directory for every attempt.
Required files include:

- `manifest.json`, effective server configuration, dependency and dataset hashes.
- `requests.jsonl`, `tokens.jsonl`, `depths.jsonl`, and raw answer text.
- `serving_metrics.csv`, `quality.csv`, `slo_goodput.csv`, `memory.csv`, and `communication.csv`.
- Separate `clean/`, `diagnostic/`, `replay/`, `profiles/`, and `failures/` directories.
- `analysis.md`, `benchmark_summary.md`, `correctness.md`, and `reproduce.md`.

Keep failure logs and partial runs. Never overwrite a historical report or raw result.
Do not include model weights or cache pools in Git.

## Git preservation

This planning branch started from `f0dc8372ec0d1d2e10347c82e630b84430c5dc5f`.
The repository already contained extensive unrelated deletions and 68 staged additions.
The pre-existing cached-diff SHA256 was:
`7018be31bd8a517574711055a72afbd6ae90737e4a585578c98ec255dbf7e1d9`.

Commit only this new planning directory with a path-limited commit.
Check every committed path and verify the pre-existing staged fingerprint afterward.
Do not reset, clean, stash, restore deleted files, or force-push shared history.
The final user instruction explicitly authorizes pushing this new planning branch.

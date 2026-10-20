# Reproduction

## Environment

Use `/home/esjung/.venvs/llada2-mini-reference/bin/python`.
The environment name does not identify the loaded model.
The model is the pinned Base Ouro-1.4B checkpoint, not LLaDA or Ouro-Thinking.
The model revision is `574fa66cb8bf5abdc979642d01cf2b79b16bfab1`.
The worktree is `/home/esjung/MLLM-EP-ouro-request-pp`.

Run commands from the repository root.
Use only physical GPUs 4,5,6,7.
The worker's logical device indices are 0,1,2,3.
The UUID assertions reject a different mapping.

## Commands

Freeze the contract and input IDs before a fresh campaign:

```bash
env CUDA_VISIBLE_DEVICES='' \
  /home/esjung/.venvs/llada2-mini-reference/bin/python \
  -m poc_flashvep.ouro_request_pp.prepare
```

Do not repeat preparation in the published result directory.
Keep the existing frozen files and raw evidence.

Check ownership of every existing GPU process.
The following helper stops only identified project burner sessions:

```bash
env CUDA_VISIBLE_DEVICES='' \
  /home/esjung/.venvs/llada2-mini-reference/bin/python \
  -m poc_flashvep.ouro_request_pp.gpu_lease stop
```

Run CPU route tests:

```bash
env CUDA_VISIBLE_DEVICES='' \
  /home/esjung/.venvs/llada2-mini-reference/bin/python \
  -m unittest poc_flashvep.ouro_request_pp.test_routes -v
```

Run the fixed-depth graph correctness gate:

```bash
env CUDA_VISIBLE_DEVICES=4,5,6,7 \
  /home/esjung/.venvs/llada2-mini-reference/bin/python -u \
  -m poc_flashvep.ouro_request_pp.campaign correctness_graph
```

Run longer-prompt correctness checks and overlap probes:

```bash
env CUDA_VISIBLE_DEVICES=4,5,6,7 \
  /home/esjung/.venvs/llada2-mini-reference/bin/python -u \
  -m poc_flashvep.ouro_request_pp.campaign extended

env CUDA_VISIBLE_DEVICES=4,5,6,7 \
  /home/esjung/.venvs/llada2-mini-reference/bin/python -u \
  -m poc_flashvep.ouro_request_pp.campaign probe
```

Run the primary matrix only after correctness passes:

```bash
env CUDA_VISIBLE_DEVICES=4,5,6,7 \
  /home/esjung/.venvs/llada2-mini-reference/bin/python -u \
  -m poc_flashvep.ouro_request_pp.campaign primary
```

The launcher refuses to overwrite an existing named log or raw run directory.
Use a separate result directory for a new campaign.
The command ledger records the actual argv, source hashes, and completion status.
Raw JSON results use lossless gzip compression.

Analyze the results on CPUs:

```bash
env CUDA_VISIBLE_DEVICES='' OPENBLAS_NUM_THREADS=1 \
  /home/esjung/.venvs/llada2-mini-reference/bin/python \
  -m poc_flashvep.ouro_request_pp.analyze
```

Run the additional validation and common backend controls after the primary campaign:

```bash
env CUDA_VISIBLE_DEVICES=4,5,6,7 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  /home/esjung/.venvs/llada2-mini-reference/bin/python -u \
  -m poc_flashvep.ouro_request_pp.post_campaign validation

env CUDA_VISIBLE_DEVICES=4,5,6,7 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  /home/esjung/.venvs/llada2-mini-reference/bin/python -u \
  -m poc_flashvep.ouro_request_pp.post_campaign controls

env CUDA_VISIBLE_DEVICES=4,5,6,7 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  /home/esjung/.venvs/llada2-mini-reference/bin/python -u \
  -m poc_flashvep.ouro_request_pp.receive_control_campaign
```

The last command first checks pooled-buffer correctness.
It then measures pooled buffers alone and pooled buffers with constant RoPE preparation.
These controls do not replace the preserved primary measurements.

Record actual kernels and peer copies in a separate heavy lane:

```bash
env CUDA_VISIBLE_DEVICES=4,5,6,7 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  /home/esjung/.venvs/llada2-mini-reference/bin/python -u \
  -m poc_flashvep.ouro_request_pp.profile_capture --depth 4 --concurrency 4

env CUDA_VISIBLE_DEVICES=4,5,6,7 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  /home/esjung/.venvs/llada2-mini-reference/bin/python -u \
  -m poc_flashvep.ouro_request_pp.profile_capture --depth 2 --concurrency 8
```

`profiles/*.nsys-rep` are the preserved raw profiler reports.
Nsight derives SQLite files from those reports. Recreate them with `nsys export --type=sqlite` if needed.
The checked-in summaries retain each export's SHA-256.
The parser accepts the local SQLite paths:

```bash
env CUDA_VISIBLE_DEVICES='' OPENBLAS_NUM_THREADS=1 \
  /home/esjung/.venvs/llada2-mini-reference/bin/python \
  -m poc_flashvep.ouro_request_pp.parse_profile \
  poc_flashvep/reports/ouro_request_pp_discovery/profiles/layer_pp4_R4_C4_v5.sqlite \
  poc_flashvep/reports/ouro_request_pp_discovery/profiles/loop_pp4_R4_C4_v5.sqlite \
  poc_flashvep/reports/ouro_request_pp_discovery/profiles/layer_pp4_R2_C8_v5.sqlite \
  poc_flashvep/reports/ouro_request_pp_discovery/profiles/loop_pp4_R2_C8_v5.sqlite
```

After parsing, run these CPU modules in order:

1. `poc_flashvep.ouro_request_pp.analyze`
2. `poc_flashvep.ouro_request_pp.trace_tables`
3. `poc_flashvep.ouro_request_pp.make_report`
4. `poc_flashvep.ouro_request_pp.final_analysis`

Run the published evidence tests:

```bash
env CUDA_VISIBLE_DEVICES='' OPENBLAS_NUM_THREADS=1 \
  /home/esjung/.venvs/llada2-mini-reference/bin/python -m unittest \
  poc_flashvep.ouro_request_pp.test_routes \
  poc_flashvep.ouro_request_pp.test_evidence -v
```

After all experiment processes exit, restore the burners:

```bash
env CUDA_VISIBLE_DEVICES=4,5,6,7 VLLM_UTILIZE_GPUS=4,5,6,7 \
  bash /home/esjung/vllm-ep/run_utilize.sh
```

Both GPU-mask overrides are necessary.
The script's server default can select a different GPU set.
The scoped restore helper also starts detached burners on the allowed free GPUs.

## Preserved implementation diagnostics

The original eager pipelines pass bitwise reference checks.
The first masked-capacity graph changes a greedy token despite small teacher-forced logit error.
That path is not the final runtime.
The exact-length graph initially omitted a retained position tensor.
The interleaved regression reproduced that lifetime error.
The runtime now retains every captured input tensor.
The same regression passes after that change.

The final runtime preserves the original SDPA length and prefill K/V strides.
It uses graph scratch only for local storage and execution.
The extra local copies remain inside the measured serving clock.
The early primary attempt stopped before a measured result completed.
Its command and nonzero completion status remain in the ledger.

Published logs distinguish the eager path, decode-only graph path, and final full-stage graph path.
Use final `v5` results for the primary comparison.
Do not combine these implementations into one timing sample.

`source_v5.tar.gz` preserves the source before optional transport and RoPE controls.
The final source keeps those controls behind explicit flags.
The default path retains the original primary behavior.

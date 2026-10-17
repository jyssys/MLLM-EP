# Reproduction

Run commands from the new worktree. Do not reset the original checkout.
The experiment permits physical GPUs 4, 5, 6, and 7 only.
The guard stops verified project burners on its requested subset and restores them after the job.
It refuses an unidentified competing context.

## Sources and environment

```bash
cd /home/esjung/MLLM-EP-async-recurrent-pp
/home/esjung/.venvs/llada2-mini-reference/bin/python -c 'from huggingface_hub import snapshot_download; print(snapshot_download("ByteDance/Ouro-1.4B-Thinking", revision="3aaa2224253a92ca45cf2e3d427c360e1ef9c93d"))'
```

The reference environment uses PyTorch 2.8.0+cu128 and Transformers 4.57.1.
The model uses its unmodified pinned custom code and SDPA.
Dataset preparation uses PyArrow 19.0.1 in an isolated dependency directory.
MATH scoring reuses the prior isolated Math-Verify 0.8.0 installation.
No shared environment requires a package replacement.

The saved specification is an exact copy of `poc_flashvep/reports/async_recurrent_pp_poc_spec.md` from the original checkout.
The originally requested destination did not exist before this experiment.
The contract records the source path and SHA-256.

## CPU checks

```bash
/home/esjung/.venvs/llada2-mini-reference/bin/python -m unittest discover -s poc_async_recurrent_pp/tests -v
```

Frozen manifests must not change after inference.
Preparation commands refuse an existing manifest or specification destination.
Use a new output directory for every GPU attempt. Preserve failed attempts and their console logs.

## Baseline certification

```bash
/home/esjung/.venvs/llada2-mini-reference/bin/python -m poc_async_recurrent_pp.scripts.gpu_session --gpus 4 --output outputs/baseline_v1 --timeout 5400 -- /home/esjung/.venvs/llada2-mini-reference/bin/python -m poc_async_recurrent_pp.scripts.baseline
```

## Controlled campaign

```bash
/home/esjung/.venvs/llada2-mini-reference/bin/python -m poc_async_recurrent_pp.scripts.campaign
```

The controller first waits for baseline certification and corrected native task sanity.
It requires the long exact-hook batch parity gate before the corrected pilot accuracy wave.
The primary pilot wave uses 16 questions per dataset and an output cap of 2,048 tokens.
Any truncation triggers a complete paired rerun at 8,192 tokens.
Pilot screening can reject a materially harmful approximation. It cannot establish quality GO.

## Analysis

```bash
/home/esjung/.venvs/llada2-mini-reference/bin/python -m poc_async_recurrent_pp.scripts.aggregate --wave poc_async_recurrent_pp/outputs/pilot_cap2048_v2
```

Use the same command for a completed higher-cap wave with its directory path.
The primary directories use the `v2` suffix. The rejected compacting driver's `v1` quality wave remains excluded.
The bootstrap resamples paired questions 10,000 times.
Layer, token, stage, and loop observations do not count as independent accuracy samples.

## Burner restoration

The shared launcher defaults to another GPU set. Always set both masks explicitly.
Do not launch another burner while a restored worker still initializes on the CPU.

```bash
CUDA_VISIBLE_DEVICES=4,5,6,7 VLLM_UTILIZE_GPUS=4,5,6,7 bash /home/esjung/vllm-ep/run_utilize.sh
```

This manual command applies only when no experiment or pending burner occupies the assigned GPUs.
The experiment guard normally performs restoration automatically.

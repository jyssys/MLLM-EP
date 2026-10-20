# Environment and provenance

The experiment uses only physical GPUs 4, 5, 6, and 7.
The launcher sets `CUDA_VISIBLE_DEVICES=4,5,6,7` for every GPU job.
TP asserts a four-rank NCCL world and checks each loaded GPU UUID.

| Physical GPU | Logical rank | UUID |
| --- | --- | --- |
| 4 | 0 | GPU-6076e2f2-5b63-3761-5586-56ceb7df8139 |
| 5 | 1 | GPU-a1a1cfcf-93a1-3544-9a5e-e58144b68730 |
| 6 | 2 | GPU-e3f3998e-0f1a-e94a-b97c-4abb0e8c2c28 |
| 7 | 3 | GPU-4cc26b88-19fc-1988-f9e0-17858aa7a99b |

All four devices are H100 80GB HBM3 GPUs. `nvidia-smi` reports 81,559 MiB per device.
Every permitted pair reports NV18. The driver is 570.211.01.
The raw topology and NVLink status are in `environment_preflight.json`.
The preflight also records jobs on other GPUs. This experiment did not signal those jobs.

| Runtime | Python | PyTorch | Transformers | CUDA build | NCCL |
| --- | --- | --- | --- | --- | --- |
| Custom TP and PP | 3.11 | 2.8.0+cu128 | 4.57.1 | 12.8 | 2.27.3 |
| Isolated native vLLM | 3.14 | 2.11.0+cu129 | 5.16.1 | 12.9 | 2.28.9 |

The custom interpreter is `/home/esjung/.venvs/llada2-mini-reference/bin/python`.
The external interpreter is `/home/esjung/.venvs/cp-composition-vllm026/bin/python`.
No package installation or replacement was necessary.
We set no NCCL algorithm or protocol override.
The custom operator trace identifies `ncclDevKernel_AllReduce_Sum_f32_RING_LL`.

The model and tokenizer use `ByteDance/Ouro-1.4B`, revision `574fa66cb8bf5abdc979642d01cf2b79b16bfab1`.
This is the base checkpoint, not the Thinking checkpoint.
The loaded model has 24 physical layers, R4, and 96 independent logical KV slots.
The hidden size is 2048. There are 16 query heads and 16 KV heads.
The head size is 128. The MLP size is 5632.
The model has 1,434,652,673 parameters. All loaded model tensors use BF16.
The weight-file SHA256 is `58872a72616c736595b8b7662079c5b12c5a162ec16eae94f21c348dfa9885af`.

The base code commit is `addd81ee64d040527aac2b27243a671c59d4e175`.
The branch is `flashvep/ouro-tp4-parallelism-discovery`.
`preflight.json` records all protected source and report hashes.
`commands/*.json` records the source hashes for each execution.
The preservation test checks those protected hashes.

The requested spec path was absent in this worktree.
The full spec existed in `/home/esjung/MLLM-EP-github/poc_flashvep/reports/`.
`spec_frozen.md` preserves that document. `preflight.json` records its SHA256.
This experiment did not modify the main repository.

## Process lifecycle incident

The native vLLM driver initially used `os._exit`, which left four idle worker processes resident.
The cleanup audit linked all four workers to this task's recorded process group and user ID.
Only that process group received a termination signal.
`excluded_runs.json` lists later measurements that could contain this memory-residency confounder.
Those files remain unchanged. We repeated the clean PP controls and profiles after the workers exited.
The launcher now refuses occupied GPUs and cleans its own surviving worker group.
The external driver now requests an engine shutdown.
The user's burner script restarted after GPU tests. See `burner_restore.json` and the final occupancy snapshot.

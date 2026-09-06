# Environment manifest

- Date/time zone: 2026-09-07, Asia/Seoul
- Repository: `/home/esjung/MLLM-EP-github`
- Branch: `flashvep/autonomous-moe-ep-night-discovery`
- Python environment: `/home/esjung/.venvs/flashvep-deepep-v020`
- vLLM: 0.20.0+cu129
- DeepEP: 1.2.1+73b6ea4
- torch: 2.11.0+cu129
- flashinfer-python: 0.6.8.post1
- NCCL observed at runtime: 2.28.9
- NVIDIA driver: 570.211.01
- Model checkpoint: Qwen3-VL-30B-A3B-Instruct snapshot
  `9c4b90e1e4ba969fd3b5378b57d966d725f1b86c`
- Model config: 48 MoE layers, 128 routed experts, top-8, hidden size
  2048, expert intermediate size 768.
- Runtime: BF16, TP2/DP2/EP4, DeepEP high-throughput,
  `DeepEPHTAll2AllManager`, `DeepEPHTPrepareAndFinalize`, eager,
  async scheduling auto-enabled, prefix cache disabled, DBO disabled.

## Allowed physical GPUs

| physical index | UUID | PCI bus | GPU | memory |
|---:|---|---|---|---:|
| 1 | GPU-a77f3471-67d4-20b0-9fab-e502d4de5adb | 08:00.0 | H100 HBM3 | 81559 MiB |
| 2 | GPU-24200107-8a7f-de46-1bc8-b81f8d3af13e | 0D:00.0 | H100 HBM3 | 81559 MiB |
| 3 | GPU-17488c15-2d4c-5d9e-d503-29b0d959a8a8 | 12:00.0 | H100 HBM3 | 81559 MiB |
| 4 | GPU-6076e2f2-5b63-3761-5586-56ceb7df8139 | 17:00.0 | H100 HBM3 | 81559 MiB |

All launches explicitly set `CUDA_VISIBLE_DEVICES=1,2,3,4`. No workload or
termination command targets physical GPUs 0/5/6/7.

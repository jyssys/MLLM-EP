# GPU topology audit

The run used only physical GPUs `2,3,6,7`. Each is an NVIDIA H100 80GB HBM3
(81,559 MiB reported). UUIDs are:

| Physical GPU | UUID |
|---:|---|
| 2 | `GPU-24200107-8a7f-de46-1bc8-b81f8d3af13e` |
| 3 | `GPU-17488c15-2d4c-5d9e-d503-29b0d959a8a8` |
| 6 | `GPU-e3f3998e-0f1a-e94a-b97c-4abb0e8c2c28` |
| 7 | `GPU-4cc26b88-19fc-1988-f9e0-17858aa7a99b` |

`nvidia-smi topo -m` reports `NV18` between every pair in the selected set;
all GPUs are in NUMA node 0. Foreign processes were present only on GPUs
0,1,4,5 and were not touched. Selected GPUs were empty before clean runs.

Runtime: vLLM `0.20.0+cu129`, DeepEP `1.2.1+73b6ea4`, PyTorch
`2.11.0+cu129`, CUDA `12.9`, Transformers `5.14.1`, Python `3.12.13`.
The physical path was BF16 `TP2-DP2-EP4-PP1`, 32 linearly placed experts per
rank, `deepep_high_throughput`, DBO/EPLB/prefix cache off, eager mode.

The complete command output is in
`artifacts/gpu_audit/environment.json`.

# Environment

## Contract and scope

The authoritative source is `recurrent_memory_compute_asymmetry_discovery_spec.md` in the original repository.
`spec.md` is an unchanged copy.
Its SHA256 is `cc4323fd201ed028d2f82383b1a928d801bd1f1ab92e29f6a38b6486eef797d6`.

This study profiles exact fixed R4 inference.
It does not implement a CP engine or a PP engine.
It does not execute the closed stale-state or prediction policies.

## Hardware

All assigned GPUs are NVIDIA H100 80GB HBM3 devices.
The driver reports approximately 81,559 MiB per GPU.
`nvidia-smi topo -m` reports NV18 between every assigned pair.
The raw topology is in `outputs/baseline_v1/environment.json`.

| Physical GPU | Local CUDA rank with the four-GPU mask | UUID suffix, excluding `GPU-` |
|---|---:|---|
| 4 | 0 | 6076e2f2-5b63-3761-5586-56ceb7df8139 |
| 5 | 1 | a1a1cfcf-93a1-3544-9a5e-e58144b68730 |
| 6 | 2 | e3f3998e-0f1a-e94a-b97c-4abb0e8c2c28 |
| 7 | 3 | 4cc26b88-19fc-1988-f9e0-17858aa7a99b |

The four-GPU mask is `CUDA_VISIBLE_DEVICES=4,5,6,7`.
A single-GPU worker uses one permitted physical index and local CUDA rank 0.
The TP2 control uses physical GPUs 4 and 5.

The guard records occupancy before each run.
It stops only verified utilization workers on assigned GPUs.
It refuses an unknown GPU context.
It restores utilization workers after success or failure.
It does not signal processes on GPUs 0–3.

## Model

| Field | Value |
|---|---|
| Checkpoint | ByteDance/Ouro-1.4B-Thinking |
| Revision | `3aaa2224253a92ca45cf2e3d427c360e1ef9c93d` |
| Precision | BF16 |
| Physical layers | 24, shared across four passes |
| Logical executions per forward | 96 |
| Hidden size | 2,048 |
| Query / KV heads | 16 / 16 |
| Head dimension | 128 |
| FFN intermediate size | 5,632 |
| Vocabulary | 49,152 |
| Configured context maximum | 65,536 |
| Training context | 4,096 |
| Evaluated hardware contexts | 1,024 / 4,096 / 8,192 / 16,384 |
| Unique parameter storage | 2,869,305,346 bytes, or 2.672249 GiB |
| Exact cache slots | 96 key tensors and 96 value tensors |

Long-context measurements characterize hardware shapes.
They do not establish long-context task accuracy.
There is no MoE router or EP All-to-All in this model.

## Pinned environments

| Lane | Python path | Torch / CUDA | Transformers | Runtime |
|---|---|---|---|---|
| Exact reference | `/home/esjung/.venvs/llada2-mini-reference/bin/python` | 2.8.0+cu128 / 12.8 | 4.57.1 | Official HF code, SDPA, eager |
| Optimized control | `/home/esjung/.venvs/cp-composition-vllm026/bin/python` | 2.11.0+cu129 / 12.9 | 5.16.1 | vLLM 0.26.0, FlashAttention3, CUDA Graphs |

The reference environment has NCCL 2.27.3.
The reference interpreter name does not identify the loaded model.
Every model probe loads Ouro weights.
No package or driver upgrade forms part of this study.

Native controls use TP1, TP2, or TP4 with DP1 / PP1 / CP1.
They disable prefix caching, chunked prefill, asynchronous scheduling, and expert parallelism.
They reserve a 32 GiB KV pool per rank.
They use capture sizes 1, 2, 4, and 8.
The actual ownership records include head counts, tensor storage, and device UUIDs.

## Native startup diagnosis

The first TP4 startup failed inside custom AllReduce initialization.
The guard selected `expandable_segments:True` for the HF reference.
That allocator did not provide a CUDA IPC handle for the test allocation.

| Controlled 16 MiB IPC probe | `cudaIpcGetMemHandle` return code |
|---|---:|
| Expandable allocator | 1, invalid argument |
| Default `cudaMalloc` allocator | 0, success |

The native launcher now removes allocator overrides before it imports vLLM.
This change affects only that launcher process.
It does not change model arithmetic or shared configuration.
The next startup succeeded.
That run then exposed a missing request-metrics option.
The launcher now sets `disable_log_stats=False`.
All failed logs remain in `native_TP4_v1` and `native_TP4_v2`.

## Coverage and resource limits

The reference sweep completed 295 clean trials across 15 context/batch cells.
Native TP1/2/4 each completed 160 clean hardware trials.
The reference sweep uses eight frozen calibration question IDs.
It does not use an untouched confirmation cohort.

The 16K/B8 cell was a capacity-safety skip, not an observed OOM.
Its approximately 96 GiB logical KV allocation exceeded available HBM before workspace.
Hardware counters failed with `ERR_NVGPUCTRPERM`.
Actual HBM traffic, L2 hit rate, and SM occupancy remain `UNAVAILABLE`.
The study did not change profiling permissions.

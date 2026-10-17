# Environment and physical device scope

## Devices

The experiment permits physical GPUs 4, 5, 6, and 7 only.
Each GPU is an NVIDIA H100 80GB HBM3 with 81,559 MiB reported memory.
The driver version is 570.211.01. Every pair in the assigned set reports NV18 connectivity.
The complete topology matrix appears in the baseline environment artifact.
No experiment changes GPU clocks or power limits.

| Physical GPU | UUID | Numerical-evaluation role | Logical CUDA device in worker |
|---|---|---|---|
| 4 | GPU-6076e2f2-5b63-3761-5586-56ceb7df8139 | Exact R4 | cuda:0 |
| 5 | GPU-a1a1cfcf-93a1-3544-9a5e-e58144b68730 | W1 stale | cuda:0 |
| 6 | GPU-e3f3998e-0f1a-e94a-b97c-4abb0e8c2c28 | W1 delta | cuda:0 |
| 7 | GPU-4cc26b88-19fc-1988-f9e0-17858aa7a99b | W0 initializer + delta | cuda:0 |

The parent quality job uses `CUDA_VISIBLE_DEVICES=4,5,6,7`.
Each independent accuracy worker receives a singleton physical mask.
Thus each worker's logical device zero maps to its assigned physical GPU.
These workers are independent full-model replicas. They are not a measured PP4 or TP4 architecture.

The proposed PP4 mapping remains conditional: logical ranks 0–3 would own physical GPUs 4–7 and six layers each.
No rank-mapping statement alone proves that this pipeline executed.
Distributed architecture measurements require the separate quality gate.

## Software and model

| Component | Actual setting |
|---|---|
| Model | ByteDance/Ouro-1.4B-Thinking |
| Checkpoint revision | 3aaa2224253a92ca45cf2e3d427c360e1ef9c93d |
| Weight SHA-256 | 6730581fd366f6d1689728c61637520228385cdfc17c0b24f19aa840b00565fb |
| Model-source SHA-256 | cb8c980b016ae8ae35b6c1193633b9b4496e89993e3a0e8c7853324f01904b30 |
| PyTorch | 2.8.0+cu128 |
| Transformers | 4.57.1 |
| CUDA build | 12.8 |
| NCCL build version | 2.27.3 |
| Attention | Official SDPA path |
| Precision | BF16 model weights and computation |
| Recurrent depth | Fixed R4, explicit output index 3 |
| Cache | 96 independent depth/layer slots |
| Generation | Official `model.generate()`, greedy, EOS enabled |
| Graphs | Eager reference execution, no custom CUDA Graph |
| Model changes | No weight or official-source patch |

The NCCL entry identifies the installed build. It is not evidence of an executed collective.
The functional quality lanes use no distributed model collective.
Reference-stage source and tokenizer hashes appear in `outputs/baseline_v1/environment.json`.
Later run contracts record expandable CUDA allocator segments to limit cache-allocation fragmentation.
This allocator setting does not change the model or prediction rule.

## Process policy

The guard stops only verified project burner workers on the requested physical mask.
It refuses an unidentified competing GPU context.
The guard restores burners after success, failure, or interruption.
The corrected quality controller also restores a finished arm's GPU while other arms continue.
Pending-launcher checks prevent a second burner during CPU-side initialization.

All burner launches set both `CUDA_VISIBLE_DEVICES` and `VLLM_UTILIZE_GPUS` explicitly.
The shared script's default mask is not the experiment mask.
No other user's GPU process receives a termination signal.

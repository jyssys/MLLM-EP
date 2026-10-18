# Environment

The physical GPUs are 4, 5, 6, and 7. Each is an NVIDIA H100 with 81,559 MiB reported HBM.
The recorded topology reports NV18 links between the allowed GPU pairs.
The topology record is `artifacts/environment_initial.json`.
We did not change GPU clocks or operate another user's GPU processes.
The scoped guard stops only recognized owned burners. It restores burners after each run and after failures.

| Replica | Physical GPU | Logical CUDA device | UUID suffix |
|---|---:|---:|---|
| 0 | 4 | 0 | 6076e2f2-5b63-3761-5586-56ceb7df8139 |
| 1 | 5 | 0 | a1a1cfcf-93a1-3544-9a5e-e58144b68730 |
| 2 | 6 | 0 | e3f3998e-0f1a-e94a-b97c-4abb0e8c2c28 |
| 3 | 7 | 0 | 4cc26b88-19fc-1988-f9e0-17858aa7a99b |

The runtime uses independent DP4, TP1 per replica, and PP1.
It does not execute TP or EP collectives between these replicas.
The checkpoint is `ByteDance/Ouro-1.4B`, revision `574fa66cb8bf5abdc979642d01cf2b79b16bfab1`.
The CDB source SHA is `abdfae0adbdd12be8c6123ccb34eee98a38afb39`.
The upstream source tree remained clean.

| Component | Version |
|---|---|
| Python | 3.14.6 |
| PyTorch | 2.11.0+cu129 |
| CUDA used by PyTorch | 12.9 |
| Transformers | 5.9.0 |
| Tokenizers | 0.22.2 |

The isolated dependency directories are unchanged from v2.
The native Python executable is `/home/esjung/.venvs/cp-composition-vllm026/bin/python`.
The analysis Python executable is `/home/esjung/.venvs/llada2-mini-reference/bin/python`.
`runtime_environment.json` records the model config and its SHA256.

Decode uses CUDA graphs. Prefill graphs remain disabled.
The selected synchronous gate path uses q=0.5, min R2, max R4, and `last_exited` KV.
The optional async CDB path is not a validated baseline in this study.
No threshold, checkpoint, precision, or cache-policy search was performed.

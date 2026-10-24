# Environment and model provenance

The hardware is eight physical H100 80-GB devices.
The initial GPU query and topology appear in `environment/`.
`gpu_uuid_map.json` identifies physical GPUs 0–7.
The topology reports NV18 links between every GPU pair.
No model-parallel TP/PP execution belongs to this study.

| Model | Checkpoint revision | Unique parameters | Parameter bytes | Fixed R | Execution | Configured context limit |
| --- | --- | --- | --- | --- | --- | --- |
| ouro | 574fa66cb8bf5abdc979642d01cf2b79b16bfab1 | 1434652673 | 2869305346 | 4 | async | 65536 |
| huginn | bb6621b65e90b6a4b9b29ef88dc83866d450470c | 3564976800 | 7129953600 | 16 | sync | 4096 |

The model facts come from the actual loaded pretrained modules.
Huginn changes its inference recurrence from checkpoint-default R32 to the registered R16.
No checkpoint tensor changes.
Its fixed prelude injection, adjacent-pair RoPE, and prelude/coda KV remain intact.

The source pin is RLT `4a350344027960a48384626aa646f1765fda6d4f`.
The isolated worktree and every removable source diff remain in provenance and worker identities.
The GPU Python is `/home/esjung/.venvs/ouro-chunk-loop-prefill/bin/python`.
The recorded environment uses Torch 2.11.0+cu129 and FlashAttention4 4.0.0b30.
`validation/hardened_tests/packages.json` contains the complete package lock.

Workers receive distinct CUDA contexts and physical-device masks.
Each worker has a separate CPU-affinity range, CUDA/JIT cache path, log, and raw attempt directory.
The dispatcher uses one measurement process per GPU.
It preserves completed hashes and refuses failed-attempt overwrite.
The resource ledger records verified burner sessions and GPU ownership checks.
Quiet groups reserve the other devices to exclude burner interference.
Other independent stages restore a finished device's burner while its peers continue.
Final handoff restores all eight owned burners.

The shared KV pool uses the same block count and layout within each model's paired cell.
Peak HBM and actual later cache shapes appear in raw rows and the DP export.
Reserved pool capacity is not initialized live KV payload.

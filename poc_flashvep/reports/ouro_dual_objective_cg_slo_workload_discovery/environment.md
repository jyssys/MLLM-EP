# Environment and ownership

The research model is ByteDance/Ouro-1.4B Base, not the Thinking checkpoint.
Revision: `574fa66cb8bf5abdc979642d01cf2b79b16bfab1`.
All research jobs use BF16 and static Full R4.
The 24 physical layers share weights across four recurrent passes.
KV namespaces remain distinct across96 logical layer executions.

## Runtime pins

| Component | Actual pin or version |
| --- | --- |
| Runtime | standalone vLLM-RLT `4a350344027960a48384626aa646f1765fda6d4f` |
| Local runtime patch SHA256 | `1feb47c40a10e2f6c0cc9a314aec86ff96c6b251954eaed31d1be0b859f67862` |
| GPU Python | `/home/esjung/.venvs/ouro-chunk-loop-prefill/bin/python`,3.14.6 |
| Torch |2.11.0+cu129 |
| Attention | flash-attn-4,4.0.0b30 |
| Transformers |5.16.1 |
| Triton |3.6.0 |
| GPU | NVIDIA H10080GB HBM3 |
| Driver |570.211.01 |
| CPU analysis | `/home/esjung/anaconda3/bin/python` |

Workers verify the checkout and patch hash before model construction.
Each immutable attempt stores model-worker source snapshots and its actual repository SHA.
Later analysis-only edits do not change the frozen selector or runtime path.
The evidence validator compares operative selector hashes across FINAL attempts.

## Physical device mapping

| Physical GPU | UUID |
| --- | --- |
|4 |`GPU-6076e2f2-5b63-3761-5586-56ceb7df8139` |
|5 |`GPU-a1a1cfcf-93a1-3544-9a5e-e58144b68730` |
|6 |`GPU-e3f3998e-0f1a-e94a-b97c-4abb0e8c2c28` |
|7 |`GPU-4cc26b88-19fc-1988-f9e0-17858aa7a99b` |

Each worker masks exactly one physical device before Torch imports.
Its logical CUDA0 maps to that physical device, not physical GPU0.
No research process creates a TP or PP group.
CPU affinity uses 16 disjoint logical CPUs per worker, indexed from physical GPU ID.
OMP, MKL, and OpenBLAS use one thread per worker.
GPU contexts, JIT caches, job IDs, logs, warm-up, and raw paths remain separate.
JIT disk-cache paths are separate between GPUs but persist across jobs on the same GPU.
Independent process starts are not cold-JIT-cache starts.
The campaign does not delete caches to manufacture a compilation contrast.
Policy order and device assignment rotate, and input-only warm-up precedes each serving window.

The primary contract prohibits research or resource changes on GPUs0–3.
The request's final all-device burner boilerplate conflicts with that contract.
This campaign follows the narrower explicit restriction.
It preserves existing GPU0–3 processes as fixed background load.
Only verified scope-owned burners on GPUs4–7 can stop or restart.
Unrelated process ownership prevents takeover.

## Memory accounting

BF16 parameter storage is 2,869,305,602 bytes in the inherited model inventory.
This is a structural parameter count, not measured HBM read traffic.
The native KV pool reserves54GiB total for K and V.
It contains 18,432 blocks of 16 positions in a shared depth-indexed block pool.
It is not four separate54GiB pools.
Per-policy peak allocated/reserved memory comes from actual Torch measurements in `serving_results.csv`.

Adaptive engines reserve staged capacity for configured C2048 even when an action selects a smaller C.
The O10 logical retained-state term does not claim that physical reserved memory shrinks with every selected C.
This study does not infer GPU bandwidth or memory traffic from model/storage sizes.

## Reproducibility limits

Four-worker throughput drift remains within the registered 3% review threshold on the measured calibration sample.
Two independent starts per device/worker-count cell do not prove all workload interference is negligible.
The serial confirmation lane runs without other authorized measurement workers or their burners.
It retains the unchanged GPU0–3 background.
The final report must keep the two measurement lanes separate.

Data: `model_source_locks.json`, `gpu_uuid_map.json`, raw `identity.json`, `GPU_policy_rotation.csv`, and `GPU_process_telemetry.csv`.

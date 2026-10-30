# Dependency contract

## Reused release

The parent correctness commit is `3b7a722898dab5f6dd797b474c1ff461659d8fa1`.
The qualified corrections remain unchanged.
Ouro uses `down_norm`.
Huginn uses `down_norm_adapter`.

The external RLT HEAD is `6b164d3844d35ab8e753df1864f0c87dc180401a`.
Its preserved local diff has SHA-256 `4904a311358c5cf6467678d0e1c30af38b5a96064cae4c96ed680943d7e694c7`.
This study does not modify that checkout.
It does not modify the read-only LoopServing reference.

The Python executable is `/home/esjung/.venvs/looped-pd-abc/bin/python`.
The qualified environment uses Torch 2.11.0+cu129, CUDA 12.9, and Transformers 5.16.1.
Attention uses FlashAttention 4, version 4.0.0b30, with `num_splits=1`.
Transport uses NIXL 1.4.1 with UCX.
The report does not claim a new attention or communication kernel.

## Models

| Model | Revision | Depth | Precision |
|---|---|---:|---|
| ByteDance/Ouro-1.4B | `574fa66cb8bf5abdc979642d01cf2b79b16bfab1` | 4 | BF16 |
| tomg-group-umd/huginn-0125 | `bb6621b65e90b6a4b9b29ef88dc83866d450470c` | 32 | BF16 |

KV uses the existing `last_exited` depth-plane layout under full fixed depth.
Huginn retains its reference-compatible per-request RNG protocol.
Neither model uses early exit, quantization, training, or approximate state transfer.

## Runtime equality

All initial arms use capacity 8 and the same cache budget.
Ouro reserves 4096 KV blocks per worker.
Huginn reserves 12288 KV blocks per worker.
Each block holds 16 token positions.
These are reserved capacities, not occupied-token counts.

The quiet comparison enables native graphs with a maximum of eight recurrent rows.
Ouro retains asynchronous scheduling.
Huginn retains its supported synchronous scheduling.
Prefill and mixed batches above the graph limit use native eager execution.
The common-eager diagnostics disable graphs in every arm.
The quiet comparison does not claim optimal graph tuning for large mixed batches.

The diagnostic code records real batch descriptors and matrix shapes.
The clean runs disable those hooks and all heavy byte checks.
Every new process applies the same qualified numerical repair before model loading.

## Evidence manifests

`configs/prerequisite_gate.json` records model-specific release status and GPU UUIDs.
`configs/cpu_audit.json` records source hashes and CPU compatibility tests.
`command_manifest.jsonl` records commands, GPU masks, configuration hashes, and process IDs.
Each run stores its contract, worker mapping, statistics, and raw output token IDs.
Version 2 and later runs also store a separate RCC source archive.
`configs/runtime_overlay_v1.zip` preserves the first diagnostic overlay.

No application source from LoopServing is copied into this implementation.
The study imports existing project adapters and calls existing RLT APIs.
The predecessor numerical report documents the reference-engine comparison.

# Environment and architecture

## Scope

This experiment uses pretrained **ByteDance/Ouro-1.4B Base**. It does not use the Thinking checkpoint.
The exact model revision is `574fa66cb8bf5abdc979642d01cf2b79b16bfab1`.
The checkpoint contains 1,434,652,673 parameters and 269 BF16 tensors.
Its safetensors file contains 2,869,336,434 bytes.
The file SHA-256 is `58872a72616c736595b8b7662079c5b12c5a162ec16eae94f21c348dfa9885af`.

The model card specifies Apache-2.0 and a research-purpose notice.
See the [pinned model repository](https://huggingface.co/ByteDance/Ouro-1.4B/tree/574fa66cb8bf5abdc979642d01cf2b79b16bfab1).
Local tensor metadata and source hashes appear in [checkpoint_audit.json](checkpoint_audit.json).

## Physical devices

The process sees `CUDA_VISIBLE_DEVICES=4,5,6,7`.
Each device is an NVIDIA H100 80GB HBM3 GPU.
All four workers check their device UUID before execution.

| Worker / CUDA device | Physical GPU | UUID suffix |
|---|---:|---|
| 0 | 4 | 6076e2f2-5b63-3761-5586-56ceb7df8139 |
| 1 | 5 | a1a1cfcf-93a1-3544-9a5e-e58144b68730 |
| 2 | 6 | e3f3998e-0f1a-e94a-b97c-4abb0e8c2c28 |
| 3 | 7 | 4cc26b88-19fc-1988-f9e0-17858aa7a99b |

[environment.json](environment.json) contains the full topology, driver, peer-access matrix, and library versions.
The four devices support CUDA peer access.
Jobs on physical GPUs 0–3 remained outside this experiment.
Those workloads remained active. Shared-host and shared-fabric isolation was not established.
The experiment stopped only identified project burners on GPUs 4–7.
Burner logs and process audits remain in this directory.
Burners do not run during measurements, including deliberately idle late-loop devices.

## Runtime

The interpreter is `/home/esjung/.venvs/llada2-mini-reference/bin/python`.
The environment name does not identify the loaded model.
The loaded model is Ouro, not LLaDA.
PyTorch is 2.8.0+cu128. Transformers is 4.57.1. CUDA is 12.8.
Both arms use the pinned official HF decoder, BF16 weights, and PyTorch SDPA.
Both arms use the same full-stage CUDA Graph implementation.

The runtime uses one process with four CPU worker threads.
Each worker owns one GPU, one CUDA stream, and one FIFO ready queue.
This differs from the specification's preferred four-process launcher.
The specification permits equivalent device-to-device transport.
This implementation uses CUDA peer copies instead of NCCL send/receive.
It does not use TP, CP, EP, or prompt chunking.

Each stage waits only for its own CUDA completion event before publication.
Other workers continue independently.
The runtime does not insert a global barrier inside a forward pass.
The terminal worker computes the LM head and argmax.
The token tensor returns directly to the entry GPU.
The controller receives one scalar token ID for request progress.

## Model structure and cache

The checkpoint has 24 physical decoder layers, hidden size 2048, and MLP intermediate size 5632.
It uses 16 query heads, 16 KV heads, and head dimension 128.
Each decoder layer uses the original four sandwich RMSNorm operations.
The original final norm and exit-gate calculation remain at each loop boundary.
Fixed depth truncates actual recurrent execution. It does not select a shallow output after four loops.

The cache index is `24 * loop_index + physical_layer_index`.
Fixed R2/R3/R4 therefore populate 48/72/96 logical slots per request.
Every slot belongs to the GPU that executes that layer at that depth.
No KV tensor crosses a GPU boundary.
R4 gives each GPU 24 logical KV slots in both placements.
Loop-PP retains all 24 decoder weights on every GPU.
Layer-PP retains six decoder layers per GPU.
Both arms replicate the same embedding, final norm, exit gate, and LM head on all four GPUs.

The config permits 65,536 positions and uses RoPE theta 1,000,000 without additional scaling.
This experiment tests prompts up to 2,048 tokens and outputs up to 128 tokens.
It does not test the full configured context limit.

## Reuse and limitations

The experiment reuses the installed checkpoint, tokenizer, environment, and preserved workload prompts.
Earlier CDB code informed the gate/KV audit only.
Its modified decoder math is not the reference for this experiment.
No previous PoC file was overwritten.
The new worktree preserves the main worktree's dirty and staged files.

Python queues, host publication waits, local graph-cache staging, and exact-length graph storage remain prototype costs.
These costs limit claims about an optimized multiprocess serving engine.
See [timing_contract.md](timing_contract.md) for the measured clock and memory definitions.

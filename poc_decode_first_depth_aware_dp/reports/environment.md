# Environment and source identity

## Contract

The authoritative contract is the user's `decode_first_depth_aware_dp_spec.md`, version 2.0.
`reports/spec.md` preserves an unchanged copy.
The original project's `poc_flashvep/reports/spec.md` remains untouched.
This study uses a separate worktree and output directory to preserve earlier experiments.

The parent commit is `eb7d6a400f794d16952cb15889e8d567a5eb0a62`.
The branch is `flashvep/decode-first-depth-aware-dp-poc`.
`artifacts/preregistration.json` records the source hash, quality criteria, and original dirty-state hashes.

## Hardware

The permitted physical devices are GPUs 4, 5, 6, and 7.
Each device is an NVIDIA H100 80GB HBM3 GPU.
The reported capacity is approximately 81,559 MiB per device.
The topology reports NV18 between each permitted pair.
Raw inventory and topology are in `artifacts/environment.json`.

| Physical GPU | Local rank with the four-GPU mask | UUID without the `GPU-` prefix |
|---|---:|---|
| 4 | 0 | 6076e2f2-5b63-3761-5586-56ceb7df8139 |
| 5 | 1 | a1a1cfcf-93a1-3544-9a5e-e58144b68730 |
| 6 | 2 | e3f3998e-0f1a-e94a-b97c-4abb0e8c2c28 |
| 7 | 3 | 4cc26b88-19fc-1988-f9e0-17858aa7a99b |

Each quality worker uses one physical GPU and local CUDA rank 0.
These are independent TP1 / EP1 / PP1 / CP1 model workers.
Four concurrent quality shards do not constitute a live DP4 placement benchmark.

The inherited guard stops only verified utilization workers on its assigned subset.
It restores utilization workers after success or failure.
It refuses unknown GPU contexts and does not signal other users' processes.
It does not signal processes on GPUs 0–3.

## Model selection

| Role | Checkpoint | Revision |
|---|---|---|
| Prior fixed-R4 anchor | ByteDance/Ouro-1.4B-Thinking | `3aaa2224253a92ca45cf2e3d427c360e1ef9c93d` |
| New adaptive candidate | ByteDance/Ouro-1.4B | `574fa66cb8bf5abdc979642d01cf2b79b16bfab1` |

The Thinking anchor's prior validation does not qualify adaptive decode.
The public vllm-rlt support table limits its Thinking validation to bounded fixed-depth synchronous execution.
The spec explicitly permits the base checkpoint as a separate adaptive candidate.
No Thinking accuracy result transfers to that base checkpoint.

The base model has 24 physical layers, maximum depth four, and hidden size 2,048.
It has 16 query heads, 16 KV heads, and head dimension 128.
The model uses BF16 and 96 independent depth/layer KV slots.
`artifacts/checkpoint.json` records file hashes and the actual config.

## Runtime

| Item | Revision or version |
|---|---|
| Official CDB code | `abdfae0adbdd12be8c6123ccb34eee98a38afb39` |
| vllm-rlt, audit only | `9d601ceaf9af63121c742df5f584467e23630a0b` |
| Native interpreter | `/home/esjung/.venvs/cp-composition-vllm026/bin/python` |
| Torch / CUDA | 2.11.0+cu129 / 12.9 |
| CDB Transformers | 5.9.0, isolated task path |
| CDB tokenizers | 0.22.2, isolated task path |
| Public FlashAttention3 | 3.0.0, isolated task path |
| Official HF comparison | Torch 2.8.0+cu128, Transformers 4.57.1, SDPA |

The CDB engine already exists upstream.
This PoC supplies loading metadata, observers, and experiment runners only.
It does not implement another serving engine.
The selected CDB path uses synchronous same-step gating and refill enabled.
The Stage-0 quality lane uses eager execution and B1.
It does not claim efficient high-concurrency CDB performance from those settings.

The official CDB model class defines additional lookahead and preloop heads.
They lack trained weights in this checkpoint and remain unused.
The loader asserts that the selected gate is the original trained `early_exit_gate`.
It also checks that gate weight and bias equal the checkpoint tensors.

## Isolated compatibility work

The initial native environment lacked the public `flash_attn_interface` module.
A CPU import reproduced that error before further GPU tests.
An official Torch-2.11/CUDA-12.9 wheel fixed the import in a task-specific path.
No shared package installation changed.

The CDB checkout pins Transformers 5.9.0 and tokenizers 0.22.2.
The original native environment instead contained Transformers 5.16.1 and tokenizers 0.23.2.
The task installs the pinned versions in its isolated dependency directory.
An attempted tokenizers 0.23.0 installation failed because that stable version was unavailable.
The final choice follows the upstream lock file, not a model-quality result.

The pinned ByteDance config omits `pad_token_id`.
Transformers 4.57 supplies an implicit `None` value.
The CDB loading wrapper explicitly restores that metadata default for Transformers 5.9.
No padding token, embedding weight, or model arithmetic changes.
Neither upstream checkout nor the checkpoint files change.

## Preserved failures

| Attempt | Failure or result |
|---|---|
| Runtime smoke v1 | Missing public FlashAttention3 module |
| Runtime smoke v2/v3 | Missing legacy config padding attribute. One attempt preceded dependency installation completion |
| Runtime smoke v4 | Incompatible tokenizers version |
| Runtime smoke v5 | JSON serialization of `Counter` through `dataclasses.asdict` |
| Runtime smoke v6 | Actual fixed/adaptive execution and exit-KV checks complete |
| Official reference smoke v1 | Numeric limits pass, one teacher top-1 difference, natural prefix differs |
| Aligned reference v1 | Missing optional accelerate support for `device_map` |
| Aligned reference v2 | Released reference code expects the older default-RoPE registry entry |

The loading wrapper avoids `device_map` for the aligned reference retry.
The study does not patch the released RoPE implementation to force that retry through.
All failed logs remain available.
The serialization correction reads actual counter values without changing execution.

# Dependency and input lock

| Component | Fixed value |
|---|---|
| Parent experiment commit | `b82eabd20a4ec168a5ad9a119c8401f97a565e78` |
| Private RLT upstream commit | `6b164d3844d35ab8e753df1864f0c87dc180401a` |
| Private RLT tracked patch SHA-256 | `4904a311358c5cf6467678d0e1c30af38b5a96064cae4c96ed680943d7e694c7` |
| Main Python | `/home/esjung/.venvs/looped-pd-abc/bin/python` |
| Reference Python | `/home/esjung/.venvs/looped-pd-abc-hf457/bin/python` |
| PyTorch, both environments | `2.11.0+cu129` |
| Main Transformers | `5.16.1` |
| Reference Transformers | `4.57.6` |
| Main `flash-attn-4` | `4.0.0b30` |
| NIXL | `1.4.1` |
| NumPy | `2.4.6` |
| pytest | `9.1.1` |
| Ouro model and tokenizer | `ByteDance/Ouro-1.4B` at `574fa66cb8bf5abdc979642d01cf2b79b16bfab1` |
| Huginn model and tokenizer | `tomg-group-umd/huginn-0125` at `bb6621b65e90b6a4b9b29ef88dc83866d450470c` |
| ShareGPT dataset revision | `192ab2185289094fc556ec8ce5ce1e8e587154ca` |
| Downloaded dataset SHA-256 | `35f0e213ce091ed9b9af2a1f0755e9d39f9ccec34ab281cd4ca60d70f6479ba4` |

The model revision also fixes tokenizer files and checkpoint-owned chat templates.
Both models use BF16, greedy generation, and natural EOS with an explicit safety cap.
Ouro executes R4. Huginn executes R32, plus its original prelude and coda.
The model does not use dynamic exit.

The cache-layout name is `last_exited`.
All fixed-depth recurrent planes remain present. This configuration does not approximate missing deeper KV.
The source disables prefix caching through `CacheConfig`'s default `enable_prefix_caching=False`.
Each independent job starts a new engine and records warmup separately.

NIXL uses the pinned CUDA/UCX path. The node exposes eight H100 80 GB GPUs with NV18 links between every pair.
`configs/environment_inventory.json` records driver, clock, CPU, RAM, topology, and physical UUID information.
Per-run `worker_map.json` files check the actual stage-to-GPU assignment.

The main engine is the patched standalone vLLM-RLT execution and PD engine.
It is not a stock vLLM HTTP endpoint.
The arrival harness includes engine and IPC work but does not measure an external network frontend.

The initial qualification predates the final metadata-only worker freeze.
`configs/heldout_runtime_freeze.json` fixes the worker and common-source hashes before held-out B serving.
Each raw run retains both the prior-runtime source archive and the follow-up source archive.
The private runtime patch remains unchanged during this follow-up.

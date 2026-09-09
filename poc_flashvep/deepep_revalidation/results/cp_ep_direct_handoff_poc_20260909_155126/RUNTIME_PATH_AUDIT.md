# Runtime Path Audit

## Scope

- Physical GPUs: 4, 5, 6, 7 only; four NVIDIA H100 80GB HBM3 GPUs connected pairwise by NV18.
- Model checkpoint: `Qwen/Qwen3-VL-30B-A3B-Instruct`, snapshot `9c4b90e1e4ba969fd3b5378b57d966d725f1b86c`.
- Model config: BF16, hidden size 2048, 48 layers, 32 query heads, 4 KV heads, head dimension 128, 128 routed experts, top-8, no shared expert, MoE intermediate size 768.
- Replay environment: Torch 2.11.0+cu129, vLLM 0.20.0, DeepEP 1.2.1 build from `73b6ea4`, SM90 enabled.
- Repository base commit: `f965710580a1984255aa88649285f95923df8338`.

## Native runtime audit

### Local vLLM

The installed vLLM has `prefill_context_parallel_size`, PCP groups, and PCP-aware MoE rank flattening. It also has an explicit MoE bridge in `model_executor/layers/fused_moe/runner/moe_runner.py`: PCP all-gathers both hidden states and router logits before MoE, then PCP reduce-scatters the MoE output. The source itself describes this as a separately added `AgRsAll2All` path for simplicity.

The standard attention interface defaults `supports_pcp = False`, and `v1/worker/cp_utils.py` asserts this capability for PCP. No installed attention implementation declares `supports_pcp = True`. Consequently this local release cannot execute Qwen3-VL GQA through its standard native PCP path. This is an environment/path limitation, not a method failure.

### Current SGLang source

Audited commit: `78da62519012a06833ae37ea9a214d101c2963b8`.

Current source supports Qwen3 MoE prefill CP and makes the boundary explicit. `layers/communicator.py::_gather_hidden_states_and_residual_moe` gathers CP token shards before MoE; `_scatter_hidden_states_moe` returns each CP-local token slice afterward. `models/qwen3_moe.py` invokes `prepare_mlp` only after the attention call has returned. No exact token-block attention-to-MoE streaming handoff or route-aware direct transport was found.

The audited checkout requires a newer Torch/FlashInfer/sgl-kernel stack than the stable local serving environment. A native setup was not attempted beyond the bounded audit because an actual Qwen weight/activation layer replay could answer the headroom gates without irreversibly replacing the working environment.

### Megatron-LM and TensorRT-LLM

- Megatron-LM commit: `be85fc5df550f2236a850ecbf0dd1bf1b4cb4814`.
- TensorRT-LLM commit: `2ece8d97d28bb5724e0c38244c7155c87fc33d24`.

Megatron supports CP+EP and MoE Parallel Folding, but the attention CP and MoE token dispatcher remain separate execution contracts. TensorRT-LLM exposes Ulysses/Helix CP configurations; Helix is decode-oriented and several MoE backends reject Helix CP. Neither source audit found the exact long-prefill block-streaming/direct-handoff primitive tested here.

## Selected faithful path

The experiment therefore used the allowed priority-C path: a forward-only, actual-Qwen layer replay with checkpoint attention/router/expert weights, captured Qwen3-VL hidden-state distributions, exact Ulysses-style CP4 collectives, DeepEP EP4 high-throughput dispatch/combine, and local fused expert execution.

This is a real four-GPU layer execution and not an analytical simulator. It is not a native continuous-serving request run. Request-level integration was intentionally not attempted after the actual layer prototype failed both the performance and strict correctness gates.

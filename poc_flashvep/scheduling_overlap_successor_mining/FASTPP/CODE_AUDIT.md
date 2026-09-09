# FastPP source audit

Official `Sys-KU/FastPP` revision
`cb3ce5b39c085a671b2c79477ac6d673b55f8eac` (includes ALP attention profiling fix).
This is an SGLang 0.4.1 fork, not the project's newer vLLM serving engine.

## Verified mechanisms

- `python/sglang/srt/managers/dynamic_chunk_controller.py`: greedy decision
  period is PP-depth steps; ALP period is twice PP depth. Rank zero chooses and
  broadcasts decisions. Metrics include all PP microbatches, not one local row.
- `alp_scheduler.py`: offline no-attention cost table for 128-token increments
  through 2048; three warmups and ten timing repetitions. Profile mode is
  `scheduler_loop_runtime_aligned_no_attention`. Runtime RLS features include
  decode context, squared prefill length, prefix/prefill interaction and bias.
  Forget factor 0.9975; relative-error clipping 0.4; pre/post fits are separate.
  Thus 'token count only' is not a faithful description of ALP.
- Cache reuse depends on profile mode/table completeness. Isolate coefficient
  paths per model/environment; avoid accidentally reusing another model's fit.
- `batch_rebalancer.py`: 128-aligned redistribution and dual TPOT/E2E goals,
  retaining KV for suspended requests. Both resume cost and tail ITL matter.
- `qwen3_moe.py` genuinely implements PP-aware layer ownership, boundary hidden
  and residual transfer, and first/last-stage embedding/output ownership.
- Its `layers/moe/fused_moe_triton/layer.py` uses all logical experts with
  `intermediate_size // tp_size`. No expert sharding appears in this class.
  `enable_ep_moe` assigning `ep_size=tp_size` is NOT evidence that Qwen3 uses EP.
  PP×TP transfer is native; Qwen3 PP×EP needs separate verified implementation.
- Important ownership detail: `model_executor/model_runner.py` imports process
  group initialization from installed `vllm.distributed`, whereas some model files
  still import SGLang group helpers. Both code paths must be checked at runtime;
  unused copies of a communicator are not execution evidence.
  Follow-up verification: this pinned `qwen3_moe.py` and its fused-MoE layer both
  use **vLLM** group helpers, matching `ModelRunner`; no communicator import
  patch is needed for this Qwen3 file. This does not establish EP sharding.

## Official baseline feasibility

Isolated Python 3.10 environment import sanity passed on 2026-09-08:
torch 2.5.1+cu124, transformers 4.51.2, vLLM 0.6.4.post1, SGLang 0.4.1.
This is dependency sanity, not a GPU or correctness pass.

Official dense path is Qwen2.5-32B, PP4, BF16, disabled overlap scheduler/radix
cache/CUDA graph, mixed chunking. PP>1 is not compatible with this fork's separate
overlap-scheduler path. Test PP-only, greedy, ALP and rebalancing under the same
supported options. The paper used A100 PCIe; H100 NVLink is a hardware transfer,
not a reproduction of its exact TP-versus-PP speedup.

## Failure criteria

Measure ALP prediction error *and* request-level regret against exact chunk,
stage-cost and joint partition oracles. Lower MAPE without E2E benefit is not
success. Native Qwen EP unsupported must be PORT/ENVIRONMENT, not METHOD failure.
No material failure is established yet.

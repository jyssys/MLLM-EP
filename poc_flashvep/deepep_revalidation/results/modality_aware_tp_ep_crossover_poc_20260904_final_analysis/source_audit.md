# Source and configuration audit

## Reproduction context

- Repository: `/home/esjung/MLLM-EP-github`
- Experiment branch: `flashvep/modality-aware-tp-ep-crossover-poc`
- Model snapshot: `/home/esjung/.cache/huggingface/hub/models--Qwen--Qwen3-VL-30B-A3B-Instruct/snapshots/9c4b90e1e4ba969fd3b5378b57d966d725f1b86c`
- Local runtime: vLLM 0.20.0, eager mode, BF16, Triton MoE backend
- Physical devices used by the completed runs: `CUDA_VISIBLE_DEVICES=1,2,3,4`
- Run protocol: two warmup requests and eight measured repetitions per workload, one persistent `LLM` instance per topology. The hook only observes existing calls and resolves CUDA events once at worker exit.

## Model facts

The model `config.json` contains a nested `text_config` with `hidden_size=2048`, `num_hidden_layers=48`, `num_experts=128`, and `num_experts_per_tok=8`, with BF16 dtype. The four-image workload expands to 3,163 prompt tokens, the two-image mixed workload to 3,022, and the text control to 2,980.

## Qwen3-VL path

The local `qwen3_vl_moe.py` constructs `Qwen3_VisionTransformer` and a Qwen3 MoE language model. The inherited `qwen3_vl.py` path runs `self.visual(pixel_values, grid_thw)` in `encoder_eager_forward`, then `_process_image_input` splits the concatenated result by `grid_thw`/`spatial_merge_size` before multimodal embeddings are inserted into the language input. The language path uses `Qwen3MoeDecoderLayer`; its order is input norm → self attention → post-attention norm → `self.mlp(hidden_states)`. `Qwen3MoeSparseMoeBlock` delegates routing and expert execution to vLLM `FusedMoE`.

## Parallel-config semantics (critical)

In the installed vLLM source (`vllm/model_executor/layers/fused_moe/config.py`):

```python
use_ep = (dp_size * pcp_size * tp_size > 1
          and parallel_config.enable_expert_parallel)
use_all2all_kernels = self.dp_size > 1 and self.use_ep
use_deepep_ht_kernels = (self.use_all2all_kernels
                          and all2all_backend == "deepep_high_throughput")
```

Therefore, the requested `TP4 / DP1 / EP4` flag does shard experts (`use_ep=True`, four EP ranks), but it **does not activate all-to-all/DeepEP kernels** because `dp_size=1`. The completed EP-flag run proves `prepare_finalize_backend=MoEPrepareAndFinalizeNoDPEPModular` on all measured layers, exactly like the TP-only run. This is a local vLLM 0.20 execution-semantic limitation, not an inferred timing result. A true DeepEP EP comparison requires a DP>1 topology (for example the optional TP2/DP2/EP4 configuration), which was not substituted into the preregistered primary comparison.

## Instrumentation

`poc_flashvep/modality_aware_tp_ep_crossover/worker_hook.py` wraps the existing `FusedMoE.forward`, modular `_prepare`, `_fused_experts`, and `_finalize` calls. It records per-layer CUDA-event spans, route expert IDs, a diagnostic `expert_id // 32` rank histogram, active experts, and backend class names. It does not mutate routing, placement, model weights, or scheduling. The event span is a layer-local elapsed duration; independent GPU clocks are not subtracted.

## CAI reference audit

The clean reference clone is `/tmp/Capacity-Aware-MoE`, commit `9c73c8eee6ca64836eb873e77aa096fb4955e658`. Its top-level `capacity_aware/capacity_patch.py` is a HuggingFace-style router monkey patch implementing capacity/token-drop policies; `VLMEvalKit/` and the evaluation harness organize multimodal experiments. It is a reference/scaffold only in this PoC; no CAI routing policy is installed in the vLLM run.

## Scope caveat

No TP2/DP2/EP4 run was added after the final matched pair because the primary question specified the TP4/DP1 pair and the available physical GPU slots were occupied by unrelated processes. Existing historical TP2/DP2 artifacts are not mixed into this result because they use different requests, seeds, and protocols. Consequently this report treats the requested “TP4/DP1 EP4” result as a **configuration-semantic audit**, not as evidence of a TP-vs-DeepEP crossover.

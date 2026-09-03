# Source audit

## Runtime and model path

The installed local source is under `/home/esjung/anaconda3/envs/flashvep-poc/lib/python3.12/site-packages/vllm` (vLLM 0.20.0). Qwen3-VL's vision path is in `vllm/model_executor/models/qwen3_vl.py`: `Qwen3_VisionPatchEmbed` (line 347), `Qwen3_VisionMLP` (376), `Qwen3_VisionBlock` (413), `Qwen3_VisionPatchMerger` (467), and `Qwen3_VisionTransformer` (519). `Qwen3_VLForConditionalGeneration.encoder_eager_forward` is at line 2056. The vLLM encoder worker calls this path from `vllm/v1/worker/encoder_cudagraph.py:376`.

`grid_thw` and cumulative sequence lengths (`cu_seqlens`) describe each image's patch sequence. The vision transformer receives concatenated image patches and returns the complete visual embedding tensor before the language model consumes it. There is no per-image ready callback in this path. Image sequences are structurally independent, but exposing image-level completion would require a runtime interface change: **POSSIBLE_WITH_RUNTIME_CHANGE**.

## Language MoE path

`vllm/model_executor/models/qwen3_moe.py` contains `Qwen3MoeSparseMoeBlock` (line 137), `Qwen3MoeAttention` (261), and `Qwen3MoeDecoderLayer` (364). The decoder layer performs self-attention, post-attention layer norm, then the MLP/MoE block (`forward` around line 416). The sparse block computes router/top-k and invokes the installed `FusedMoE` implementation. The local wrapper installed read-only NVTX hooks at decoder, attention, router/MoE, expert, and DeepEP prepare/finalize boundaries; it did not alter tensors, routing, weights, streams, or scheduling.

## DeepEP HT semantics

`vllm/model_executor/layers/fused_moe/prepare_finalize/deepep_ht.py` defines `DeepEPHTPrepareAndFinalize` (line 28). `_do_dispatch` (line 97) captures a previous compute event, switches to the communication stream, calls `get_dispatch_layout`, and calls `buffer.dispatch(previous_event=...)`. `_finalize` (line 336) calls `buffer.combine(previous_event=...)`, then waits on the returned event before copying the combined output back. Thus the integration uses stream/event dependencies and collective completion; it does not expose a prefix-specific completion contract.

The DeepEP C++ path (`csrc/deep_ep.cpp`, local checkout) uses the caller stream for layout and synchronizes the communication stream when no prior event is supplied. In the bounded replay, kernel names directly identify `deep_ep::layout::get_dispatch_layout`, `deep_ep::intranode::notify_dispatch`, `deep_ep::intranode::dispatch`, `cached_notify_combine`, and `intranode::combine`. NCCL barrier/all-gather/all-reduce kernels are observed but their logical sub-phase is not inferred.

## Profiling hooks and limitations

`poc_flashvep/resource_atlas/atlas_hook.py` is loaded via local `sitecustomize.py`, avoiding permanent installed-source edits. Full-serving driver smoke completed successfully in attempts `20260903_131420` and `20260903_131609` with TP2/DP2/EP4, but child worker CUDA activity was absent from the exported SQLite. Attempts using `--trace=nvtx` hit a segmentation fault in NCCL 2.28.9's `nvtxExtInitOnce_v3` during `ncclGetUniqueId`; this is recorded as an environment/profiler limitation. The final `.nsys-rep`/SQLite therefore deliberately uses exact-route bounded replay evidence, not fabricated full-serving phase attribution.

## Requested NVTX phase coverage

The hook contains labels for `VISION_PATCH`, `VISION_ATTN`, `VISION_MLP`, `VISION_MERGER`, `LLM_PREFILL`, `LLM_ATTN`, `ROUTER_TOPK`, `EXPERT_GEMM`, `DEEPEP_DISPATCH`, `DEEPEP_COMBINE`, and `LLM_MOE`. Fine-grained QKV/O-projection/norm and decode subranges were not safely separable at Python boundaries in this installed path and remain source-inferred/unknown in the signature table. No source package was modified.

# Full-serving Qwen3-VL resource-atlas source audit

## Reproducibility

- Source revision used for the local read-only hook/parser: `aa4de81` (`fix: include measured vision encoder in capture window`).
- Installed runtime: vLLM `0.20.0` from `/home/esjung/anaconda3/envs/flashvep-poc/lib/python3.12/site-packages/vllm`.
- Model: local Qwen/Qwen3-VL-30B-A3B-Instruct snapshot `9c4b90e1e4ba969fd3b5378b57d966d725f1b86c`.
- Runtime: BF16, V1, eager, TP2/DP2/EP4/PP1, DeepEP high-throughput, TritonExperts, linear placement, DBO off, prefix cache off.
- Visibility: `CUDA_VISIBLE_DEVICES=1,2,3,4`; the CUDA-local device IDs 0--3 correspond to physical GPUs 1--4.

## Vision path

`qwen3_vl.py` defines `Qwen3_VisionPatchEmbed` (line 347), `Qwen3_VisionMLP` (376), `Qwen3_VisionBlock` (413), `Qwen3_VisionPatchMerger` (467), and `Qwen3_VisionTransformer` (519). `Qwen3_VisionTransformer.forward` (784) performs patch embedding, positional addition, a loop over vision blocks (`cu_seqlens`/`grid_thw`), optional deep-stack mergers, and the final merger before returning the complete visual embedding tensor. `grid_thw` and `cu_seqlens` keep image/frame sequences structurally separated, but the current vLLM path exposes no per-image ready callback: all visual embeddings are returned before LM prefill.

The local hook marks `VISION_PATCH`, `VISION_ATTN`, `VISION_MLP`, and `VISION_MERGER`. The full-serving capture contains all four ranges and their CUDA kernels. The patch range contains `nvjet_tst_*`; attention contains SM90 FlashAttention/CUTLASS and auxiliary normalization/elementwise kernels; MLP contains `nvjet_tst_*`, GELU/elementwise and layer-norm kernels; merger contains small `nvjet_tst_*` and layer-norm kernels. Nsight Systems does not expose TensorCore/HBM utilization in this capture, so those properties remain `UNKNOWN` or source-based inference.

## Language/MoE path

`qwen3_moe.py` defines `Qwen3MoeDecoderLayer` (364; `forward` 416). Its order is input norm/residual, `self_attn`, post-attention norm/residual, then `self.mlp`. The sparse MLP (`Qwen3MoeSparseMoeBlock`) performs router/top-k and invokes the modular `FusedMoE` implementation. The hook marks `LLM_ATTN`, `ROUTER_TOPK`, `DEEPEP_DISPATCH`, `EXPERT_GEMM`, and `DEEPEP_COMBINE`; decode markers are recovered from nested markers inside the `LLM_DECODE` wrapper.

`deepep_ht.py` defines `DeepEPHTPrepareAndFinalize` (line 28). `_do_dispatch` (97 onward) obtains a previous CUDA event, switches to the communication stream, calls `get_dispatch_layout` (the layout kernel), then `buffer.dispatch(previous_event=..., async_finish=...)`. `_finalize` (around 336) calls `buffer.combine(previous_event=..., async_finish=...)`, waits on the returned event when required, and copies the combined output back. Thus DeepEP uses stream/event dependencies and collective completion; there is no prefix-specific completion contract. Kernel launches on the communication stream can outlive the Python/NVTX wrapper, so the parser gives stable DeepEP/top-k/expert kernel names precedence over naive CPU-range containment.

## Capture method and child-worker fix

The successful full-serving capture used Nsight Systems 2024.6.2 with `--capture-range=cudaProfilerApi`, `--trace=cuda,nvtx,osrt`, `--trace-fork-before-exec=true`, `--wait=all`, and no CPU sampling/context-switch tracing. After model/NCCL/DeepEP initialization and two warmups, the driver touched a signal file; the read-only `sitecustomize` hook in each CUDA-owning vLLM child worker called `torch.cuda.cudart().cudaProfilerStart()`. A second signal caused each child to call `cudaProfilerStop()`. This avoids the prior NVTX/NCCL initialization crash and starts collection in the worker process that owns the CUDA context. The capture ended with the expected Nsight stop signal (`nsys` returned 143), but the `.nsys-rep` and SQLite exports are complete.

## Attribution limitations

NVTX range durations in `resource_signature.csv` are sums over all observed worker/rank instances and overlap in time; they are not an end-to-end critical-path sum. `cuda_kernel_time_ms` is the sum of individual CUDA kernel durations (also across devices), not utilization. QKV/O-projection/norm/residual do not have independent safe Python boundaries in this vLLM release and are marked source-inferred where no dedicated range exists. TP communication is directly observed through `cross_device_reduce_1stage` kernel names and is reported as a diagnostic aggregate because it is nested in attention/other wrappers. Decode sub-phase intervals are derived from nested ranges and are reported separately from prefill.

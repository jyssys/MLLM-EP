# Prior boundary

## Profiling and graph execution

[NVIDIA Nsight Systems](https://docs.nvidia.com/nsight-systems/UserGuide/index.html)
already provides CPU, CUDA, NVTX, graph, and kernel timelines.
Graph-level tracing has lower overhead than node tracing.
Installed Nsight 2024.6 lacks some current documentation features.
We did not assume support for newer NVTX precapture projection or advanced NCCL tracing.
CPU sampling is unavailable under the server's perf permission policy.
No global permission or clock setting changed.

[NVIDIA CUDA Graph guidance](https://docs.nvidia.com/dl-cuda-graph/troubleshooting/performance-issues.html)
already discusses launch gaps and static-buffer costs.
[PyTorch CUDA Graph documentation](https://docs.pytorch.org/docs/2.14/notes/cuda.html)
requires stable replay addresses and long-lived input/output storage.
The documentation version is not the installed PyTorch version.
Only APIs verified in installed PyTorch 2.8 were used.
Graph reuse, warmup, and profiling are not research novelty.

## The observed copy cause is generic

[PyTorch 2.8 TensorShape.cpp](https://raw.githubusercontent.com/pytorch/pytorch/v2.8.0/aten/src/ATen/native/TensorShape.cpp)
implements `slice_scatter` through `clone_preserve_strides`.
That helper clones the underlying storage and reconstructs the original view.
This behavior preserves strides and storage offset for functionalization.
It is intentional behavior, not an incorrect numerical result.

The measured runtime passes a layer KV view into that operation.
The view belongs to a monolithic cache containing all layers and both K/V tensors.
A compact value-preserving output does not require that full backing allocation.
The selected control changes this runtime layout choice, not PyTorch globally.
The observation and correction do not establish a new multi-GPU architecture.

## Parallel serving and diffusion systems

[FastPP, OSDI 2026](https://www.usenix.org/conference/osdi26/presentation/hwang)
addresses pipeline imbalance through chunk control and decode scheduling.
Generic PP bubbles and improved pipeline scheduling are covered prior.
This PoC does not implement a PP scheduler or treat dependent refinements as independent microbatches.

[BlockServe](https://arxiv.org/abs/2607.08930) uses block-grained continuous batching
and mixed-state execution for diffusion serving.
Independent-request throughput is not equivalent to B1 latency.
No new batching method is claimed here.

[Denoising Surface](https://arxiv.org/abs/2610.00499) models block-step workload structure
and predicts serving cost. A block-by-iteration heatmap is not novel by itself.
This PoC measures physical operations rather than training a workload predictor.

[Epoch](https://arxiv.org/abs/2609.09748) compiles diffusion blocks for sparse MoE serving.
It reduces work associated with decoded positions and dense EP traffic.
This PoC changes no acceptance, router, expert work, or live-token policy.
Native EP1 has no EP dispatch/combine A2A. No measured Epoch comparison ran.
The older 5.77× live-only accounting is not an Epoch runtime result.

[FlashBlock](https://proceedings.mlr.press/v306/chen26g.html) already caches block-external attention.
[LoSA](https://proceedings.mlr.press/v306/xi26d.html) reduces KV work through sparse attention and stable-token reuse.
The exact control here neither reuses stale attention outputs nor changes masks.

[vLLM decode context parallelism](https://github.com/vllm-project/vllm-project.github.io/blob/main/_posts/2026-08-07-decode-context-parallelism.md)
is an existing KV-sharding direction.
The earlier exact CP prototype remains closed. We did not build another CP candidate.
Dynamic context parallelism for training is a separate use of the DCP acronym.

## Claim boundary

This campaign can establish a physical runtime cost and an exact E2E control.
It cannot claim novelty from generic storage compaction, CUDA Graphs, TP/DP/PP,
distributed softmax, or ordinary block-level batching.
A useful generic baseline correction must precede a claim about a new architecture.

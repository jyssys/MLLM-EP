# Nsight Systems findings

## Capture

- Nsight Systems 2024.6.2.225-246235244400v0.
- Full serving profile: `nsys/full_serving_resource_atlas.nsys-rep` (112,885,849 bytes).
- Exported SQLite: `nsys/full_serving_resource_atlas.sqlite` (353,030,144 bytes).
- Child worker CUDA contexts: PIDs 3647414, 3647415, 3647418, 3647419 on visible devices 2, 3, 0, 1 (physical GPUs 3, 4, 1, 2 respectively).
- `driver_status.json` exit codes `[0,0]`.

## Kernel evidence

SQLite contains 424,716 kernel rows and 3 CUDA streams. Dominant observed families:

| family | count | summed kernel time | dominant name |
|---|---:|---:|---|
| DeepEP dispatch/notify | 25,344 | 3,424.97 ms | `deep_ep::intranode::notify_dispatch` |
| DeepEP combine/notify | 25,344 | 2,453.81 ms | `deep_ep::intranode::cached_notify_combine` |
| NCCL/TP collective | 25,612 | 3,884.49 ms | `ncclDevKernel_AllGather_RING_LL` |
| Expert | 23,088 | 335.37 ms | `fused_moe_kernel` |
| Router | 24,224 | 91.38 ms | `vllm::moe::topkGating` |
| Layout | 12,672 | 48.71 ms | `deep_ep::layout::get_dispatch_layout` |
| Attention | 600 | 8.48 ms | CUTLASS FlashAttention forward |

The full trace therefore includes actual child CUDA kernels, not only the driver process. NVTX export is empty because the run intentionally avoided the earlier NCCL/NVTX initialization crash; attribution is from kernel names and same-device stage records.

## Timeline divergence

Normal rows show sub-millisecond DeepEP kernels. Tail rows show a long `notify_dispatch` kernel on the same communication stream (up to 98.97 ms in the profiled capture) or an inflated CUDA-event dispatch span in the online run, followed by ordinary expert/combine kernels. A corresponding `deep_ep::intranode::barrier` kernel (up to 495.56 ms in the capture) confirms peer/collective synchronization machinery is present. The run does not establish a separate host allocator cause.

The one-wave capture intentionally used only one warmup and therefore also shows a first-use prefill event (M=512) with long layout/combine spans; that initialization sample is not used to define the warmed decode root cause. In the warmed online population, the 99.19 ms `deepep_dispatch` stage event corresponds closely to the 98.97 ms `notify_dispatch` kernel, tying corrected event measurement to an actual child-worker kernel rather than host logging delay.

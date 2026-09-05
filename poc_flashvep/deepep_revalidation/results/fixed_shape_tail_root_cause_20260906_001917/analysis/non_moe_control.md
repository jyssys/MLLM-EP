# Non-MoE control (bounded)

The corrected hook's event table is MoE-boundary focused and does not expose a native V1 scheduler id or separate attention event row. As a bounded control, the Nsight child-worker trace was queried for non-MoE kernels in the same captured serving wave.

- FlashAttention forward kernels: 600 instances, 8.48 ms summed GPU time, maximum instance 0.019 ms.
- Router `topkGating`: 12,672 instances, 55.25 ms summed GPU time, maximum instance 0.006 ms.
- NCCL AllGather and vLLM cross-device reduction are visible as separate collectives.
- No multi-hundred-millisecond attention kernel coincides with the dispatch-tail class; the long kernel rows are `deep_ep::intranode::notify_dispatch`, `cached_notify_combine`, or the DeepEP barrier.

This is a kernel-family control, not a per-iteration attention event correlation. It is sufficient to reject a compute-heavy attention kernel as the primary explanation for the giant fixed-shape MoE tail; explicit attention events remain a follow-up instrumentation improvement if scheduler-level attribution is required.

# Nsight corroboration

The prior fixed-shape root-cause run on this same Qwen3-VL / TP2-DP2-EP4 /
DeepEP HT path captured all child workers with Nsight Systems 2024.6.2.225.
SQLite contained `deep_ep::intranode::notify_dispatch` (up to 98.97 ms),
`deep_ep::intranode::barrier` (up to 495.56 ms), DeepEP dispatch/combine,
Triton expert kernels, and NCCL collectives.  The same run had a 99.19 ms
CUDA-event dispatch span, linking the tail to communication notification /
barrier execution rather than expert GEMM.

This follow-up does not rerun the expensive Nsight capture.  The new hook
provides same-device CUDA event timing around the two stock
`EventOverlap.current_stream_wait()` calls and records `comm_stream_id` and
stream-drain duration.  No cross-device CUDA timestamp subtraction is used.
The native `EventHandle` has no query method, so a direct ready/not-ready bit
is not exposed by this installed DeepEP build.

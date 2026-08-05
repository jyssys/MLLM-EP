# FlashVEP Phase 0/1 PoC

This directory contains only the repository audit and exact-baseline profiling
work requested for FlashVEP Phase 0 and Phase 1. Profiling is opt-in. No tile
execution, scheduling, overlap implementation, custom CUDA/Triton kernel, or
model/checkpoint change belongs in this directory during this session.

See `STATUS.md` and `reports/offline_wavefront_quick_poc.md` for the current
scheduler-free TP2/DP2/EP4 capture and EP4 operator-replay gate. The current
decision is NO-GO: the best expert-centered wavefront result was 1.0343x even
though correctness and actual CUDA-event overlap passed. Attention/Router and
vLLM integration were not started. Older reports preserve the Batch 16/32,
TP4/EP4, and TP7 decisions.

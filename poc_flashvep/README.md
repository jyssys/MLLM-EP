# FlashVEP Phase 0/1 PoC

This directory contains only the repository audit and exact-baseline profiling
work requested for FlashVEP Phase 0 and Phase 1. Profiling is opt-in. No tile
execution, scheduling, overlap implementation, custom CUDA/Triton kernel, or
model/checkpoint change belongs in this directory during this session.

See `STATUS.md` and `reports/batch16_32_quick_poc.md` for the current
TP2/DP2/EP4 Quick PoC gate. The current decision is HOLD: Batch 16 was valid,
while Batch 32 exceeded the profiler-overhead limit. Phase 2 was not started.
The older reports preserve the TP4/EP4 gate and prior TP7 blocker.

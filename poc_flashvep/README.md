# FlashVEP Phase 0/1 PoC

This directory contains only the repository audit and exact-baseline profiling
work requested for FlashVEP Phase 0 and Phase 1. Profiling is opt-in. No tile
execution, scheduling, overlap implementation, custom CUDA/Triton kernel, or
model/checkpoint change belongs in this directory during this session.

See `STATUS.md` and `reports/phase1_profile_tp4.md` for the current TP4/EP4
gate. The gate failed at a 1.0349x optimistic median speedup, so Phase 2 was
not started. The older `reports/phase1_profile.md` preserves the prior TP7
blocker.

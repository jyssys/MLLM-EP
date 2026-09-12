# Experiment log

All timestamps use Asia/Seoul unless explicitly marked.

## 2026-09-12

- Created isolated worktree on `flashvep/llada2-flash-100b-ep-scaling-method-poc`.
- Saved and read the complete working specification.
- Audited physical GPU topology: GPUs 0--3 are H100 80GB and fully NVLink-connected.
- Kept the pre-existing task-owned idle burner running during CPU-only setup.
- Pinned model revision `744c3f8c6c8317d2377d6d16d8a3d4be2caef563`; began downloading all 42 weight shards.
- Cloned official dInfer at `1ffeb961cd258bede74fcf5ca8a416ae6d57b18f`.
- Verified that the official four-GPU example is TP4, not sparse EP4.
- Located the DeepEP dispatch/local-expert/combine path and the contiguous 64-expert/rank EP4 mapping.
- Created isolated SGLang 0.5.3.post1 environment and began dependency validation.
- Applied only substrate-enabling patches: rank-local expert load filtering and explicit DeepEP benchmark arguments.
- Verified the downloaded 100B revision and completed all 42 shards.
- Found and fixed a second official loader defect: unquantized local experts
  were fused only on the rank containing global expert zero. An independent
  checkpoint replay reduced routed-output error from 15.39% to 0.373%.
- Established dense-TP4+routed-EP4 with 64 complete experts/rank, 75.1%
  remote assignments, DeepEP normal dispatch/combine, BF16 Triton experts,
  exact source-row gather, and no CPU offload.
- Ran matched TP4/EP4 correctness and bounded quality checks; GSM8K and
  HumanEval scores match despite BF16 trajectory-level string differences.
- Ran submitted-batch sweep 1/2/4/8/16/32 with three independent restarts.
  Observed a nominal crossover but strong sign-changing shared-state variance.
- Ran physical forward microbatch sweep 1/2/4/8/16. Existing static tuning is
  the largest effect: EP4 mini4 to mini8 reduces clean BCT 47.36%.
- Captured 13,578 logical sparse-layer invocations for temporal analysis.
  Lag-1 top-k overlap is 74.29%, exact set agreement 29.11%, destination
  overlap 93.18%, and rank-load cosine 0.9942.
- Captured observer-light block and observer-heavy inner-MoE timing; measured
  +19.46% and +328.27% observer tax respectively at the representative
  microbatch-8 point. No instrumented wall is used as a clean speedup.
- Measured DeepEP normal-mode communication for global M=4..512. The median
  remains in a narrow 0.257--0.424 ms band, demonstrating fixed/startup cost.
- Completed candidate tournament. No novelty-eligible credible oracle reaches
  8%; prototype gate not met. Final label: `CHARACTERIZATION-ONLY`.

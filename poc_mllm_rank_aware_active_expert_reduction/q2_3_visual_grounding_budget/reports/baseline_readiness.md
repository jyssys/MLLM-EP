# Baseline readiness

Status: **READY**.

- Model: Qwen3-VL-30B-A3B-Instruct `9c4b90e1e4ba969fd3b5378b57d966d725f1b86c`, BF16, 48 layers, E128, K8, d2048/m768.
- Physical contract: TRUE TP2-DP2-EP4-PP1 / DeepEP on GPUs 2,3,6,7.
- Prior bounded evidence: Q1=PHYSICAL_ACTIVE_EXPERT_BOTTLENECK_SUPPORTED_BOUNDED; Q3=RANK_AWARE_STRUCTURED_HEADROOM_SUPPORTED_NONDEPLOYABLE_UPPER; Q2-2=VISUAL_ROUTING_CHANGES_BUT_NOT_PATH_IMPORTANCE.
- This run is instrumented semantic evidence. TPOT, E2E, and physical saving were not measured.

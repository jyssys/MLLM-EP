# Q2-2 baseline readiness

Status: **READY**

- Frozen model revision: `9c4b90e1e4ba969fd3b5378b57d966d725f1b86c`; BF16, d=2048, m=768, E=128, K=8, 48 layers.
- Physical scope: GPUs 2,3,6,7 only; `TRUE_TP2_DP2_EP4_PP1` with `deepep_high_throughput`.
- Prior bounded result is preserved: Q1/Q3 passed while attention-VDS Q2 failed, so Q2-2 tests a new causal target rather than reusing the failed proxy.
- This run is instrumented causal/functional evidence. It is not clean timing and makes no TPOT/E2E claim.

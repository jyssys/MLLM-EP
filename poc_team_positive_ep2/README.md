# TEAM positive-control -> EP2 residual PoC

This workspace follows `team_positive_control_ep2_reproduction_residual_poc_spec.md`.

The gate order is strict:

1. official SDAR baseline vs official TEAM on one H100;
2. semantics-preserving true EP2 only after Stage A passes;
3. residual bottleneck profiling only after the EP2 positive control;
4. candidate oracles derived only from measured residual mass.

Physical GPU policy:

- Stage A: `CUDA_VISIBLE_DEVICES=6`
- Stage B onward: `CUDA_VISIBLE_DEVICES=6,7`
- no other GPU is permitted

Large checkpoint files and observer-heavy raw traces are intentionally excluded
from git. Compact evidence, analysis, reports, and reproduction scripts are kept.

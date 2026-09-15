# RefineGEMM PoC

This directory contains the exact-route, owner-local expert-kernel study for
LLaDA2.0-Flash 100B EP4. The final decision is `NO-GEMM-HEADROOM`: refinement
strongly changes the per-expert row distribution, but an actual two-subgroup
hybrid never beats the strongest whole-invocation kernel and the generous
perfect shape-removal ceiling is below 5% request E2E.

Start with [reports/final_decision.md](reports/final_decision.md). Machine
evidence is in the top-level CSV/Parquet files; the six `kernel2_*` directories
contain the three independent GPU restarts for dense and post-compaction replay.


# LLaDA2.0-Flash EP4 Discovery PoC

Discovery-first analysis of logical refinement progress versus physical sparse-MoE
execution on the validated LLaDA2.0-Flash 100B dense-TP4/routed-EP4 substrate.

The performance baseline is submitted batch 32, generation budget 32, block length
32, and static `mini_batch_size=32`. Clean runs and observer-heavy traces are kept
strictly separate.

Final status: **`CHARACTERIZATION-SIGNAL`**. Row-shape features robustly improve
expert-latency prediction, but the p99-trimmed perfect fragmentation oracle is
only 2.85--3.31% clean request E2E and the post-compaction residual only
1.13--2.12%. No live method was implemented after the oracle gate failed.

- Main report: `../poc_flashvep/reports/dllm_moe_ep_discovery_poc.md`
- Final decision: `reports/final_decision.md`
- Reproduce analysis: `python scripts/analyze_discovery.py && python scripts/build_supplemental_analysis.py`
- Validate invariants: `python tests/check_analysis_invariants.py`
- Runtime instrumentation: `patches/0001-instrument-LLaDA2-EP4-refinement-discovery-traces.patch`

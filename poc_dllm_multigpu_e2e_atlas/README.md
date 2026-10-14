# dLLM multi-GPU E2E atlas

Verdict: **GENERIC_ONLY**.

The campaign found a real local-copy cost in native TP4.
A layer KV view triggered a full monolithic-storage clone.
One exact compact-copy control reduced held-out B1 E2E by 39.54%
[37.33%, 41.54%] across 24 requests. It did not change the parallel architecture.

- [Report](report.md): measurements, limits, correctness, and decision.
- [Timing contract](timing_contract.md): clocks, lanes, boundaries, and gates.
- [Prior boundary](prior_overlap.md): existing profiling, graph, and serving work.
- [Reproduction](reproduce.md): commands and evidence map.
- [Final verdict](results/final_verdict.json): machine-readable summary.

Only physical GPUs 4, 5, 6, and 7 ran experiments.
Full fine-grained critical-path attribution remains PARTIAL.
Official benchmark accuracy and alternative-layout throughput are NOT_RUN.
No EP All-to-All exists in the native EP1 baseline.

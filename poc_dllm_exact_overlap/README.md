# LLaDA2.0-Flash Exact Overlap Discovery PoC

This directory contains the isolated true-EP4 study of exact, resource-
complementary work that the current runtime serializes. Stage 1 builds a
generic dependency/resource/concurrency atlas. Stage 2 is entered only if the
generic gate is weak or refinement phase may materially change overlap
efficiency.

Primary hardware is physical H100 GPUs 0--3 only. Primary topology is dense
TP4 plus routed EP4 with DeepEP dispatch, owner-local fused expert execution,
and reverse combine.

Final result: `NO-OVERLAP-SIGNAL`. The strongest credible direct-request exact
oracle is 3.50%. A legal independent-wave three-stage pipeline has a measured
4.35--4.59% service/throughput upper bound, but cannot be converted into
same-request latency because denoising waves are causally sequential and the
best-static runtime uses one full physical wave. See `reports/final_decision.md`
and the main report for evidence boundaries.
